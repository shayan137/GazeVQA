#!/bin/bash
#SBATCH --job-name=g3vqa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Serves the answering model (GPU 0..TP-1) and the judge (next GPU) with vLLM
# on one node, then runs scripts/run_vqa.py against both.
#
#   sbatch --export=ALL,ANSWER_MODEL=Qwen/Qwen3.5-27B,ANSWER_NAME=qwen3.5-27b slurm/run_vqa.sbatch
#
# A model too large to share the node with the judge (Qwen3.5-122B-A10B on
# 4 GPUs): set ANSWER_TP=4 NO_JUDGE=1 and --gpus-per-node=4, then judge the
# answers with scripts/rejudge_vqa.py.
#
# Set EXEC to run inside a container, e.g. EXEC="apptainer exec --nv image.sif".

set -euo pipefail

: "${ANSWER_MODEL:?set ANSWER_MODEL (HF id or local path)}"
: "${ANSWER_NAME:?set ANSWER_NAME (served model name)}"
ANSWER_TP="${ANSWER_TP:-1}"
ANSWER_MAX_LEN="${ANSWER_MAX_LEN:-8192}"
JUDGE_MODEL="${JUDGE_MODEL:-/path/to/Qwen3.8-27B}"
JUDGE_NAME="${JUDGE_NAME:-qwen3.8-27b}"
NO_JUDGE="${NO_JUDGE:-0}"
CONFIG="${CONFIG:-configs/vqa.yaml}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/vqa/${ANSWER_NAME}.csv}"
EXEC="${EXEC:-}"

mkdir -p logs
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

wait_for_server() {
    local port=$1 pid=$2
    for _ in $(seq 1 180); do
        kill -0 "$pid" 2>/dev/null || { echo "vLLM server on port $port died, see logs/"; exit 1; }
        curl -sf "http://localhost:${port}/v1/models" >/dev/null && return 0
        sleep 10
    done
    echo "vLLM server on port $port not ready after 30 min"; exit 1
}

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ANSWER_TP - 1))) $EXEC vllm serve "$ANSWER_MODEL" \
    --served-model-name "$ANSWER_NAME" --port 8000 \
    --tensor-parallel-size "$ANSWER_TP" --max-model-len "$ANSWER_MAX_LEN" \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    > "logs/answer_server_${SLURM_JOB_ID}.log" 2>&1 &
PIDS+=($!)

JUDGE_ARGS=(--no-judge)
if [[ "$NO_JUDGE" != "1" ]]; then
    CUDA_VISIBLE_DEVICES=$ANSWER_TP $EXEC vllm serve "$JUDGE_MODEL" \
        --served-model-name "$JUDGE_NAME" --port 8001 \
        --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code \
        > "logs/judge_server_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
    JUDGE_ARGS=(--judge-url http://localhost:8001/v1 --judge-model "$JUDGE_NAME")
    wait_for_server 8001 "${PIDS[1]}"
fi
wait_for_server 8000 "${PIDS[0]}"

$EXEC python scripts/run_vqa.py --config "$CONFIG" \
    --answer-url http://localhost:8000/v1 --answer-model "$ANSWER_NAME" \
    "${JUDGE_ARGS[@]}" --output-csv "$OUTPUT_CSV" --resume
