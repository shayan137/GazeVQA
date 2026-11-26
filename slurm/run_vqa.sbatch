#!/bin/bash
#SBATCH --job-name=g3vqa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Serves one VLM with vLLM on this node and asks it every question with scripts/run_vqa.py.
#
#   mkdir -p logs
#   sbatch --export=ALL,ANSWER_MODEL=Qwen/Qwen3.5-27B,ANSWER_NAME=qwen3.5-27b slurm/run_vqa.sbatch
#
# Larger models need more GPUs: set ANSWER_TP to the number of GPUs and change --gpus-per-node.
# Set EXEC to run inside a container, for example EXEC="apptainer exec --nv image.sif".
#
# The temporal questions send one mp4 per question. vLLM only accepts per-request video settings when the
# server trusts them, so the sampling is fixed at server level: every frame of the file is kept (render_vqa_inputs.py
# already wrote at most 64 frames) and one video is allowed per prompt. A video prompt has about 12k tokens,
# so the context length is 32768.

set -euo pipefail

: "${ANSWER_MODEL:?set ANSWER_MODEL (Hugging Face id or local path)}"
: "${ANSWER_NAME:?set ANSWER_NAME (the name the model is served under)}"
ANSWER_TP="${ANSWER_TP:-1}"
ANSWER_MAX_LEN="${ANSWER_MAX_LEN:-32768}"
MEDIA_ROOT="${MEDIA_ROOT:-}"          # folder with the rendered images and videos, needed for --video-transport file
CONFIG="${CONFIG:-configs/vqa.yaml}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/vqa/${ANSWER_NAME}.csv}"
EXEC="${EXEC:-}"

mkdir -p logs
export HF_HOME="${HF_HOME:-$PWD/model_cache/huggingface}"
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ANSWER_TP - 1))) $EXEC vllm serve "$ANSWER_MODEL" \
    --served-model-name "$ANSWER_NAME" --port 8000 \
    --tensor-parallel-size "$ANSWER_TP" --max-model-len "$ANSWER_MAX_LEN" \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1, "video": 1}' \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    ${MEDIA_ROOT:+--allowed-local-media-path "$MEDIA_ROOT"} \
    > "logs/answer_server_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!

for _ in $(seq 1 180); do
    kill -0 "$SERVER" 2>/dev/null || { echo "the vLLM server stopped, see logs/"; exit 1; }
    curl -sf "http://localhost:8000/v1/models" >/dev/null && break
    sleep 10
done
curl -sf "http://localhost:8000/v1/models" >/dev/null || { echo "the vLLM server did not start within 30 minutes"; exit 1; }

$EXEC python scripts/run_vqa.py --config "$CONFIG" \
    --answer-url http://localhost:8000/v1 --answer-model "$ANSWER_NAME" \
    --output-csv "$OUTPUT_CSV" --resume
