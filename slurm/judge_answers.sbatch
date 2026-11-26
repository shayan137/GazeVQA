#!/bin/bash
#SBATCH --job-name=g3vqa_judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Serves the AnyJev judge with vLLM and scores the answers of every model in ANSWERS_DIR.
#
#   mkdir -p logs
#   sbatch --export=ALL,ANSWERS_DIR=outputs/vqa,OUT_DIR=outputs/judge slurm/judge_answers.sbatch
#
# Set EXEC to run inside a container, for example EXEC="apptainer exec --nv image.sif".

set -euo pipefail

ANSWERS_DIR="${ANSWERS_DIR:-outputs/vqa}"
OUT_DIR="${OUT_DIR:-outputs/judge}"
JUDGE_MODEL="${JUDGE_MODEL:-morriszjm/Tacit-9B}"
MODELS="${MODELS:-}"                  # comma-separated model names, default all files in ANSWERS_DIR
PORT="${PORT:-8100}"
EXEC="${EXEC:-}"

mkdir -p logs "$OUT_DIR"
export HF_HOME="${HF_HOME:-$PWD/model_cache/huggingface}"
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

# The judge reads text only. Prefix caching is off because every prompt is different.
$EXEC vllm serve "$JUDGE_MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --max-num-seqs 256 --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    > "logs/judge_server_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!

for _ in $(seq 1 120); do
    kill -0 "$SERVER" 2>/dev/null || { echo "the vLLM server stopped, see logs/"; exit 1; }
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "the vLLM server did not start within 20 minutes"; exit 1; }

$EXEC python scripts/judge_answers.py --server "http://127.0.0.1:${PORT}" --model-path "$JUDGE_MODEL" \
    --answers-dir "$ANSWERS_DIR" --out-dir "$OUT_DIR" ${MODELS:+--models "$MODELS"}
