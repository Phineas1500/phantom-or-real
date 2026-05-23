#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_27b_infer
#SBATCH --output=slurm_logs/qwen35_27b_infer_%j.out
#SBATCH --error=slurm_logs/qwen35_27b_infer_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/full/with_errortype

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export BD_PATH="${BD_PATH:-/scratch/scholar/$USER/beyond-deduction}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
PORT="${PORT:-8000}"
SERVER_LOG="slurm_logs/qwen35_27b_vllm_${SLURM_JOB_ID}.log"
TASK_ARGS=(${TASKS:-property ontology})
HEIGHT_ARGS=(${HEIGHTS:-1 2 3 4})
MODEL_SLUG="${MODEL_SLUG:-qwen35_27b}"
OUTPUT_DIR="${OUTPUT_DIR:-results/full/with_errortype}"
MAKE_SPLITS="${MAKE_SPLITS:-auto}"
LIMIT_ARGS=()
if [[ -n "${SKIP:-}" ]]; then
    LIMIT_ARGS+=(--skip "$SKIP")
fi
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS+=(--limit "$LIMIT")
fi
mkdir -p "$OUTPUT_DIR"

MODEL="$MODEL" PORT="$PORT" TENSOR_PARALLEL_SIZE=2 bash scripts/serve_qwen35_27b.sh > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

python - "$PORT" <<'PY'
import sys
import time
import requests

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"vLLM ready at {url}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        print(f"waiting for vLLM: {exc}", flush=True)
    time.sleep(10)
raise SystemExit(f"vLLM did not become ready at {url}")
PY

python scripts/run_inference.py \
    --examples-dir data/full \
    --tasks "${TASK_ARGS[@]}" \
    --heights "${HEIGHT_ARGS[@]}" \
    --model "$MODEL" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --api-key not-needed \
    --concurrency "${CONCURRENCY:-32}" \
    --max-tokens "${MAX_TOKENS:-512}" \
    --temperature 0 \
    --model-slug "$MODEL_SLUG" \
    --output-dir "$OUTPUT_DIR" \
    "${LIMIT_ARGS[@]}"

if [[ "$MAKE_SPLITS" == "1" || ( "$MAKE_SPLITS" == "auto" && "$MODEL_SLUG" == "qwen35_27b" && "$OUTPUT_DIR" == "results/full/with_errortype" && -z "${SKIP:-}" ) ]]; then
    python scripts/stage2_validate_qwen_outputs.py \
        results/full/with_errortype/qwen35_27b_infer_property.jsonl \
        results/full/with_errortype/qwen35_27b_infer_subtype.jsonl
    python scripts/stage2_make_splits.py \
        --jsonl-dir results/full/with_errortype \
        --output results/stage2/splits.jsonl \
        --summary docs/stage2_splits_summary.json
else
    echo "Skipping split regeneration for shard/custom output: MODEL_SLUG=$MODEL_SLUG OUTPUT_DIR=$OUTPUT_DIR SKIP=${SKIP:-<unset>} MAKE_SPLITS=$MAKE_SPLITS"
fi

echo "Finished: $(date)"
