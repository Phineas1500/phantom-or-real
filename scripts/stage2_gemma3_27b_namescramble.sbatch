#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=ns27b_all
#SBATCH --output=slurm_logs/ns27b_all_%j.out
#SBATCH --error=slurm_logs/ns27b_all_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs data/stage2/namescramble_27b results/stage2/namescramble_infer_27b results/stage2/activations_namescramble_27b docs

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

MODEL="${MODEL:-google/gemma-3-27b-it}"
MODEL_KEY="${MODEL_KEY:-gemma3_27b}"
PORT="${PORT:-8000}"
PER_HEIGHT="${PER_HEIGHT:-250}"
CONDITIONS=(${CONDITIONS:-nonce natural})
TASKS=(${TASKS:-infer_property infer_subtype})
HEIGHTS=(${HEIGHTS:-1 2 3 4})
DATA_DIR="${DATA_DIR:-data/stage2/namescramble_27b}"
INFER_DIR="${INFER_DIR:-results/stage2/namescramble_infer_27b}"
ACT_DIR="${ACT_DIR:-results/stage2/activations_namescramble_27b}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_EXTRACTION="${SKIP_EXTRACTION:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

if [[ "$SKIP_PREP" != "1" ]]; then
  python scripts/stage2_diagnostics_namescramble.py     --input-dir results/full/with_errortype     --model-name gemma3-27b     --tasks "${TASKS[@]}"     --heights "${HEIGHTS[@]}"     --per-height "$PER_HEIGHT"     --conditions "${CONDITIONS[@]}"     --output-dir "$DATA_DIR"     --nonce-pool-out docs/stage2/nonce_pool_27b.json     --diagnostics-out docs/namescramble_27b_prepared.json
fi

SERVER_PID=""
cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_server EXIT

if [[ "$SKIP_INFERENCE" != "1" ]]; then
  SERVER_LOG="slurm_logs/ns27b_vllm_${SLURM_JOB_ID}.log"
  MODEL="$MODEL" PORT="$PORT" TENSOR_PARALLEL_SIZE=2 bash scripts/serve_gemma3_27b.sh > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  python - "$PORT" <<'WAITPY'
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
WAITPY

  for cond in "${CONDITIONS[@]}"; do
    for task in "${TASKS[@]}"; do
      for h in "${HEIGHTS[@]}"; do
        echo "Running cond=${cond} task=${task} h=${h}"
        python scripts/stage2_namescramble_infer.py           --input-jsonl "${DATA_DIR}/${cond}/${task}_h${h}.jsonl"           --output-jsonl "${INFER_DIR}/${cond}/${task}_h${h}.jsonl"           --model "$MODEL"           --base-url "http://127.0.0.1:${PORT}/v1"           --api-key not-needed           --concurrency "${CONCURRENCY:-32}"           --max-tokens "${MAX_TOKENS:-512}"           --temperature 0.0           --max-attempts 4
      done
    done
  done
fi

cleanup_server
trap - EXIT

python scripts/stage2_namescramble_combine.py   --input-dir "$INFER_DIR"   --conditions "${CONDITIONS[@]}"   --tasks "${TASKS[@]}"   --heights "${HEIGHTS[@]}"   --summary docs/namescramble_27b_combine_summary.json

if [[ "$SKIP_EXTRACTION" != "1" ]]; then
  for cond in "${CONDITIONS[@]}"; do
    for task in "${TASKS[@]}"; do
      echo "Extracting cond=${cond} task=${task}"
      python scripts/stage2_extract.py         --jsonl "${INFER_DIR}/${cond}/${task}.jsonl"         --model "$MODEL"         --model-key "${MODEL_KEY}_${cond}"         --task "$task"         --layers 45         --out-dir "$ACT_DIR"         --batch-size "${EXTRACT_BATCH_SIZE:-16}"         --n-devices 2         --n-ctx 4096         --drop-parse-failed         --load-mode no-processing
    done
  done
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  python scripts/stage2_namescramble_eval_raw_probe.py     --splits results/stage2/splits.jsonl     --activation-dir results/stage2/activations     --namescramble-infer-dir "$INFER_DIR"     --namescramble-activations-dir "$ACT_DIR"     --model-key "$MODEL_KEY"     --tasks "${TASKS[@]}"     --layers 45     --conditions "${CONDITIONS[@]}"     --split-family s1     --output docs/namescramble_27b_l45_raw_probe_s1.json
fi

nvidia-smi
echo "Finished: $(date)"
