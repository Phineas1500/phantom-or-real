#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=subspace_erasure
#SBATCH --output=slurm_logs/subspace_erasure_%j.out
#SBATCH --error=slurm_logs/subspace_erasure_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/erasure docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH=/scratch/scholar/$USER/beyond-deduction
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

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

ERASURE_TASK="${ERASURE_TASK:-infer_property}"
ERASURE_JSONL="${ERASURE_JSONL:-results/full/with_errortype/gemma3_27b_infer_property.jsonl}"
ERASURE_LAYERS="${ERASURE_LAYERS:-15,30,40,45,53}"
ERASURE_CONDITIONS="${ERASURE_CONDITIONS:-baseline,erase_raw,erase_orthogonal,erase_gaussian}"
ERASURE_HEIGHTS="${ERASURE_HEIGHTS:-3,4}"
ERASURE_PER_HEIGHT_LABEL="${ERASURE_PER_HEIGHT_LABEL:-4}"
ERASURE_SAMPLES_PER_ROW="${ERASURE_SAMPLES_PER_ROW:-1}"
ERASURE_TEMPERATURE="${ERASURE_TEMPERATURE:-0}"
ERASURE_SAMPLE_FLAG="${ERASURE_SAMPLE_FLAG:-}"
ERASURE_MAX_NEW_TOKENS="${ERASURE_MAX_NEW_TOKENS:-96}"
ERASURE_ROW_SHARD="${ERASURE_ROW_SHARD:-}"
ERASURE_SUFFIX="${ERASURE_SUFFIX:-subspace_erasure_27b_property_smoke}"

ERASURE_SHARD_ARGS=()
if [ -n "$ERASURE_ROW_SHARD" ]; then
    ERASURE_SHARD_ARGS=(--row-shard "$ERASURE_ROW_SHARD")
fi

python -u scripts/stage2_subspace_erasure.py \
    --jsonl "$ERASURE_JSONL" \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task "$ERASURE_TASK" \
    --layers "$ERASURE_LAYERS" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights "$ERASURE_HEIGHTS" \
    --per-height-label "$ERASURE_PER_HEIGHT_LABEL" \
    --samples-per-row "$ERASURE_SAMPLES_PER_ROW" \
    --conditions "$ERASURE_CONDITIONS" \
    --max-new-tokens "$ERASURE_MAX_NEW_TOKENS" \
    --temperature "$ERASURE_TEMPERATURE" \
    $ERASURE_SAMPLE_FLAG \
    ${ERASURE_SHARD_ARGS[@]+"${ERASURE_SHARD_ARGS[@]}"} \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl "results/stage2/erasure/${ERASURE_SUFFIX}.jsonl" \
    --direction-output "results/stage2/erasure/${ERASURE_SUFFIX}_directions.npz" \
    --output "docs/${ERASURE_SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
