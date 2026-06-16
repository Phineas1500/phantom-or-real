#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen_erasure
#SBATCH --output=slurm_logs/qwen_erasure_%j.out
#SBATCH --error=slurm_logs/qwen_erasure_%j.err

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

QWEN_ERASURE_TASK="${QWEN_ERASURE_TASK:-infer_property}"
QWEN_ERASURE_JSONL="${QWEN_ERASURE_JSONL:-results/full/with_errortype/qwen35_27b_infer_property.jsonl}"
QWEN_ERASURE_LAYERS="${QWEN_ERASURE_LAYERS:-16,31,40,45,53}"
QWEN_ERASURE_CONDITIONS="${QWEN_ERASURE_CONDITIONS:-baseline,erase_raw,erase_orthogonal,erase_gaussian}"
QWEN_ERASURE_HEIGHTS="${QWEN_ERASURE_HEIGHTS:-3,4}"
QWEN_ERASURE_PER_HEIGHT_LABEL="${QWEN_ERASURE_PER_HEIGHT_LABEL:-4}"
QWEN_ERASURE_SAMPLES_PER_ROW="${QWEN_ERASURE_SAMPLES_PER_ROW:-8}"
QWEN_ERASURE_TEMPERATURE="${QWEN_ERASURE_TEMPERATURE:-0.7}"
QWEN_ERASURE_MAX_NEW_TOKENS="${QWEN_ERASURE_MAX_NEW_TOKENS:-96}"
QWEN_ERASURE_ROW_SHARD="${QWEN_ERASURE_ROW_SHARD:-}"
QWEN_ERASURE_DO_SAMPLE="${QWEN_ERASURE_DO_SAMPLE:-1}"

if [ -z "${QWEN_ERASURE_SUFFIX:-}" ]; then
    if [ -n "$QWEN_ERASURE_ROW_SHARD" ]; then
        SHARD_TAG="${QWEN_ERASURE_ROW_SHARD/\//of}"
        QWEN_ERASURE_SUFFIX="qwen35_subspace_erasure_27b_property_sampled_k8_shard${SHARD_TAG}"
    else
        QWEN_ERASURE_SUFFIX="qwen35_subspace_erasure_27b_property_sampled_k8"
    fi
fi

QWEN_SAMPLE_ARGS=()
if [ "$QWEN_ERASURE_DO_SAMPLE" = "1" ]; then
    QWEN_SAMPLE_ARGS=(--do-sample)
fi

QWEN_SHARD_ARGS=()
if [ -n "$QWEN_ERASURE_ROW_SHARD" ]; then
    QWEN_SHARD_ARGS=(--row-shard "$QWEN_ERASURE_ROW_SHARD")
fi

python -u scripts/stage2_qwen_subspace_erasure.py \
    --jsonl "$QWEN_ERASURE_JSONL" \
    --model Qwen/Qwen3.5-27B \
    --model-key qwen35_27b \
    --task "$QWEN_ERASURE_TASK" \
    --layers "$QWEN_ERASURE_LAYERS" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights "$QWEN_ERASURE_HEIGHTS" \
    --per-height-label "$QWEN_ERASURE_PER_HEIGHT_LABEL" \
    --samples-per-row "$QWEN_ERASURE_SAMPLES_PER_ROW" \
    --conditions "$QWEN_ERASURE_CONDITIONS" \
    --max-new-tokens "$QWEN_ERASURE_MAX_NEW_TOKENS" \
    --temperature "$QWEN_ERASURE_TEMPERATURE" \
    ${QWEN_SAMPLE_ARGS[@]+"${QWEN_SAMPLE_ARGS[@]}"} \
    ${QWEN_SHARD_ARGS[@]+"${QWEN_SHARD_ARGS[@]}"} \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --device-map auto \
    --disable-thinking \
    --out-jsonl "results/stage2/erasure/${QWEN_ERASURE_SUFFIX}.jsonl" \
    --direction-output "results/stage2/erasure/${QWEN_ERASURE_SUFFIX}_directions.npz" \
    --output "docs/${QWEN_ERASURE_SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
