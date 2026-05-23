#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_27b_extract_l45
#SBATCH --output=slurm_logs/qwen35_27b_extract_l45_%j.out
#SBATCH --error=slurm_logs/qwen35_27b_extract_l45_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/activations

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

python scripts/stage2_validate_qwen_outputs.py \
    results/full/with_errortype/qwen35_27b_infer_property.jsonl \
    results/full/with_errortype/qwen35_27b_infer_subtype.jsonl

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
LAYERS="${LAYERS:-45}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

python scripts/stage2_extract_hf.py \
    --jsonl results/full/with_errortype/qwen35_27b_infer_property.jsonl \
    --model "$MODEL" \
    --model-key qwen35_27b \
    --layers "$LAYERS" \
    --batch-size "${BATCH_SIZE:-8}" \
    --n-ctx 4096 \
    --drop-parse-failed \
    --disable-thinking \
    --out-dir results/stage2/activations \
    "${LIMIT_ARGS[@]}"

python scripts/stage2_extract_hf.py \
    --jsonl results/full/with_errortype/qwen35_27b_infer_subtype.jsonl \
    --model "$MODEL" \
    --model-key qwen35_27b \
    --layers "$LAYERS" \
    --batch-size "${BATCH_SIZE:-8}" \
    --n-ctx 4096 \
    --drop-parse-failed \
    --disable-thinking \
    --out-dir results/stage2/activations \
    "${LIMIT_ARGS[@]}"

echo "Finished: $(date)"
