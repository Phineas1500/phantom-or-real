#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen_patch_c2cln
#SBATCH --output=slurm_logs/qwen_patch_c2cln_%j.out
#SBATCH --error=slurm_logs/qwen_patch_c2cln_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/patching docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH="${BD_PATH:-/scratch/scholar/$USER/beyond-deduction}"
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

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
TASK="${TASK:-infer_subtype}"
JSONL="${JSONL:-results/full/with_errortype/qwen35_27b_infer_subtype.jsonl}"

python -u scripts/stage2_qwen_patch_hf.py \
    --jsonl "$JSONL" \
    --model "$MODEL" \
    --model-key qwen35_27b \
    --task "$TASK" \
    --splits results/stage2/splits.jsonl \
    --split-family "${SPLIT_FAMILY:-s1}" \
    --split "${SPLIT:-test}" \
    --pair-limit "${PAIR_LIMIT:-7}" \
    --pair-seed "${PAIR_SEED:-20260523}" \
    --layers "${LAYERS:-35,40,45}" \
    --landmarks "${LANDMARKS:-last_prompt}" \
    --patch-direction corrupt_to_clean \
    --patch-modes corrupt,noise \
    --noise-seed "${NOISE_SEED:-20260525}" \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --device-map auto \
    --disable-thinking \
    --out-jsonl "results/stage2/patching/qwen35_27b_${TASK}_corrupt_to_clean_margin_pilot.jsonl" \
    --output "docs/qwen35_27b_${TASK}_corrupt_to_clean_patching_margin_pilot.json"

nvidia-smi
echo "Finished: $(date)"
