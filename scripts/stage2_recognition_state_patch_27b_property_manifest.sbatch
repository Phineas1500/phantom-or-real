#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=recog_state_patch
#SBATCH --output=slurm_logs/recog_state_patch_%j.out
#SBATCH --error=slurm_logs/recog_state_patch_%j.err

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

PATCH_LAYERS="${PATCH_LAYERS:-30,40,45}"
PATCH_CONDITIONS="${PATCH_CONDITIONS:-baseline,patch_recognition,patch_shuffled,noise_matched}"
PATCH_SAMPLES_PER_ROW="${PATCH_SAMPLES_PER_ROW:-1}"
PATCH_TEMPERATURE="${PATCH_TEMPERATURE:-0}"
PATCH_SAMPLE_FLAG="${PATCH_SAMPLE_FLAG:-}"
PATCH_MAX_NEW_TOKENS="${PATCH_MAX_NEW_TOKENS:-96}"
PATCH_SUFFIX="${PATCH_SUFFIX:-recognition_state_patch_27b_property_manifest}"

python -u scripts/stage2_recognition_state_patch.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --prefix-trajectory-jsonl results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl \
    --prefix-checkpoint 0 \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --layers "$PATCH_LAYERS" \
    --conditions "$PATCH_CONDITIONS" \
    --samples-per-row "$PATCH_SAMPLES_PER_ROW" \
    --max-new-tokens "$PATCH_MAX_NEW_TOKENS" \
    --temperature "$PATCH_TEMPERATURE" \
    $PATCH_SAMPLE_FLAG \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl "results/stage2/erasure/${PATCH_SUFFIX}.jsonl" \
    --output "docs/${PATCH_SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
