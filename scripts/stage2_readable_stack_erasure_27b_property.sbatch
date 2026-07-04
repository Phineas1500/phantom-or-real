#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=readable_stack_erasure
#SBATCH --output=slurm_logs/readable_stack_erasure_%j.out
#SBATCH --error=slurm_logs/readable_stack_erasure_%j.err

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
nvidia-smi

STACKERASE_ROW_SHARD="${STACKERASE_ROW_SHARD:-0/2}"
SHARD_TAG="${STACKERASE_ROW_SHARD/\//of}"
STACKERASE_SUFFIX="${STACKERASE_SUFFIX:-readable_stack_erasure_27b_property_k8_shard${SHARD_TAG}}"

python -u scripts/stage2_subspace_erasure.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --layers "${STACKERASE_LAYERS:-15,30,40,45,53}" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights 3,4 \
    --per-height-label 4 \
    --samples-per-row "${STACKERASE_SAMPLES_PER_ROW:-8}" \
    --conditions "${STACKERASE_CONDITIONS:-baseline,erase_raw,erase_readable_stack,erase_random_stack}" \
    --random-stack-draws "${STACKERASE_RANDOM_DRAWS:-3}" \
    --random-stack-seed "${STACKERASE_RANDOM_SEED:-20260704}" \
    --inlp-npz "${STACKERASE_INLP_NPZ:-results/stage2/erasure/inlp_direction_stacks_27b_property_5layer.npz}" \
    --stack-rank "${STACKERASE_STACK_RANK:-9}" \
    --max-new-tokens 96 \
    --temperature 0.7 \
    --do-sample \
    --row-shard "$STACKERASE_ROW_SHARD" \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl "results/stage2/erasure/${STACKERASE_SUFFIX}.jsonl" \
    --direction-output "results/stage2/erasure/${STACKERASE_SUFFIX}_directions.npz" \
    --stack-output "results/stage2/erasure/${STACKERASE_SUFFIX}_stacks.npz" \
    --output "docs/${STACKERASE_SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
