#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_das27b
#SBATCH --output=slurm_logs/stage2_das27b_%j.out
#SBATCH --error=slurm_logs/stage2_das27b_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/das docs

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

PATCH_DIRECTION="${DAS_PATCH_DIRECTION:-clean_to_corrupt}"
LAYERS="${DAS_LAYERS:-45}"
LANDMARKS="${DAS_LANDMARKS:-last_prompt}"
RANKS="${DAS_RANKS:-1,2,4}"
PAIR_LIMIT="${DAS_PAIR_LIMIT:-8}"
PAIR_SEED="${DAS_PAIR_SEED:-20260501}"
SITE_TAG="l${LAYERS//,/_l}_${LANDMARKS//,/_}"
SUFFIX="das_subspace_27b_${SITE_TAG}_property_${PATCH_DIRECTION}_pilot"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PATCH_DIRECTION=${PATCH_DIRECTION}"
echo "LAYERS=${LAYERS}"
echo "LANDMARKS=${LANDMARKS}"
echo "RANKS=${RANKS}"
nvidia-smi

python -u scripts/stage2_das_subspace_interchange.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --split test \
    --pair-limit "${PAIR_LIMIT}" \
    --pair-seed "${PAIR_SEED}" \
    --layers "${LAYERS}" \
    --landmarks "${LANDMARKS}" \
    --ranks "${RANKS}" \
    --basis-scope leave_one_out \
    --patch-direction "${PATCH_DIRECTION}" \
    --intervention-modes das_subspace,orthogonal_subspace,random_subspace,matched_gaussian,source \
    --noise-repeats 3 \
    --noise-seed 20260605 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --positive-control-report docs/positive_control_format_gemma3_27b_l45.json \
    --out-jsonl "results/stage2/das/${SUFFIX}.jsonl" \
    --output "docs/${SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
