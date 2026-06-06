#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_atp27b
#SBATCH --output=slurm_logs/stage2_atp27b_%j.out
#SBATCH --error=slurm_logs/stage2_atp27b_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/atp docs

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

PATCH_DIRECTION="${ATP_PATCH_DIRECTION:-corrupt_to_clean}"
LAYERS="${ATP_LAYERS:-35,40,45,50}"
LANDMARKS="${ATP_LANDMARKS:-last_prompt}"
PAIR_LIMIT="${ATP_PAIR_LIMIT:-4}"
PAIR_SEED="${ATP_PAIR_SEED:-20260501}"
TOP_K="${ATP_TOP_K:-3}"
SITE_TAG="l${LAYERS//,/_l}_${LANDMARKS//,/_}"
SUFFIX="atp_rank_validate_27b_${SITE_TAG}_property_${PATCH_DIRECTION}_pilot"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PATCH_DIRECTION=${PATCH_DIRECTION}"
echo "LAYERS=${LAYERS}"
echo "LANDMARKS=${LANDMARKS}"
echo "PAIR_LIMIT=${PAIR_LIMIT}"
echo "TOP_K=${TOP_K}"
nvidia-smi

python -u scripts/stage2_atp_rank_validate.py \
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
    --patch-direction "${PATCH_DIRECTION}" \
    --top-k-sites "${TOP_K}" \
    --noise-repeats 3 \
    --noise-seed 20260606 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --positive-control-report docs/positive_control_format_gemma3_27b_l45.json \
    --out-jsonl "results/stage2/atp/${SUFFIX}.jsonl" \
    --output "docs/${SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
