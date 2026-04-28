#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage2_27b_pilot
#SBATCH --output=slurm_logs/stage2_27b_pilot_%j.out
#SBATCH --error=slurm_logs/stage2_27b_pilot_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/pilots/extract_27b_L30_h4

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

OUT_DIR=results/stage2/pilots/extract_27b_L30_h4

python scripts/stage2_extract.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --layers 30 \
    --batch-size 16 \
    --n-devices 2 \
    --n-ctx 4096 \
    --height 4 \
    --limit 16 \
    --load-mode no-processing \
    --out-dir "$OUT_DIR"

python scripts/validate_activations.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --output docs/stage2_pilot_validation_27b_infer_property.json \
    --hf-cache "$HF_HOME" \
    --n-ctx 4096 \
    --height 4 \
    --limit 16 \
    --layers 30 \
    --activation-dir "$OUT_DIR" \
    --strict

python scripts/stage2_extract.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --layers 30 \
    --batch-size 16 \
    --n-devices 2 \
    --n-ctx 4096 \
    --height 4 \
    --limit 16 \
    --load-mode no-processing \
    --out-dir "$OUT_DIR"

python scripts/validate_activations.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --output docs/stage2_pilot_validation_27b_infer_subtype.json \
    --hf-cache "$HF_HOME" \
    --n-ctx 4096 \
    --height 4 \
    --limit 16 \
    --layers 30 \
    --activation-dir "$OUT_DIR" \
    --strict

nvidia-smi
echo "Finished: $(date)"
