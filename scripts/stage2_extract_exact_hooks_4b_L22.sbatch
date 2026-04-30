#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_exact_hooks_4b_l22
#SBATCH --output=slurm_logs/stage2_exact_hooks_4b_l22_%j.out
#SBATCH --error=slurm_logs/stage2_exact_hooks_4b_l22_%j.err
#SBATCH --mail-user=semani@purdue.edu
#SBATCH --mail-type=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
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

python -u scripts/stage2_extract_exact_transcoder_hooks.py \
    --property-jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
    --subtype-jsonl results/full/with_errortype/gemma3_4b_infer_subtype.jsonl \
    --model google/gemma-3-4b-it \
    --model-key gemma3_4b \
    --layer 22 \
    --batch-size 32 \
    --n-devices 1 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --output-dtype bfloat16 \
    --out-dir results/stage2/activations

nvidia-smi
echo "Finished: $(date)"
