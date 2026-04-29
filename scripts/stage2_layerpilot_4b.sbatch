#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --job-name=stage2_4b_layers
#SBATCH --output=slurm_logs/stage2_4b_layers_%j.out
#SBATCH --error=slurm_logs/stage2_4b_layers_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

mkdir -p slurm_logs results/stage2/pilots/layer_selection_4b docs

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
nvidia-smi

python scripts/stage2_layer_pilot_4b.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --task infer_property \
  --out-dir results/stage2/pilots/layer_selection_4b \
  --output docs/layer_selection.json \
  --seed 20260428 \
  --per-height 125 \
  --batch-size 32 \
  --n-devices 1 \
  --n-ctx 4096 \
  --load-mode no-processing

echo "Finished: $(date)"