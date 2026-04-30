#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage2_d7_4b_stab
#SBATCH --output=slurm_logs/stage2_d7_4b_stab_%j.out
#SBATCH --error=slurm_logs/stage2_d7_4b_stab_%j.err
#SBATCH --mail-user=semani@purdue.edu
#SBATCH --mail-type=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p slurm_logs docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

python -u scripts/stage2_analyze_sae_features.py \
    --feature-dir results/stage2/sae_features \
    --model-key gemma3_4b \
    --tasks infer_property infer_subtype \
    --layer 22 \
    --sae-ids layer_22_width_16k_l0_small layer_22_width_262k_l0_small \
    --top-k 128 \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --probe-reports docs/sae_probe_4b_l22_16k_s1.json docs/sae_probe_4b_l22_262k_s1.json \
    --output docs/sae_feature_stability_4b_l22_s1.json

echo "Finished: $(date)"
