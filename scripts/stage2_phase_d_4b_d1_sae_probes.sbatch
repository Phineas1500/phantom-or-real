#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_d1_4b_sae
#SBATCH --output=slurm_logs/stage2_d1_4b_sae_%j.out
#SBATCH --error=slurm_logs/stage2_d1_4b_sae_%j.err
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

echo "Node: $(hostname)"
echo "Started: $(date)"

TASKS=(infer_property infer_subtype)
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10

probe_resid() {
    local layer="$1"
    local width="$2"
    local sae_id="layer_${layer}_width_${width}_l0_small"
    for split in s1 s3; do
        local suffix="$split"
        if [ "$split" = "s3" ]; then
            suffix=s3_target_symbol
        fi
        python -u scripts/stage2_probe_sae.py \
            --feature-dir results/stage2/sae_features \
            --model-key gemma3_4b \
            --tasks "${TASKS[@]}" \
            --layer "$layer" \
            --sae-id "$sae_id" \
            --top-k 128 \
            --splits results/stage2/splits.jsonl \
            --split-family "$split" \
            --c-values "$C_VALUES" \
            --bootstrap-samples 1000 \
            --output "docs/sae_probe_4b_l${layer}_${width}_${suffix}.json"
    done
}

probe_mlp_out() {
    for split in s1 s3; do
        local suffix="$split"
        if [ "$split" = "s3" ]; then
            suffix=s3_target_symbol
        fi
        python -u scripts/stage2_probe_sae.py \
            --feature-dir results/stage2/sae_features \
            --activation-site mlp_out_hook \
            --model-key gemma3_4b \
            --tasks "${TASKS[@]}" \
            --layer 22 \
            --sae-id layer_22_width_16k_l0_small \
            --top-k 128 \
            --splits results/stage2/splits.jsonl \
            --split-family "$split" \
            --c-values "$C_VALUES" \
            --bootstrap-samples 1000 \
            --output "docs/sae_probe_4b_l22_mlp_out_hook_16k_${suffix}.json"
    done
}

probe_resid 20 16k
probe_resid 20 262k
probe_resid 22 16k
probe_resid 22 262k
probe_mlp_out

echo "Finished: $(date)"
