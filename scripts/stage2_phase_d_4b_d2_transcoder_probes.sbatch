#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage2_d2_4b_tc
#SBATCH --output=slurm_logs/stage2_d2_4b_tc_%j.out
#SBATCH --error=slurm_logs/stage2_d2_4b_tc_%j.err
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

for width in 16k 262k; do
    sae_id="layer_22_width_${width}_l0_small_affine"
    for split in s1 s3; do
        suffix="$split"
        if [ "$split" = "s3" ]; then
            suffix=s3_target_symbol
        fi
        python -u scripts/stage2_probe_sae.py \
            --feature-dir results/stage2/sae_features \
            --activation-site mlp_in_weighted \
            --model-key gemma3_4b \
            --tasks "${TASKS[@]}" \
            --layer 22 \
            --sae-id "$sae_id" \
            --top-k 128 \
            --splits results/stage2/splits.jsonl \
            --split-family "$split" \
            --c-values "$C_VALUES" \
            --bootstrap-samples 1000 \
            --output "docs/transcoder_probe_4b_l22_${width}_affine_exact_${suffix}.json"
    done
done

echo "Finished: $(date)"
