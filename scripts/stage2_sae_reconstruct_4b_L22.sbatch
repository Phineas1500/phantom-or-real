#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage2_sae_recon_4b_l22
#SBATCH --output=slurm_logs/stage2_sae_recon_4b_l22_%j.out
#SBATCH --error=slurm_logs/stage2_sae_recon_4b_l22_%j.err
#SBATCH --mail-user=semani@purdue.edu
#SBATCH --mail-type=ALL


set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p slurm_logs results/stage2/sae_reconstructions docs

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

COMMON_ARGS=(
    --activation-dir results/stage2/activations
    --feature-dir results/stage2/sae_features
    --out-dir results/stage2/sae_reconstructions
    --model-key gemma3_4b
    --tasks infer_property infer_subtype
    --layer 22
    --sae-release gemma-scope-2-4b-it-res-all
    --sae-ids layer_22_width_16k_l0_small layer_22_width_262k_l0_small
    --top-k 128
    --splits results/stage2/splits.jsonl
    --device cuda:0
    --dtype bfloat16
    --output-dtype bfloat16
    --chunk-size 512
    --verify-dense-rows 8
    --solver lbfgs
    --bootstrap-samples 1000
    --seed 20260427
)

python -u scripts/stage2_sae_reconstruction_diagnostics.py \
    "${COMMON_ARGS[@]}" \
    --split-family s1 \
    --output docs/sae_reconstruction_probe_4b_l22_s1.json

python -u scripts/stage2_sae_reconstruction_diagnostics.py \
    "${COMMON_ARGS[@]}" \
    --reuse-existing \
    --split-family s3 \
    --output docs/sae_reconstruction_probe_4b_l22_s3_target_symbol.json

nvidia-smi
echo "Finished: $(date)"
