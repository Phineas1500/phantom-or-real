#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --job-name=sae_recon_l45_mlp
#SBATCH --output=slurm_logs/sae_recon_l45_mlp_%j.out
#SBATCH --error=slurm_logs/sae_recon_l45_mlp_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

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

ACTIVATION_SITE="${ACTIVATION_SITE:-mlp_out_hook}"
SPLIT_FAMILY="${SPLIT_FAMILY:-s1}"
TOP_K="${TOP_K:-128}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-200}"
SAE_ID="${SAE_ID:-layer_45_width_16k_l0_small}"
SAE_RELEASE="${SAE_RELEASE:-gemma-scope-2-27b-it-mlp-all}"
OUTPUT="docs/sae_reconstruction_probe_27b_l45_${ACTIVATION_SITE}_16k_${SPLIT_FAMILY}.json"

for task in infer_property infer_subtype; do
    prefix="results/stage2/activations/gemma3_27b_${task}_L45_${ACTIVATION_SITE}"
    test -f "${prefix}.safetensors"
    test -f "${prefix}.meta.json"
    test -f "${prefix}.example_ids.jsonl"
    feature_prefix="results/stage2/sae_features/gemma3_27b_${task}_L45_${ACTIVATION_SITE}_${SAE_ID}_top${TOP_K}"
    test -f "${feature_prefix}.safetensors"
    test -f "${feature_prefix}.meta.json"
    test -f "${feature_prefix}.example_ids.jsonl"
done

python scripts/stage2_sae_reconstruction_diagnostics.py \
    --activation-dir results/stage2/activations \
    --feature-dir results/stage2/sae_features \
    --out-dir results/stage2/sae_reconstructions \
    --model-key gemma3_27b \
    --tasks infer_property infer_subtype \
    --layer 45 \
    --activation-site "$ACTIVATION_SITE" \
    --sae-release "$SAE_RELEASE" \
    --sae-ids "$SAE_ID" \
    --top-k "$TOP_K" \
    --splits results/stage2/splits.jsonl \
    --split-family "$SPLIT_FAMILY" \
    --output "$OUTPUT" \
    --device cuda:0 \
    --dtype bfloat16 \
    --output-dtype bfloat16 \
    --chunk-size 512 \
    --verify-dense-rows 8 \
    --solver lbfgs \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --seed 20260427

nvidia-smi
echo "Finished: $(date)"
