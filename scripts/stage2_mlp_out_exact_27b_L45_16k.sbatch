#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage2_mlp_exact_l45_16k
#SBATCH --output=slurm_logs/stage2_mlp_exact_l45_16k_%j.out
#SBATCH --error=slurm_logs/stage2_mlp_exact_l45_16k_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/sae_features docs

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
export BD_PATH=/scratch/scholar/$USER/beyond-deduction

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

OUT_DIR=results/stage2/activations
FEATURE_DIR=results/stage2/sae_features
LAYER=45
SITE=mlp_out_hook
SAE_RELEASE=gemma-scope-2-27b-it-mlp-all
SAE_ID=layer_45_width_16k_l0_small
HF_SUBFOLDER=mlp_out_all

for task in infer_property infer_subtype; do
    prefix="$OUT_DIR/gemma3_27b_${task}_L${LAYER}_${SITE}"
    if [ ! -f "${prefix}.safetensors" ]; then
        echo "Missing exact MLP-output activations: ${prefix}.safetensors" >&2
        exit 1
    fi
done

COMMON_SAE_ARGS=(
    --out-dir "$FEATURE_DIR"
    --sae-release "$SAE_RELEASE"
    --sae-id "$SAE_ID"
    --hf-repo-id google/gemma-scope-2-27b-it
    --hf-subfolder "$HF_SUBFOLDER"
    --hf-revision main
    --device cuda:0
    --dtype bfloat16
    --chunk-size 512
    --top-k 128
)

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_27b_infer_property_L${LAYER}_${SITE}" \
    "${COMMON_SAE_ARGS[@]}"

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_27b_infer_subtype_L${LAYER}_${SITE}" \
    "${COMMON_SAE_ARGS[@]}"

COMMON_SAE_PROBE_ARGS=(
    --feature-dir "$FEATURE_DIR"
    --activation-site "$SITE"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layer "$LAYER"
    --sae-id "$SAE_ID"
    --top-k 128
    --splits results/stage2/splits.jsonl
    --solver liblinear
    --bootstrap-samples 1000
)

python -u scripts/stage2_probe_sae.py \
    "${COMMON_SAE_PROBE_ARGS[@]}" \
    --split-family s1 \
    --output docs/sae_probe_27b_l45_mlp_out_hook_16k_s1.json

python -u scripts/stage2_probe_sae.py \
    "${COMMON_SAE_PROBE_ARGS[@]}" \
    --split-family s3 \
    --output docs/sae_probe_27b_l45_mlp_out_hook_16k_s3_target_symbol.json

nvidia-smi
echo "Finished: $(date)"
