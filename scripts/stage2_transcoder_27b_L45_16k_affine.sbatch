#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_tc_l45_16k
#SBATCH --output=slurm_logs/stage2_tc_l45_16k_%j.out
#SBATCH --error=slurm_logs/stage2_tc_l45_16k_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/activations results/stage2/sae_features docs

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

OUT_DIR=results/stage2/activations
FEATURE_DIR=results/stage2/sae_features
LAYER=45
SITE=mlp_in
HOOK_TEMPLATE='blocks.{layer}.ln2.hook_normalized'
TRANSCODER_RELEASE=gemma-scope-2-27b-it-transcoders-all
TRANSCODER_ID=layer_45_width_16k_l0_small_affine
HF_SUBFOLDER=transcoder_all

PROPERTY_PREFIX="$OUT_DIR/gemma3_27b_infer_property_L45_${SITE}"
SUBTYPE_PREFIX="$OUT_DIR/gemma3_27b_infer_subtype_L45_${SITE}"

if [ ! -f "${PROPERTY_PREFIX}.safetensors" ]; then
    python scripts/stage2_extract.py \
        --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
        --model google/gemma-3-27b-it \
        --model-key gemma3_27b \
        --task infer_property \
        --layers "$LAYER" \
        --activation-site "$SITE" \
        --hook-template "$HOOK_TEMPLATE" \
        --batch-size 32 \
        --n-devices 2 \
        --n-ctx 4096 \
        --load-mode no-processing \
        --out-dir "$OUT_DIR"
else
    echo "Reusing ${PROPERTY_PREFIX}.safetensors"
fi

if [ ! -f "${SUBTYPE_PREFIX}.safetensors" ]; then
    python scripts/stage2_extract.py \
        --jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
        --model google/gemma-3-27b-it \
        --model-key gemma3_27b \
        --task infer_subtype \
        --layers "$LAYER" \
        --activation-site "$SITE" \
        --hook-template "$HOOK_TEMPLATE" \
        --batch-size 32 \
        --n-devices 2 \
        --n-ctx 4096 \
        --load-mode no-processing \
        --out-dir "$OUT_DIR"
else
    echo "Reusing ${SUBTYPE_PREFIX}.safetensors"
fi

COMMON_TRANSCODER_ARGS=(
    --out-dir "$FEATURE_DIR"
    --sae-release "$TRANSCODER_RELEASE"
    --sae-id "$TRANSCODER_ID"
    --hf-repo-id google/gemma-scope-2-27b-it
    --hf-subfolder "$HF_SUBFOLDER"
    --hf-revision main
    --device cuda:0
    --dtype bfloat16
    --chunk-size 512
    --top-k 128
)

python scripts/stage2_extract_sae_features.py \
    --activation-prefix "$PROPERTY_PREFIX" \
    "${COMMON_TRANSCODER_ARGS[@]}"

python scripts/stage2_extract_sae_features.py \
    --activation-prefix "$SUBTYPE_PREFIX" \
    "${COMMON_TRANSCODER_ARGS[@]}"

COMMON_RAW_PROBE_ARGS=(
    --activation-dir "$OUT_DIR"
    --activation-site "$SITE"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layers "$LAYER"
    --splits results/stage2/splits.jsonl
    --bootstrap-samples 1000
)

python scripts/stage2_probe_raw.py \
    "${COMMON_RAW_PROBE_ARGS[@]}" \
    --split-family s1 \
    --output docs/raw_probe_27b_l45_mlp_in_s1.json

python scripts/stage2_probe_raw.py \
    "${COMMON_RAW_PROBE_ARGS[@]}" \
    --split-family s3 \
    --output docs/raw_probe_27b_l45_mlp_in_s3_target_symbol.json

COMMON_TRANSCODER_PROBE_ARGS=(
    --feature-dir "$FEATURE_DIR"
    --activation-site "$SITE"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layer "$LAYER"
    --sae-id "$TRANSCODER_ID"
    --top-k 128
    --splits results/stage2/splits.jsonl
    --solver liblinear
    --bootstrap-samples 1000
)

python scripts/stage2_probe_sae.py \
    "${COMMON_TRANSCODER_PROBE_ARGS[@]}" \
    --split-family s1 \
    --output docs/transcoder_probe_27b_l45_16k_affine_s1.json

python scripts/stage2_probe_sae.py \
    "${COMMON_TRANSCODER_PROBE_ARGS[@]}" \
    --split-family s3 \
    --output docs/transcoder_probe_27b_l45_16k_affine_s3_target_symbol.json

nvidia-smi
echo "Finished: $(date)"
