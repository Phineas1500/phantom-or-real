#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_cc_65k
#SBATCH --output=slurm_logs/stage2_cc_65k_%j.out
#SBATCH --error=slurm_logs/stage2_cc_65k_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/activations results/stage2/crosscoder_features docs

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
FEATURE_DIR=results/stage2/crosscoder_features
LAYERS=16,31,40,53
LAYER_LIST=(16 31 40 53)
CROSSCODER_ID=layer_16_31_40_53_width_65k_l0_medium

need_property_extract=0
need_subtype_extract=0
for layer in "${LAYER_LIST[@]}"; do
    if [ ! -f "$OUT_DIR/gemma3_27b_infer_property_L${layer}.safetensors" ]; then
        need_property_extract=1
    fi
    if [ ! -f "$OUT_DIR/gemma3_27b_infer_subtype_L${layer}.safetensors" ]; then
        need_subtype_extract=1
    fi
done

if [ "$need_property_extract" -eq 1 ]; then
    python scripts/stage2_extract.py \
        --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
        --model google/gemma-3-27b-it \
        --model-key gemma3_27b \
        --task infer_property \
        --layers "$LAYERS" \
        --batch-size 32 \
        --n-devices 2 \
        --n-ctx 4096 \
        --load-mode no-processing \
        --out-dir "$OUT_DIR"
else
    echo "Reusing property raw residual layers ${LAYERS}"
fi

if [ "$need_subtype_extract" -eq 1 ]; then
    python scripts/stage2_extract.py \
        --jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
        --model google/gemma-3-27b-it \
        --model-key gemma3_27b \
        --task infer_subtype \
        --layers "$LAYERS" \
        --batch-size 32 \
        --n-devices 2 \
        --n-ctx 4096 \
        --load-mode no-processing \
        --out-dir "$OUT_DIR"
else
    echo "Reusing subtype raw residual layers ${LAYERS}"
fi

COMMON_CROSSCODER_ARGS=(
    --activation-dir "$OUT_DIR"
    --model-key gemma3_27b
    --layers "$LAYERS"
    --out-dir "$FEATURE_DIR"
    --crosscoder-id "$CROSSCODER_ID"
    --hf-repo-id google/gemma-scope-2-27b-it
    --hf-subfolder crosscoder
    --hf-revision main
    --device cuda:0
    --dtype bfloat16
    --chunk-size 512
    --top-k 128
)

if [ ! -f "$FEATURE_DIR/gemma3_27b_infer_property_crosscoder_${CROSSCODER_ID}_top128.safetensors" ]; then
    python scripts/stage2_extract_crosscoder_features.py \
        --task infer_property \
        "${COMMON_CROSSCODER_ARGS[@]}"
else
    echo "Reusing property crosscoder features"
fi

if [ ! -f "$FEATURE_DIR/gemma3_27b_infer_subtype_crosscoder_${CROSSCODER_ID}_top128.safetensors" ]; then
    python scripts/stage2_extract_crosscoder_features.py \
        --task infer_subtype \
        "${COMMON_CROSSCODER_ARGS[@]}"
else
    echo "Reusing subtype crosscoder features"
fi

COMMON_RAW_CONCAT_ARGS=(
    --activation-dir "$OUT_DIR"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layers "$LAYERS"
    --splits results/stage2/splits.jsonl
    --solver liblinear
    --bootstrap-samples 1000
)

python scripts/stage2_probe_raw_concat.py \
    "${COMMON_RAW_CONCAT_ARGS[@]}" \
    --split-family s1 \
    --output docs/raw_probe_27b_crosscoder_layers_16_31_40_53_s1.json

python scripts/stage2_probe_raw_concat.py \
    "${COMMON_RAW_CONCAT_ARGS[@]}" \
    --split-family s3 \
    --output docs/raw_probe_27b_crosscoder_layers_16_31_40_53_s3_target_symbol.json

COMMON_CROSSCODER_PROBE_ARGS=(
    --feature-dir "$FEATURE_DIR"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --crosscoder-id "$CROSSCODER_ID"
    --top-k 128
    --splits results/stage2/splits.jsonl
    --solver liblinear
    --bootstrap-samples 1000
)

python scripts/stage2_probe_crosscoder.py \
    "${COMMON_CROSSCODER_PROBE_ARGS[@]}" \
    --split-family s1 \
    --output docs/crosscoder_probe_27b_layers_16_31_40_53_65k_medium_s1.json

python scripts/stage2_probe_crosscoder.py \
    "${COMMON_CROSSCODER_PROBE_ARGS[@]}" \
    --split-family s3 \
    --output docs/crosscoder_probe_27b_layers_16_31_40_53_65k_medium_s3_target_symbol.json

nvidia-smi
echo "Finished: $(date)"
