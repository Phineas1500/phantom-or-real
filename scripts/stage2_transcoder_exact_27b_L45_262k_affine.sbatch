#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_tc_exact_262k
#SBATCH --output=slurm_logs/stage2_tc_exact_262k_%j.out
#SBATCH --error=slurm_logs/stage2_tc_exact_262k_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/activations results/stage2/sae_features results/stage2/transcoder_components_exact docs

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
COMPONENT_DIR=results/stage2/transcoder_components_exact
LAYER=45
TRANSCODER_RELEASE=gemma-scope-2-27b-it-transcoders-all
TRANSCODER_ID=layer_45_width_262k_l0_small_affine
HF_SUBFOLDER=transcoder_all
INPUT_SITE=mlp_in_weighted
TARGET_SITE=mlp_out_hook

python -u scripts/stage2_extract_exact_transcoder_hooks.py \
    --property-jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --subtype-jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --layer "$LAYER" \
    --batch-size 32 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --output-dtype bfloat16 \
    --out-dir "$OUT_DIR"

COMMON_SAE_ARGS=(
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

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_27b_infer_property_L45_${INPUT_SITE}" \
    "${COMMON_SAE_ARGS[@]}"

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_27b_infer_subtype_L45_${INPUT_SITE}" \
    "${COMMON_SAE_ARGS[@]}"

COMMON_RAW_ARGS=(
    --activation-dir "$OUT_DIR"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layers "$LAYER"
    --splits results/stage2/splits.jsonl
    --bootstrap-samples 1000
)

python -u scripts/stage2_probe_raw.py \
    "${COMMON_RAW_ARGS[@]}" \
    --activation-site "$INPUT_SITE" \
    --split-family s1 \
    --output docs/raw_probe_27b_l45_mlp_in_weighted_s1.json

python -u scripts/stage2_probe_raw.py \
    "${COMMON_RAW_ARGS[@]}" \
    --activation-site "$INPUT_SITE" \
    --split-family s3 \
    --output docs/raw_probe_27b_l45_mlp_in_weighted_s3_target_symbol.json

python -u scripts/stage2_probe_raw.py \
    "${COMMON_RAW_ARGS[@]}" \
    --activation-site "$TARGET_SITE" \
    --split-family s1 \
    --output docs/raw_probe_27b_l45_mlp_out_hook_s1.json

python -u scripts/stage2_probe_raw.py \
    "${COMMON_RAW_ARGS[@]}" \
    --activation-site "$TARGET_SITE" \
    --split-family s3 \
    --output docs/raw_probe_27b_l45_mlp_out_hook_s3_target_symbol.json

COMMON_TRANSCODER_PROBE_ARGS=(
    --feature-dir "$FEATURE_DIR"
    --activation-site "$INPUT_SITE"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layer "$LAYER"
    --sae-id "$TRANSCODER_ID"
    --top-k 128
    --splits results/stage2/splits.jsonl
    --solver liblinear
    --bootstrap-samples 1000
)

python -u scripts/stage2_probe_sae.py \
    "${COMMON_TRANSCODER_PROBE_ARGS[@]}" \
    --split-family s1 \
    --output docs/transcoder_probe_27b_l45_262k_affine_exact_s1.json

python -u scripts/stage2_probe_sae.py \
    "${COMMON_TRANSCODER_PROBE_ARGS[@]}" \
    --split-family s3 \
    --output docs/transcoder_probe_27b_l45_262k_affine_exact_s3_target_symbol.json

COMMON_COMPONENT_ARGS=(
    --input-activation-dir "$OUT_DIR"
    --target-activation-dir "$OUT_DIR"
    --feature-dir "$FEATURE_DIR"
    --out-dir "$COMPONENT_DIR"
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layer "$LAYER"
    --input-activation-site "$INPUT_SITE"
    --target-activation-site "$TARGET_SITE"
    --transcoder-release "$TRANSCODER_RELEASE"
    --transcoder-id "$TRANSCODER_ID"
    --top-k 128
    --splits results/stage2/splits.jsonl
    --device cuda:0
    --dtype bfloat16
    --output-dtype bfloat16
    --chunk-size 128
    --verify-dense-rows 4
    --solver liblinear
    --c-values 1.0
    --bootstrap-samples 0
    --skip-target-probe
    --skip-orientation untransposed
)

python -u scripts/stage2_transcoder_component_diagnostics.py \
    "${COMMON_COMPONENT_ARGS[@]}" \
    --split-family s1 \
    --output docs/transcoder_component_probe_27b_l45_262k_affine_exact_s1.json

python -u scripts/stage2_transcoder_component_diagnostics.py \
    "${COMMON_COMPONENT_ARGS[@]}" \
    --reuse-existing \
    --split-family s3 \
    --output docs/transcoder_component_probe_27b_l45_262k_affine_exact_s3_target_symbol.json

nvidia-smi
echo "Finished: $(date)"
