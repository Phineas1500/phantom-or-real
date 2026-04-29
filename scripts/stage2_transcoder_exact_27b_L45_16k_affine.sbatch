#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage2_tc_exact_16k
#SBATCH --output=slurm_logs/stage2_tc_exact_16k_%j.out
#SBATCH --error=slurm_logs/stage2_tc_exact_16k_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/sae_features results/stage2/transcoder_components_exact docs

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
TRANSCODER_ID=layer_45_width_16k_l0_small_affine
HF_SUBFOLDER=transcoder_all
INPUT_SITE=mlp_in_weighted
TARGET_SITE=mlp_out_hook
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10

for task in infer_property infer_subtype; do
    for site in "$INPUT_SITE" "$TARGET_SITE"; do
        prefix="$OUT_DIR/gemma3_27b_${task}_L45_${site}"
        if [ ! -f "${prefix}.safetensors" ]; then
            echo "Missing required exact activation file: ${prefix}.safetensors" >&2
            echo "Run scripts/stage2_transcoder_exact_27b_L45_262k_affine.sbatch first." >&2
            exit 1
        fi
    done
done

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

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_27b_infer_property_L45_${INPUT_SITE}" \
    "${COMMON_TRANSCODER_ARGS[@]}"

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_27b_infer_subtype_L45_${INPUT_SITE}" \
    "${COMMON_TRANSCODER_ARGS[@]}"

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
    --c-values "$C_VALUES"
    --bootstrap-samples 1000
)

python -u scripts/stage2_probe_sae.py \
    "${COMMON_TRANSCODER_PROBE_ARGS[@]}" \
    --split-family s1 \
    --output docs/transcoder_probe_27b_l45_16k_affine_exact_lowc_s1.json

python -u scripts/stage2_probe_sae.py \
    "${COMMON_TRANSCODER_PROBE_ARGS[@]}" \
    --split-family s3 \
    --output docs/transcoder_probe_27b_l45_16k_affine_exact_lowc_s3_target_symbol.json

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
    --output docs/transcoder_component_probe_27b_l45_16k_affine_exact_s1.json

python -u scripts/stage2_transcoder_component_diagnostics.py \
    "${COMMON_COMPONENT_ARGS[@]}" \
    --reuse-existing \
    --split-family s3 \
    --output docs/transcoder_component_probe_27b_l45_16k_affine_exact_s3_target_symbol.json

nvidia-smi
echo "Finished: $(date)"
