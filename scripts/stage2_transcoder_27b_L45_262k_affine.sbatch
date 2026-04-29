#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_tc_l45_262k
#SBATCH --output=slurm_logs/stage2_tc_l45_262k_%j.out
#SBATCH --error=slurm_logs/stage2_tc_l45_262k_%j.err

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
TRANSCODER_RELEASE=gemma-scope-2-27b-it-transcoders-all
TRANSCODER_ID=layer_45_width_262k_l0_small_affine
NEURONPEDIA_SOURCE=45-gemmascope-2-transcoder-262k
HF_SUBFOLDER=transcoder_all

PROPERTY_PREFIX="$OUT_DIR/gemma3_27b_infer_property_L45_${SITE}"
SUBTYPE_PREFIX="$OUT_DIR/gemma3_27b_infer_subtype_L45_${SITE}"

if [ ! -f "${PROPERTY_PREFIX}.safetensors" ] || [ ! -f "${SUBTYPE_PREFIX}.safetensors" ]; then
    echo "Missing L45 ${SITE} activations. Run scripts/stage2_transcoder_27b_L45_16k_affine.sbatch first."
    exit 1
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
    --chunk-size 128
    --top-k 128
)

python scripts/stage2_extract_sae_features.py \
    --activation-prefix "$PROPERTY_PREFIX" \
    "${COMMON_TRANSCODER_ARGS[@]}"

python scripts/stage2_extract_sae_features.py \
    --activation-prefix "$SUBTYPE_PREFIX" \
    "${COMMON_TRANSCODER_ARGS[@]}"

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
    --output docs/transcoder_probe_27b_l45_262k_affine_s1.json

python scripts/stage2_probe_sae.py \
    "${COMMON_TRANSCODER_PROBE_ARGS[@]}" \
    --split-family s3 \
    --output docs/transcoder_probe_27b_l45_262k_affine_s3_target_symbol.json

python scripts/stage2_analyze_sae_features.py \
    --feature-dir "$FEATURE_DIR" \
    --activation-site "$SITE" \
    --model-key gemma3_27b \
    --tasks infer_property infer_subtype \
    --layer "$LAYER" \
    --sae-ids "$TRANSCODER_ID" \
    --top-k 128 \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --output docs/transcoder_feature_stability_27b_l45_262k_affine_s1.json \
    --output-top-n 100 \
    --probe-reports docs/transcoder_probe_27b_l45_262k_affine_s1.json \
    --solver liblinear

python scripts/stage2_neuronpedia_feature_audit.py \
    --feature-report docs/transcoder_feature_stability_27b_l45_262k_affine_s1.json \
    --sae-id "$TRANSCODER_ID" \
    --source-id "$NEURONPEDIA_SOURCE" \
    --top-n 10 \
    --output-json docs/neuronpedia_transcoder_audit_27b_l45_262k.json \
    --output-md docs/neuronpedia_transcoder_audit_27b_l45_262k.md

nvidia-smi
echo "Finished: $(date)"
