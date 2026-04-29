#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_probe_l40_concat
#SBATCH --output=slurm_logs/stage2_probe_l40_concat_%j.out
#SBATCH --error=slurm_logs/stage2_probe_l40_concat_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

TASKS=(infer_property infer_subtype)
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10

require_feature() {
    local prefix="$1"
    test -f "${prefix}.safetensors"
    test -f "${prefix}.meta.json"
    test -f "${prefix}.example_ids.jsonl"
}

probe_l40_resid() {
    local width="$1"
    local sae_id="layer_40_width_${width}_l0_small"

    echo "Checking L40 residual SAE ${width} artifacts"
    require_feature "results/stage2/sae_features/gemma3_27b_infer_property_L40_${sae_id}_top128"
    require_feature "results/stage2/sae_features/gemma3_27b_infer_subtype_L40_${sae_id}_top128"

    echo "Probing L40 residual SAE ${width}: S1"
    python -u scripts/stage2_probe_sae.py \
        --feature-dir results/stage2/sae_features \
        --model-key gemma3_27b \
        --tasks "${TASKS[@]}" \
        --layer 40 \
        --sae-id "${sae_id}" \
        --top-k 128 \
        --splits results/stage2/splits.jsonl \
        --split-family s1 \
        --c-values "${C_VALUES}" \
        --bootstrap-samples 1000 \
        --output "docs/sae_probe_27b_l40_${width}_s1.json"

    echo "Probing L40 residual SAE ${width}: S3"
    python -u scripts/stage2_probe_sae.py \
        --feature-dir results/stage2/sae_features \
        --model-key gemma3_27b \
        --tasks "${TASKS[@]}" \
        --layer 40 \
        --sae-id "${sae_id}" \
        --top-k 128 \
        --splits results/stage2/splits.jsonl \
        --split-family s3 \
        --c-values "${C_VALUES}" \
        --bootstrap-samples 1000 \
        --output "docs/sae_probe_27b_l40_${width}_s3_target_symbol.json"
}

run_concat() {
    local combo_name="$1"
    local split_family="$2"
    local output="$3"
    shift 3

    echo "Probing concat ${combo_name}: ${split_family}"
    python -u scripts/stage2_probe_sparse_concat.py \
        --combo-name "${combo_name}" \
        "$@" \
        --tasks "${TASKS[@]}" \
        --splits results/stage2/splits.jsonl \
        --split-family "${split_family}" \
        --c-values "${C_VALUES}" \
        --bootstrap-samples 1000 \
        --output "${output}"
}

probe_l40_resid 16k
probe_l40_resid 262k

L30_RESID_PATTERNS=(
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L30_layer_30_width_16k_l0_small_top128
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L30_layer_30_width_262k_l0_small_top128
)

L40_RESID_PATTERNS=(
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L40_layer_40_width_16k_l0_small_top128
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L40_layer_40_width_262k_l0_small_top128
)

L45_RESID_PATTERNS=(
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L45_layer_45_width_16k_l0_small_top128
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L45_layer_45_width_262k_l0_small_top128
)

L45_ALL_PATTERNS=(
    "${L45_RESID_PATTERNS[@]}"
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L45_mlp_in_weighted_layer_45_width_16k_l0_small_affine_top128
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L45_mlp_in_weighted_layer_45_width_262k_l0_small_affine_top128
    --feature-pattern results/stage2/sae_features/gemma3_27b_{task}_L45_mlp_out_hook_layer_45_width_16k_l0_small_top128
)

run_concat \
    l40_resid16k_resid262k_broadc \
    s1 \
    docs/sparse_concat_probe_27b_l40_resid16k_resid262k_broadc_s1.json \
    "${L40_RESID_PATTERNS[@]}"

run_concat \
    l40_resid16k_resid262k_broadc \
    s3 \
    docs/sparse_concat_probe_27b_l40_resid16k_resid262k_broadc_s3_target_symbol.json \
    "${L40_RESID_PATTERNS[@]}"

run_concat \
    l40_l45_resid16k_resid262k_broadc \
    s1 \
    docs/sparse_concat_probe_27b_l40_l45_resid16k_resid262k_broadc_s1.json \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_RESID_PATTERNS[@]}"

run_concat \
    l40_l45_resid16k_resid262k_broadc \
    s3 \
    docs/sparse_concat_probe_27b_l40_l45_resid16k_resid262k_broadc_s3_target_symbol.json \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_RESID_PATTERNS[@]}"

run_concat \
    l40_l45_all_sparse_broadc \
    s1 \
    docs/sparse_concat_probe_27b_l40_l45_all_sparse_broadc_s1.json \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_ALL_PATTERNS[@]}"

run_concat \
    l40_l45_all_sparse_broadc \
    s3 \
    docs/sparse_concat_probe_27b_l40_l45_all_sparse_broadc_s3_target_symbol.json \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_ALL_PATTERNS[@]}"

run_concat \
    l30_l40_l45_resid16k_resid262k_broadc \
    s1 \
    docs/sparse_concat_probe_27b_l30_l40_l45_resid16k_resid262k_broadc_s1.json \
    "${L30_RESID_PATTERNS[@]}" \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_RESID_PATTERNS[@]}"

run_concat \
    l30_l40_l45_resid16k_resid262k_broadc \
    s3 \
    docs/sparse_concat_probe_27b_l30_l40_l45_resid16k_resid262k_broadc_s3_target_symbol.json \
    "${L30_RESID_PATTERNS[@]}" \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_RESID_PATTERNS[@]}"

run_concat \
    l30_l40_l45_all_sparse_broadc \
    s1 \
    docs/sparse_concat_probe_27b_l30_l40_l45_all_sparse_broadc_s1.json \
    "${L30_RESID_PATTERNS[@]}" \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_ALL_PATTERNS[@]}"

run_concat \
    l30_l40_l45_all_sparse_broadc \
    s3 \
    docs/sparse_concat_probe_27b_l30_l40_l45_all_sparse_broadc_s3_target_symbol.json \
    "${L30_RESID_PATTERNS[@]}" \
    "${L40_RESID_PATTERNS[@]}" \
    "${L45_ALL_PATTERNS[@]}"

echo "Finished: $(date)"
