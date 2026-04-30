#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_d4_4b_concat
#SBATCH --output=slurm_logs/stage2_d4_4b_concat_%j.out
#SBATCH --error=slurm_logs/stage2_d4_4b_concat_%j.err
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

run_concat() {
    local combo_name="$1"
    local split="$2"
    local output="$3"
    shift 3
    python -u scripts/stage2_probe_sparse_concat.py \
        --combo-name "$combo_name" \
        "$@" \
        --tasks "${TASKS[@]}" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --c-values "$C_VALUES" \
        --bootstrap-samples 1000 \
        --output "$output"
}

L20_RESID=(
    --feature-pattern results/stage2/sae_features/gemma3_4b_{task}_L20_layer_20_width_16k_l0_small_top128
    --feature-pattern results/stage2/sae_features/gemma3_4b_{task}_L20_layer_20_width_262k_l0_small_top128
)
L22_RESID=(
    --feature-pattern results/stage2/sae_features/gemma3_4b_{task}_L22_layer_22_width_16k_l0_small_top128
    --feature-pattern results/stage2/sae_features/gemma3_4b_{task}_L22_layer_22_width_262k_l0_small_top128
)
L22_ALL=(
    "${L22_RESID[@]}"
    --feature-pattern results/stage2/sae_features/gemma3_4b_{task}_L22_mlp_in_weighted_layer_22_width_16k_l0_small_affine_top128
    --feature-pattern results/stage2/sae_features/gemma3_4b_{task}_L22_mlp_in_weighted_layer_22_width_262k_l0_small_affine_top128
    --feature-pattern results/stage2/sae_features/gemma3_4b_{task}_L22_mlp_out_hook_layer_22_width_16k_l0_small_top128
)

run_concat l22_all_sparse s1 docs/sparse_concat_probe_4b_l22_all_sparse_s1.json "${L22_ALL[@]}"
run_concat l22_all_sparse s3 docs/sparse_concat_probe_4b_l22_all_sparse_s3_target_symbol.json "${L22_ALL[@]}"
run_concat l20_l22_resid_only s1 docs/sparse_concat_probe_4b_l20_l22_resid_s1.json "${L20_RESID[@]}" "${L22_RESID[@]}"
run_concat l20_l22_resid_only s3 docs/sparse_concat_probe_4b_l20_l22_resid_s3_target_symbol.json "${L20_RESID[@]}" "${L22_RESID[@]}"
run_concat l20_l22_all_sparse s1 docs/sparse_concat_probe_4b_l20_l22_all_sparse_s1.json "${L20_RESID[@]}" "${L22_ALL[@]}"
run_concat l20_l22_all_sparse s3 docs/sparse_concat_probe_4b_l20_l22_all_sparse_s3_target_symbol.json "${L20_RESID[@]}" "${L22_ALL[@]}"

echo "Finished: $(date)"
