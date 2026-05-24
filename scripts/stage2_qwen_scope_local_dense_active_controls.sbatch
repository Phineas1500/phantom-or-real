#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen_local_dense_active
#SBATCH --output=slurm_logs/qwen_local_dense_active_%j.out
#SBATCH --error=slurm_logs/qwen_local_dense_active_%j.err

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

TASKS=(infer_property infer_subtype)
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10

run_single_dense_active() {
    local split_family="$1"
    local suffix="$2"

    python -u scripts/stage2_probe_dense_active_sparse.py \
        --feature-prefix "results/stage2/sae_features/qwen35_27b_infer_property_L53_mlp_out_local_topk_mlp_out_l53_w4096_k64_${suffix}_top64" \
        --feature-prefix "results/stage2/sae_features/qwen35_27b_infer_subtype_L53_mlp_out_local_topk_mlp_out_l53_w4096_k64_${suffix}_top64" \
        --feature-prefix "results/stage2/sae_features/qwen35_27b_infer_property_L53_mlp_in_weighted_local_topk_tc_mlp_in_weighted_to_mlp_out_l53_w4096_k64_${suffix}_top64" \
        --feature-prefix "results/stage2/sae_features/qwen35_27b_infer_subtype_L53_mlp_in_weighted_local_topk_tc_mlp_in_weighted_to_mlp_out_l53_w4096_k64_${suffix}_top64" \
        --feature-prefix "results/stage2/crosscoder_features/qwen35_27b_infer_property_crosscoder_local_topk_layers_16_31_40_53_w4096_k64_${suffix}_top64" \
        --feature-prefix "results/stage2/crosscoder_features/qwen35_27b_infer_subtype_crosscoder_local_topk_layers_16_31_40_53_w4096_k64_${suffix}_top64" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --c-values "$C_VALUES" \
        --bootstrap-samples 1000 \
        --output "docs/qwen_scope_dense_active_local_sparse_probe_27b_${suffix}.json"
}

run_concat_dense_active() {
    local split_family="$1"
    local suffix="$2"

    python -u scripts/stage2_probe_sparse_concat.py \
        --combo-name qwen_l53_residual_local_mlpout_localtc_dense_active \
        --feature-pattern results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_50_top50 \
        --feature-pattern results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_100_top100 \
        --feature-pattern "results/stage2/sae_features/qwen35_27b_{task}_L53_mlp_out_local_topk_mlp_out_l53_w4096_k64_${suffix}_top64" \
        --feature-pattern "results/stage2/sae_features/qwen35_27b_{task}_L53_mlp_in_weighted_local_topk_tc_mlp_in_weighted_to_mlp_out_l53_w4096_k64_${suffix}_top64" \
        --tasks "${TASKS[@]}" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --dense-active \
        --c-values "$C_VALUES" \
        --bootstrap-samples 1000 \
        --output "docs/qwen_scope_dense_active_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_${suffix}.json"
}

run_single_dense_active s1 s1
run_single_dense_active s3 s3
run_concat_dense_active s1 s1
run_concat_dense_active s3 s3

nvidia-smi
echo "Finished: $(date)"
