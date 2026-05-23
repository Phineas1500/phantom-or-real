#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen_scope_sparse_concat
#SBATCH --output=slurm_logs/qwen_scope_sparse_concat_%j.out
#SBATCH --error=slurm_logs/qwen_scope_sparse_concat_%j.err

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

require_feature() {
    local prefix="$1"
    test -f "${prefix}.safetensors"
    test -f "${prefix}.meta.json"
    test -f "${prefix}.example_ids.jsonl"
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

L45_PATTERNS=(
    --feature-pattern results/stage2/sae_features/qwen35_27b_{task}_L45_qwenscope_qwen35_27b_w80k_l0_50_top50
    --feature-pattern results/stage2/sae_features/qwen35_27b_{task}_L45_qwenscope_qwen35_27b_w80k_l0_100_top100
)

L53_PATTERNS=(
    --feature-pattern results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_50_top50
    --feature-pattern results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_100_top100
)

for task in "${TASKS[@]}"; do
    for pattern in "${L45_PATTERNS[@]}" "${L53_PATTERNS[@]}"; do
        if [[ "$pattern" == --feature-pattern ]]; then
            continue
        fi
        require_feature "${pattern/\{task\}/$task}"
    done
done

run_concat \
    qwen_l45_w80k_l0_50_l0_100 \
    s1 \
    docs/qwen_scope_sparse_concat_probe_27b_l45_w80k_l0_50_l0_100_s1.json \
    "${L45_PATTERNS[@]}"

run_concat \
    qwen_l45_w80k_l0_50_l0_100 \
    s3 \
    docs/qwen_scope_sparse_concat_probe_27b_l45_w80k_l0_50_l0_100_s3.json \
    "${L45_PATTERNS[@]}"

run_concat \
    qwen_l53_w80k_l0_50_l0_100 \
    s1 \
    docs/qwen_scope_sparse_concat_probe_27b_l53_w80k_l0_50_l0_100_s1.json \
    "${L53_PATTERNS[@]}"

run_concat \
    qwen_l53_w80k_l0_50_l0_100 \
    s3 \
    docs/qwen_scope_sparse_concat_probe_27b_l53_w80k_l0_50_l0_100_s3.json \
    "${L53_PATTERNS[@]}"

run_concat \
    qwen_l45_l53_w80k_l0_50_l0_100 \
    s1 \
    docs/qwen_scope_sparse_concat_probe_27b_l45_l53_w80k_l0_50_l0_100_s1.json \
    "${L45_PATTERNS[@]}" \
    "${L53_PATTERNS[@]}"

run_concat \
    qwen_l45_l53_w80k_l0_50_l0_100 \
    s3 \
    docs/qwen_scope_sparse_concat_probe_27b_l45_l53_w80k_l0_50_l0_100_s3.json \
    "${L45_PATTERNS[@]}" \
    "${L53_PATTERNS[@]}"

echo "Finished: $(date)"
