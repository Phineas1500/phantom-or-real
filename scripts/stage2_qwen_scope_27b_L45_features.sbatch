#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen_scope_27b_l45_features
#SBATCH --output=slurm_logs/qwen_scope_27b_l45_features_%j.out
#SBATCH --error=slurm_logs/qwen_scope_27b_l45_features_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/sae_features

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

LAYER="${LAYER:-45}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

run_qwen_scope() {
    local repo_id="$1"
    local sae_id="$2"
    local top_k="$3"

    python scripts/stage2_extract_qwen_scope_features.py \
        --activation-prefix "results/stage2/activations/qwen35_27b_infer_property_L${LAYER}" \
        --out-dir results/stage2/sae_features \
        --sae-repo-id "$repo_id" \
        --sae-id "$sae_id" \
        --layer "$LAYER" \
        --top-k "$top_k" \
        --device cuda:0 \
        --dtype bfloat16 \
        --chunk-size "${CHUNK_SIZE:-128}" \
        "${LIMIT_ARGS[@]}"

    python scripts/stage2_extract_qwen_scope_features.py \
        --activation-prefix "results/stage2/activations/qwen35_27b_infer_subtype_L${LAYER}" \
        --out-dir results/stage2/sae_features \
        --sae-repo-id "$repo_id" \
        --sae-id "$sae_id" \
        --layer "$LAYER" \
        --top-k "$top_k" \
        --device cuda:0 \
        --dtype bfloat16 \
        --chunk-size "${CHUNK_SIZE:-128}" \
        "${LIMIT_ARGS[@]}"
}

run_qwen_scope \
    Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
    qwenscope_qwen35_27b_w80k_l0_50 \
    50

run_qwen_scope \
    Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100 \
    qwenscope_qwen35_27b_w80k_l0_100 \
    100

echo "Finished: $(date)"
