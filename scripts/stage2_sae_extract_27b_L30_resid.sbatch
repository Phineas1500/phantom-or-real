#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_sae_l30_extract
#SBATCH --output=slurm_logs/stage2_sae_l30_extract_%j.out
#SBATCH --error=slurm_logs/stage2_sae_l30_extract_%j.err

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

extract_l30_resid() {
    local width="$1"
    local chunk_size="$2"
    local sae_id="layer_30_width_${width}_l0_small"

    echo "Extracting L30 residual SAE ${width}: ${sae_id}"
    python -u scripts/stage2_extract_sae_features.py \
        --activation-prefix results/stage2/activations/gemma3_27b_infer_property_L30 \
        --out-dir results/stage2/sae_features \
        --sae-release gemma-scope-2-27b-it-res-all \
        --sae-id "${sae_id}" \
        --hf-repo-id google/gemma-scope-2-27b-it \
        --hf-subfolder resid_post_all \
        --hf-revision main \
        --device cuda:0 \
        --dtype bfloat16 \
        --chunk-size "${chunk_size}" \
        --top-k 128

    python -u scripts/stage2_extract_sae_features.py \
        --activation-prefix results/stage2/activations/gemma3_27b_infer_subtype_L30 \
        --out-dir results/stage2/sae_features \
        --sae-release gemma-scope-2-27b-it-res-all \
        --sae-id "${sae_id}" \
        --hf-repo-id google/gemma-scope-2-27b-it \
        --hf-subfolder resid_post_all \
        --hf-revision main \
        --device cuda:0 \
        --dtype bfloat16 \
        --chunk-size "${chunk_size}" \
        --top-k 128
}

extract_l30_resid 16k 512
extract_l30_resid 262k 256

echo "Finished: $(date)"
