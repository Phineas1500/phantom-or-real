#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage2_tc_exact_4b_l22_16k
#SBATCH --output=slurm_logs/stage2_tc_exact_4b_l22_16k_%j.out
#SBATCH --error=slurm_logs/stage2_tc_exact_4b_l22_16k_%j.err
#SBATCH --mail-user=semani@purdue.edu
#SBATCH --mail-type=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
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

OUT_DIR=results/stage2/activations
FEATURE_DIR=results/stage2/sae_features
LAYER=22
INPUT_SITE=mlp_in_weighted
TRANSCODER_RELEASE=gemma-scope-2-4b-it-transcoders-all
TRANSCODER_ID=layer_22_width_16k_l0_small_affine
HF_REPO_ID=google/gemma-scope-2-4b-it
HF_SUBFOLDER=transcoder_all

for task in infer_property infer_subtype; do
    prefix="$OUT_DIR/gemma3_4b_${task}_L${LAYER}_${INPUT_SITE}"
    if [ ! -f "${prefix}.safetensors" ]; then
        echo "Missing exact transcoder input activations: ${prefix}.safetensors" >&2
        echo "Run scripts/stage2_extract_exact_hooks_4b_L22.sbatch first." >&2
        exit 1
    fi
done

COMMON_TRANSCODER_ARGS=(
    --out-dir "$FEATURE_DIR"
    --sae-release "$TRANSCODER_RELEASE"
    --sae-id "$TRANSCODER_ID"
    --hf-repo-id "$HF_REPO_ID"
    --hf-subfolder "$HF_SUBFOLDER"
    --hf-revision main
    --device cuda:0
    --dtype bfloat16
    --chunk-size 512
    --top-k 128
)

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_4b_infer_property_L${LAYER}_${INPUT_SITE}" \
    "${COMMON_TRANSCODER_ARGS[@]}"

python -u scripts/stage2_extract_sae_features.py \
    --activation-prefix "$OUT_DIR/gemma3_4b_infer_subtype_L${LAYER}_${INPUT_SITE}" \
    "${COMMON_TRANSCODER_ARGS[@]}"

nvidia-smi
echo "Finished: $(date)"
