#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage2_decode_manifest
#SBATCH --output=slurm_logs/stage2_decode_manifest_%j.out
#SBATCH --error=slurm_logs/stage2_decode_manifest_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/decode_time docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH=/scratch/scholar/$USER/beyond-deduction
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

TRACE_LAYERS="${TRACE_LAYERS:-45,53}"
TRACE_MAX_NEW_TOKENS="${TRACE_MAX_NEW_TOKENS:-96}"
TRACE_SUFFIX="${TRACE_SUFFIX:-decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap}"

python -u scripts/stage2_decode_projection_trace.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --layers "$TRACE_LAYERS" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --selection-mode manifest_recognition_gap \
    --manifest docs/commitment_rowset_manifest.json \
    --manifest-require-no-decode-trace \
    --max-new-tokens "$TRACE_MAX_NEW_TOKENS" \
    --temperature 0 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl "results/stage2/decode_time/${TRACE_SUFFIX}.jsonl" \
    --direction-output "results/stage2/decode_time/${TRACE_SUFFIX}_direction.npz" \
    --output "docs/${TRACE_SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
