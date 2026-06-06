#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_prefix_margin
#SBATCH --output=slurm_logs/stage2_prefix_margin_%j.out
#SBATCH --error=slurm_logs/stage2_prefix_margin_%j.err

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

PREFIX_CHECKPOINTS="${PREFIX_CHECKPOINTS:-0,1,4,8,16,32,64,final}"
PREFIX_SUFFIX="${PREFIX_SUFFIX:-prefix_conditioned_margin_trajectory_gemma_manifest}"

python -u scripts/stage2_prefix_conditioned_margin_trajectory.py \
    --decode-jsonl results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap.jsonl \
    --source-jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --checkpoints "$PREFIX_CHECKPOINTS" \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl "results/stage2/decode_time/${PREFIX_SUFFIX}.jsonl" \
    --output "docs/${PREFIX_SUFFIX}.json" \
    --md-output "docs/${PREFIX_SUFFIX}.md"

nvidia-smi
echo "Finished: $(date)"
