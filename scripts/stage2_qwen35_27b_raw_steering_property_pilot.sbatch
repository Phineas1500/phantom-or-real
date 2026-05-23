#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_raw_steer
#SBATCH --output=slurm_logs/qwen35_raw_steer_%j.out
#SBATCH --error=slurm_logs/qwen35_raw_steer_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/steering docs

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

LAYER="${LAYER:-45}"
LAYER_TAG="l${LAYER}"

python -u scripts/stage2_qwen_steer_raw_direction.py \
    --jsonl results/full/with_errortype/qwen35_27b_infer_property.jsonl \
    --model Qwen/Qwen3.5-27B \
    --model-key qwen35_27b \
    --task infer_property \
    --layer "$LAYER" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights 3,4 \
    --per-height-label 1 \
    --conditions baseline,raw,orthogonal \
    --strengths=-1,1 \
    --intervention-scope last_token_each_forward \
    --max-new-tokens 96 \
    --temperature 0 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --device-map auto \
    --disable-thinking \
    --out-jsonl "results/stage2/steering/qwen35_raw_${LAYER_TAG}_property_steering_pilot.jsonl" \
    --direction-output "results/stage2/steering/qwen35_raw_${LAYER_TAG}_property_steering_direction.npz" \
    --output "docs/qwen35_raw_steering_27b_${LAYER_TAG}_property_pilot.json"

nvidia-smi
echo "Finished: $(date)"
