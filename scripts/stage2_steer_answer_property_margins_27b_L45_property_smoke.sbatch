#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_margin27b
#SBATCH --output=slurm_logs/stage2_margin27b_%j.out
#SBATCH --error=slurm_logs/stage2_margin27b_%j.err

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

python -u scripts/stage2_steer_answer_property_margins.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --layer 45 \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --answer-label-source gold \
    --answer-target polarity \
    --option-seed 20260430 \
    --heights 3,4 \
    --per-height-label 2 \
    --conditions baseline,toward_gold,away_gold,orthogonal \
    --strengths 0.5,1 \
    --intervention-scope last_token_each_forward \
    --max-new-tokens 8 \
    --temperature 0 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --resume \
    --out-jsonl results/stage2/steering/answer_property_margins_27b_l45_polarity_smoke.jsonl \
    --direction-output results/stage2/steering/answer_property_margins_27b_l45_polarity_direction.npz \
    --output docs/answer_property_margins_27b_l45_polarity_smoke.json

nvidia-smi
echo "Finished: $(date)"
