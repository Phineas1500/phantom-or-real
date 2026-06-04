#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_fmtctl27b
#SBATCH --output=slurm_logs/stage2_fmtctl27b_%j.out
#SBATCH --error=slurm_logs/stage2_fmtctl27b_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/positive_control docs

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

echo "node=$(hostname)"
date
nvidia-smi

python -u scripts/stage2_positive_control_format.py   --model google/gemma-3-27b-it   --model-key gemma3_27b   --layer 45   --train-limit "${TRAIN_LIMIT:-24}"   --eval-limit "${EVAL_LIMIT:-8}"   --conditions baseline,toward_upper,toward_lower,orthogonal,gaussian   --strengths "${STRENGTHS:-0.5,1,2,4}"   --intervention-scope last_token_each_forward   --max-new-tokens "${MAX_NEW_TOKENS:-32}"   --temperature 0   --min-uppercase-delta "${MIN_UPPERCASE_DELTA:-0.25}"   --n-devices 2   --n-ctx 2048   --dtype bfloat16   --load-mode no-processing   --out-jsonl results/stage2/positive_control/format_gemma3_27b_l45.jsonl   --direction-output results/stage2/positive_control/format_gemma3_27b_l45_direction.npz   --output docs/positive_control_format_gemma3_27b_l45.json

nvidia-smi
