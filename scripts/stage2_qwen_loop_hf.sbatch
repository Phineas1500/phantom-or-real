#!/bin/bash
#SBATCH -A scholar
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --job-name=qwen_loop
#SBATCH --output=slurm_logs/qwen_loop_%j.out
#SBATCH --error=slurm_logs/qwen_loop_%j.err
set -euo pipefail
cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/erasure docs
module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom
export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH=/scratch/scholar/$USER/beyond-deduction
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "Node: $(hostname)"; echo "Started: $(date)"
echo "Mode: ${QWEN_LOOP_MODE:?set QWEN_LOOP_MODE=CAL or 0..5}"
if [ "$QWEN_LOOP_MODE" = "CAL" ]; then
    python -u scripts/stage2_qwen_loop_hf.py --calibration
else
    python -u scripts/stage2_qwen_loop_hf.py --shard-index "$QWEN_LOOP_MODE" --shard-count 6
fi
echo "Finished: $(date)"
