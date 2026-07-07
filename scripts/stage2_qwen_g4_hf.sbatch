#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen_g4
#SBATCH --output=slurm_logs/qwen_g4hf_%j.out
#SBATCH --error=slurm_logs/qwen_g4hf_%j.err
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
python -u scripts/stage2_qwen_g4_hf.py
echo "Finished: $(date)"
