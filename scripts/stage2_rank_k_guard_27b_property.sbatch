#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=rank_k_guard
#SBATCH --output=slurm_logs/rank_k_guard_%j.out
#SBATCH --error=slurm_logs/rank_k_guard_%j.err

set -euo pipefail
cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/erasure docs
module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom
export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH=/scratch/scholar/$USER/beyond-deduction
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "Node: $(hostname)"; echo "Started: $(date)"

python -u scripts/stage2_rank_k_guard.py \
    --samples-per-row "${RANK_GUARD_SAMPLES:-8}" \
    --layers "${RANK_GUARD_LAYERS:-30,40,45}" \
    --rank-list "${RANK_GUARD_RANKS:-1,2,3,4,6,8}" \
    --basis-modes "${RANK_GUARD_BASIS_MODES:-leave_one_row_out,in_sample}" \
    --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing

echo "Finished: $(date)"
