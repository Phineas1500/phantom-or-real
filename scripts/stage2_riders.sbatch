#!/bin/bash
#SBATCH -A scholar
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --job-name=riders
#SBATCH --output=slurm_logs/riders_%j.out
#SBATCH --error=slurm_logs/riders_%j.err
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
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "Node: $(hostname)"; echo "Started: $(date)"; echo "Rider: ${RIDER:?set RIDER=FR1 or LR2}"
if [ "$RIDER" = "FR1" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --fr1 --rank-list 8 \
        --shard-index 0 --shard-count 1 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
else
    python -u scripts/stage2_rank_k_guard_v2.py \
        --donor-ladder --rank-list 8 \
        --selection-seed 20260812 --per-height 32 \
        --shard-index 0 --shard-count 1 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
fi
