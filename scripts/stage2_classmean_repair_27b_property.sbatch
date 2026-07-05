#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=classmean_repair
#SBATCH --output=slurm_logs/classmean_repair_%j.out
#SBATCH --error=slurm_logs/classmean_repair_%j.err

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

python -u scripts/stage2_rank_k_guard_v2.py \
    --class-mean \
    --rank-list 8 \
    --shard-index "${CLASSMEAN_SHARD_INDEX:-0}" \
    --shard-count "${CLASSMEAN_SHARD_COUNT:-2}" \
    --per-height "${CLASSMEAN_PER_HEIGHT:-16}" \
    --samples-per-row "${CLASSMEAN_SAMPLES:-8}" \
    --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing

echo "Finished: $(date)"
