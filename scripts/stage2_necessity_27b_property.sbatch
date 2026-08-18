#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --job-name=necessity_k
#SBATCH --output=slurm_logs/necessity_k_%j.out
#SBATCH --error=slurm_logs/necessity_k_%j.err

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
echo "Node: $(hostname)"; echo "Started: $(date)"; echo "Shard: ${NECESSITY_SHARD:?set NECESSITY_SHARD=0..3}"

FIIC_EXCLUDE="3073,3290,3415,4322,4675,6188,6327,8035,8298,8874,9549,10079,10714,3109,3134,3471,3680,3685,3738,4270,5235,6312,6367,6411,7047,7812,8388,9855,10524"

if [ "${NECESSITY_SITE:-0}" = "1" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --necessity-site \
        --rank-list 8 \
        --shard-index "$NECESSITY_SHARD" --shard-count 3 \
        --correct-per-height 24 --correct-seed 20260809 \
        --exclude-rows "$FIIC_EXCLUDE" \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
elif [ "${NECESSITY_PRIME:-0}" = "1" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --necessity-prime \
        --rank-list 8 \
        --shard-index "$NECESSITY_SHARD" --shard-count 3 \
        --correct-per-height 24 --correct-seed 20260809 \
        --exclude-rows "$FIIC_EXCLUDE" \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
elif [ "$NECESSITY_SHARD" = "3" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --necessity --necessity-anchor \
        --rank-list 8 \
        --shard-index 0 --shard-count 1 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
else
    python -u scripts/stage2_rank_k_guard_v2.py \
        --necessity \
        --rank-list 8 \
        --shard-index "$NECESSITY_SHARD" --shard-count 3 \
        --correct-per-height 24 --correct-seed 20260809 \
        --exclude-rows "$FIIC_EXCLUDE" \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
fi

echo "Finished: $(date)"
