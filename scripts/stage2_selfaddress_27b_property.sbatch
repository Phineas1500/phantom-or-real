#!/bin/bash
#SBATCH -A scholar
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --job-name=selfaddress_l
#SBATCH --output=slurm_logs/selfaddress_l_%j.out
#SBATCH --error=slurm_logs/selfaddress_l_%j.err

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
echo "Node: $(hostname)"; echo "Started: $(date)"; echo "Mode: ${SELFADDRESS_MODE:?set SELFADDRESS_MODE=L0 or 0..3}"

if [ "$SELFADDRESS_MODE" = "L0" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --selfaddress-calibration \
        --rank-list 8 \
        --shard-index 0 --shard-count 1 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
elif [ "${SELFADDRESS_MODE#C}" != "$SELFADDRESS_MODE" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --selfaddress-loo \
        --rank-list 8 \
        --shard-index "${SELFADDRESS_MODE#C}" --shard-count 4 \
        --per-height 32 --selection-seed 20260812 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
elif [ "${SELFADDRESS_MODE#R}" != "$SELFADDRESS_MODE" ] && [ "$SELFADDRESS_MODE" != "RG" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --selfaddress-loo \
        --rank-list 8 \
        --shard-index "${SELFADDRESS_MODE#R}" --shard-count 4 \
        --per-height 32 --selection-seed 20260817 \
        --exclude-rows "3073,3290,3415,4322,4675,6188,6327,8035,8298,8874,9549,10079,10714,$(cat docs/l3_exclude_rows.txt)" \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
elif [ "$SELFADDRESS_MODE" = "RG" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --selfaddress --gold-only \
        --rank-list 8 \
        --percand-samples 8 \
        --shard-index 0 --shard-count 1 \
        --per-height 32 --selection-seed 20260817 \
        --exclude-rows "3073,3290,3415,4322,4675,6188,6327,8035,8298,8874,9549,10079,10714,$(cat docs/l3_exclude_rows.txt)" \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
elif [ "$SELFADDRESS_MODE" = "G" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --selfaddress --gold-only \
        --rank-list 8 \
        --percand-samples 8 \
        --shard-index 0 --shard-count 1 \
        --per-height 32 --selection-seed 20260812 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
elif [ "$SELFADDRESS_MODE" = "P0" ] || [ "$SELFADDRESS_MODE" = "P1" ]; then
    python -u scripts/stage2_rank_k_guard_v2.py \
        --selfaddress-prime \
        --rank-list 8 \
        --shard-index "${SELFADDRESS_MODE#P}" --shard-count 2 \
        --per-height 32 --selection-seed 20260812 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
else
    python -u scripts/stage2_rank_k_guard_v2.py \
        --selfaddress \
        --rank-list 8 \
        --shard-index "$SELFADDRESS_MODE" --shard-count "${SELFADDRESS_COUNT:-4}" \
        --per-height 32 --selection-seed 20260812 \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
fi

echo "Finished: $(date)"
