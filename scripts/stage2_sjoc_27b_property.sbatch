#!/bin/bash
#SBATCH -A scholar
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --job-name=sjoc_m
#SBATCH --output=slurm_logs/sjoc_m_%j.out
#SBATCH --error=slurm_logs/sjoc_m_%j.err

set -euo pipefail
cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2 docs
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
echo "Node: $(hostname)"; echo "Started: $(date)"; echo "Mode: ${SJOC_MODE:?set SJOC_MODE=M0}"

if [ "$SJOC_MODE" = "M0" ]; then
    python -u scripts/stage2_sjoc_m0_capture.py \
        --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing
else
    echo "unknown SJOC_MODE=$SJOC_MODE" >&2; exit 2
fi

echo "Finished: $(date)"
