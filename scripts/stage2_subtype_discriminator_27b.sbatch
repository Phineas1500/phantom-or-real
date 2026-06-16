#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=subtype_disc
#SBATCH --output=slurm_logs/subtype_disc_%j.out
#SBATCH --error=slurm_logs/subtype_disc_%j.err

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

python -u scripts/stage2_subtype_discriminator.py \
    --samples-per-row "${SUBTYPE_DISC_SAMPLES:-8}" \
    --ladder-layers "${SUBTYPE_DISC_LADDER:-15,20,25,30,35,40,45,50,53}" \
    --old-trio-layers "${SUBTYPE_DISC_OLD_TRIO:-30,40,45}" \
    --top-offtrio-layers "${SUBTYPE_DISC_TOP_OFFTRIO:-3}" \
    --rank-k "${SUBTYPE_DISC_RANK_K:-4}" \
    --n-devices 2 --n-ctx 4096 --dtype bfloat16 --load-mode no-processing

echo "Finished: $(date)"
