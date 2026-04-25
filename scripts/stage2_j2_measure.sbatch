#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_j2_measure
#SBATCH --output=slurm_logs/stage2_j2_measure_%j.out
#SBATCH --error=slurm_logs/stage2_j2_measure_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/pilots

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

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

python scripts/stage2_j2_measure.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --height 4 \
    --rows 64 \
    --model google/gemma-3-27b-it \
    --layer 30 \
    --n-devices 2 \
    --n-ctx 4096 \
    --batch-sizes 1,4,8,16,32,64 \
    --sae-release gemma-scope-2-27b-it-res-all \
    --sae-ids layer_30_width_16k_l0_small,layer_30_width_262k_l0_small \
    --sae-batch-sizes 32,128,512 \
    --sae-device cuda:0 \
    --sae-dtype bfloat16 \
    --sae-top-k 128 \
    --out-dir results/stage2/pilots

echo "Finished: $(date)"
