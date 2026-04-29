#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_raw_xc_layers
#SBATCH --output=slurm_logs/stage2_raw_xc_layers_%j.out
#SBATCH --error=slurm_logs/stage2_raw_xc_layers_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

TASKS=(infer_property infer_subtype)
LAYERS=16,31,40,53
C_VALUES=0.01,0.1,1,10

python -u scripts/stage2_probe_raw.py \
    --activation-dir results/stage2/activations \
    --model-key gemma3_27b \
    --tasks "${TASKS[@]}" \
    --layers "${LAYERS}" \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --c-values "${C_VALUES}" \
    --bootstrap-samples 1000 \
    --output docs/raw_probe_27b_crosscoder_layers_individual_s1.json

python -u scripts/stage2_probe_raw.py \
    --activation-dir results/stage2/activations \
    --model-key gemma3_27b \
    --tasks "${TASKS[@]}" \
    --layers "${LAYERS}" \
    --splits results/stage2/splits.jsonl \
    --split-family s3 \
    --c-values "${C_VALUES}" \
    --bootstrap-samples 1000 \
    --output docs/raw_probe_27b_crosscoder_layers_individual_s3_target_symbol.json

echo "Finished: $(date)"
