#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage2_tc_hook_audit
#SBATCH --output=slurm_logs/stage2_tc_hook_audit_%j.out
#SBATCH --error=slurm_logs/stage2_tc_hook_audit_%j.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export BD_PATH=/scratch/scholar/$USER/beyond-deduction

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

python -u scripts/stage2_transcoder_hook_audit.py \
    --property-jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --subtype-jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --layer 45 \
    --rows-per-task 8 \
    --batch-size 8 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --output-dtype float32 \
    --drop-parse-failed \
    --transcoder-release gemma-scope-2-27b-it-transcoders-all \
    --transcoder-id layer_45_width_262k_l0_small_affine \
    --device cuda:0 \
    --output docs/transcoder_hook_audit_27b_l45_262k.json

nvidia-smi
echo "Finished: $(date)"
