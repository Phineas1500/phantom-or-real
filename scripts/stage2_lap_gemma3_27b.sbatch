#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage2_lap_gemma27b
#SBATCH --output=slurm_logs/stage2_lap_gemma27b_%j.out
#SBATCH --error=slurm_logs/stage2_lap_gemma27b_%j.err

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

MODEL="${MODEL:-google/gemma-3-27b-it}"
LAYERS="${LAYERS:-15,30,45}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_SPLITS="${EVAL_SPLITS:-test,all}"

python -u scripts/stage2_logit_lens_lap.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model "$MODEL" \
    --model-key gemma3_27b \
    --task infer_property \
    --layers "$LAYERS" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --eval-splits "$EVAL_SPLITS" \
    --drop-parse-failed \
    --batch-size "$BATCH_SIZE" \
    --dtype bfloat16 \
    --activation-dtype bfloat16 \
    --device-map auto \
    --output docs/lap_gemma3_27b_infer_property_s1.json

python -u scripts/stage2_logit_lens_lap.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
    --model "$MODEL" \
    --model-key gemma3_27b \
    --task infer_subtype \
    --layers "$LAYERS" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --eval-splits "$EVAL_SPLITS" \
    --drop-parse-failed \
    --batch-size "$BATCH_SIZE" \
    --dtype bfloat16 \
    --activation-dtype bfloat16 \
    --device-map auto \
    --output docs/lap_gemma3_27b_infer_subtype_s1.json

python scripts/stage2_steering_effectiveness_diagnostics.py
python scripts/stage2_causal_abstraction_dashboard.py

nvidia-smi
echo "Finished: $(date)"
