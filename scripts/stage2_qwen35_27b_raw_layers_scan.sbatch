#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen35_raw_layers
#SBATCH --output=slurm_logs/qwen35_raw_layers_%j.out
#SBATCH --error=slurm_logs/qwen35_raw_layers_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs docs results/stage2/activations results/stage2/probes

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

python scripts/stage2_validate_qwen_outputs.py \
    results/full/with_errortype/qwen35_27b_infer_property.jsonl \
    results/full/with_errortype/qwen35_27b_infer_subtype.jsonl

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
LAYERS="${LAYERS:-16,31,40,53}"
BATCH_SIZE="${BATCH_SIZE:-8}"
C_VALUES="${C_VALUES:-0.01,0.1,1,10}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

python scripts/stage2_extract_hf.py \
    --jsonl results/full/with_errortype/qwen35_27b_infer_property.jsonl \
    --model "$MODEL" \
    --model-key qwen35_27b \
    --layers "$LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --n-ctx 4096 \
    --drop-parse-failed \
    --disable-thinking \
    --out-dir results/stage2/activations \
    "${LIMIT_ARGS[@]}"

python scripts/stage2_extract_hf.py \
    --jsonl results/full/with_errortype/qwen35_27b_infer_subtype.jsonl \
    --model "$MODEL" \
    --model-key qwen35_27b \
    --layers "$LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --n-ctx 4096 \
    --drop-parse-failed \
    --disable-thinking \
    --out-dir results/stage2/activations \
    "${LIMIT_ARGS[@]}"

TASKS=(infer_property infer_subtype)

python -u scripts/stage2_probe_raw.py \
    --activation-dir results/stage2/activations \
    --model-key qwen35_27b \
    --tasks "${TASKS[@]}" \
    --layers "$LAYERS" \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --c-values "$C_VALUES" \
    --bootstrap-samples 1000 \
    --save-probes-dir results/stage2/probes/qwen35_27b_layers_16_31_40_53_s1 \
    --output docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s1.json

python -u scripts/stage2_probe_raw.py \
    --activation-dir results/stage2/activations \
    --model-key qwen35_27b \
    --tasks "${TASKS[@]}" \
    --layers "$LAYERS" \
    --splits results/stage2/splits.jsonl \
    --split-family s3 \
    --c-values "$C_VALUES" \
    --bootstrap-samples 1000 \
    --save-probes-dir results/stage2/probes/qwen35_27b_layers_16_31_40_53_s3 \
    --output docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s3.json

echo "Finished: $(date)"
