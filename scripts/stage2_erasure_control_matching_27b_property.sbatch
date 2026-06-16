#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=ctlmatch_erasure
#SBATCH --output=slurm_logs/ctlmatch_erasure_%j.out
#SBATCH --error=slurm_logs/ctlmatch_erasure_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/erasure docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH=/scratch/scholar/$USER/beyond-deduction
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

CTLMATCH_TASK="${CTLMATCH_TASK:-infer_property}"
CTLMATCH_JSONL="${CTLMATCH_JSONL:-results/full/with_errortype/gemma3_27b_infer_property.jsonl}"
CTLMATCH_LAYERS="${CTLMATCH_LAYERS:-15,30,40,45,53}"
CTLMATCH_HEIGHTS="${CTLMATCH_HEIGHTS:-3,4}"
CTLMATCH_PER_HEIGHT_LABEL="${CTLMATCH_PER_HEIGHT_LABEL:-4}"
CTLMATCH_SAMPLES_PER_ROW="${CTLMATCH_SAMPLES_PER_ROW:-4}"
CTLMATCH_CONDITIONS="${CTLMATCH_CONDITIONS:-baseline,erase_raw,erase_height,erase_orthogonal,erase_gaussian}"
CTLMATCH_CONTROL_SCALES="${CTLMATCH_CONTROL_SCALES:-0.25,0.5,1.0}"
CTLMATCH_RAW_SCALE="${CTLMATCH_RAW_SCALE:-1.0}"
CTLMATCH_HEIGHT_SCALE="${CTLMATCH_HEIGHT_SCALE:-1.0}"
CTLMATCH_HEIGHT_CONTROL_MIN="${CTLMATCH_HEIGHT_CONTROL_MIN:-4}"
CTLMATCH_C_VALUES="${CTLMATCH_C_VALUES:-0.01,0.1,1.0,10.0}"
CTLMATCH_TEMPERATURE="${CTLMATCH_TEMPERATURE:-0.7}"
CTLMATCH_MAX_NEW_TOKENS="${CTLMATCH_MAX_NEW_TOKENS:-96}"
CTLMATCH_ROW_SHARD="${CTLMATCH_ROW_SHARD:-}"
CTLMATCH_DO_SAMPLE="${CTLMATCH_DO_SAMPLE:-1}"

if [ -z "$CTLMATCH_ROW_SHARD" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    CTLMATCH_ROW_SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
fi
echo "CTLMATCH_ROW_SHARD=${CTLMATCH_ROW_SHARD:-<none>}"

if [ -z "${CTLMATCH_SUFFIX:-}" ]; then
    if [ -n "$CTLMATCH_ROW_SHARD" ]; then
        SHARD_TAG="${CTLMATCH_ROW_SHARD/\//of}"
        CTLMATCH_SUFFIX="erasure_control_matching_27b_property_k4_shard${SHARD_TAG}"
    else
        CTLMATCH_SUFFIX="erasure_control_matching_27b_property_k4"
    fi
fi

CTLMATCH_SAMPLE_ARGS=()
if [ "$CTLMATCH_DO_SAMPLE" = "1" ]; then
    CTLMATCH_SAMPLE_ARGS=(--do-sample)
else
    CTLMATCH_SAMPLE_ARGS=(--no-do-sample)
fi

CTLMATCH_SHARD_ARGS=()
if [ -n "$CTLMATCH_ROW_SHARD" ]; then
    CTLMATCH_SHARD_ARGS=(--row-shard "$CTLMATCH_ROW_SHARD")
fi

python -u scripts/stage2_erasure_control_matching.py \
    --jsonl "$CTLMATCH_JSONL" \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task "$CTLMATCH_TASK" \
    --layers "$CTLMATCH_LAYERS" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights "$CTLMATCH_HEIGHTS" \
    --per-height-label "$CTLMATCH_PER_HEIGHT_LABEL" \
    --samples-per-row "$CTLMATCH_SAMPLES_PER_ROW" \
    --conditions "$CTLMATCH_CONDITIONS" \
    --control-scales "$CTLMATCH_CONTROL_SCALES" \
    --raw-scale "$CTLMATCH_RAW_SCALE" \
    --height-scale "$CTLMATCH_HEIGHT_SCALE" \
    --height-control-min "$CTLMATCH_HEIGHT_CONTROL_MIN" \
    --c-values "$CTLMATCH_C_VALUES" \
    --max-new-tokens "$CTLMATCH_MAX_NEW_TOKENS" \
    --temperature "$CTLMATCH_TEMPERATURE" \
    ${CTLMATCH_SAMPLE_ARGS[@]+"${CTLMATCH_SAMPLE_ARGS[@]}"} \
    ${CTLMATCH_SHARD_ARGS[@]+"${CTLMATCH_SHARD_ARGS[@]}"} \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl "results/stage2/erasure/${CTLMATCH_SUFFIX}.jsonl" \
    --direction-output "results/stage2/erasure/${CTLMATCH_SUFFIX}_directions.npz" \
    --output "docs/${CTLMATCH_SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
