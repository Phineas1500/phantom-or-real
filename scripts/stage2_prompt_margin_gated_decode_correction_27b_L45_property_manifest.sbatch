#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --job-name=prompt_margin_gate
#SBATCH --output=slurm_logs/prompt_margin_gate_%j.out
#SBATCH --error=slurm_logs/prompt_margin_gate_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/decode_time docs

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

DECODE_LAYER="${DECODE_LAYER:-45}"
PROMPT_GVF_THRESHOLD="${PROMPT_GVF_THRESHOLD:--15}"
DECODE_STRENGTHS="${DECODE_STRENGTHS:-1}"
DECODE_CONDITIONS="${DECODE_CONDITIONS:-baseline,prompt_margin_raw,prompt_margin_orthogonal,prompt_margin_gaussian}"
DECODE_MAX_NEW_TOKENS="${DECODE_MAX_NEW_TOKENS:-96}"
DECODE_SUFFIX="${DECODE_SUFFIX:-prompt_margin_gated_decode_correction_27b_l${DECODE_LAYER}_property_manifest_gvfltneg15}"

python -u scripts/stage2_prompt_margin_gated_decode_correction.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --prefix-trajectory-jsonl results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl \
    --prefix-checkpoint 0 \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --layer "$DECODE_LAYER" \
    --activation-dir results/stage2/activations \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --prompt-gold-vs-foil-threshold "$PROMPT_GVF_THRESHOLD" \
    --conditions "$DECODE_CONDITIONS" \
    --strengths "$DECODE_STRENGTHS" \
    --max-new-tokens "$DECODE_MAX_NEW_TOKENS" \
    --temperature 0 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl "results/stage2/decode_time/${DECODE_SUFFIX}.jsonl" \
    --direction-output "results/stage2/decode_time/${DECODE_SUFFIX}_direction.npz" \
    --output "docs/${DECODE_SUFFIX}.json"

nvidia-smi
echo "Finished: $(date)"
