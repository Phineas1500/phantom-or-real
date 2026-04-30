#!/usr/bin/env bash
set -euo pipefail

if [ -n "${PHANTOM_ROOT:-}" ]; then
    cd "$PHANTOM_ROOT"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p docs results/stage2/activations results/stage2/steering

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=disabled

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

python3 -u scripts/stage2_steer_forced_choice_direction.py \
    --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
    --model google/gemma-3-4b-it \
    --model-key gemma3_4b \
    --task infer_property \
    --layer 22 \
    --activation-dir results/stage2/activations \
    --activation-prefix results/stage2/activations/gemma3_4b_infer_property_L22_forced_choice \
    --batch-size 16 \
    --splits results/stage2/splits_4b_property.jsonl \
    --split-family s1 \
    --option-seed 20260430 \
    --heights 3,4 \
    --per-height-label 2 \
    --conditions baseline,toward_gold,away_gold,orthogonal \
    --strengths 0.5,1 \
    --intervention-scope last_token_each_forward \
    --max-new-tokens 16 \
    --temperature 0 \
    --n-devices 1 \
    --n-ctx 2048 \
    --dtype bfloat16 \
    --output-dtype bfloat16 \
    --load-mode no-processing \
    --resume \
    --out-jsonl results/stage2/steering/forced_choice_4b_l22_property_smoke.jsonl \
    --direction-output results/stage2/steering/forced_choice_4b_l22_property_direction.npz \
    --output docs/forced_choice_steering_4b_l22_property_smoke.json

echo "Finished: $(date)"
