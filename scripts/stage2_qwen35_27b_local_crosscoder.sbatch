#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_local_crosscoder
#SBATCH --output=slurm_logs/qwen35_local_crosscoder_%j.out
#SBATCH --error=slurm_logs/qwen35_local_crosscoder_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/crosscoder_features results/stage2/local_dictionaries docs

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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

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

ACTIVATION_DIR=results/stage2/activations
FEATURE_DIR=results/stage2/crosscoder_features
DICT_DIR=results/stage2/local_dictionaries
LAYERS="${LAYERS:-16,31,40,53}"
D_SAE="${D_SAE:-4096}"
LATENT_TOP_K="${LATENT_TOP_K:-64}"
FEATURE_TOP_K="${FEATURE_TOP_K:-64}"
STEPS="${STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-256}"
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10
RAW_C_VALUES="${RAW_C_VALUES:-0.0003,0.001,0.003,0.01}"

IFS=',' read -ra LAYER_ARRAY <<< "$LAYERS"
for task in infer_property infer_subtype; do
    for layer in "${LAYER_ARRAY[@]}"; do
        layer="$(printf '%s' "$layer" | xargs)"
        prefix="$ACTIVATION_DIR/qwen35_27b_${task}_L${layer}"
        test -f "${prefix}.safetensors"
        test -f "${prefix}.meta.json"
        test -f "${prefix}.example_ids.jsonl"
    done
done

run_raw_concat() {
    local split_family="$1"
    local output="$2"
    echo "Probing raw concat layers ${LAYERS}: ${split_family}"
    python -u scripts/stage2_probe_raw_concat.py \
        --activation-dir "$ACTIVATION_DIR" \
        --model-key qwen35_27b \
        --tasks infer_property infer_subtype \
        --layers "$LAYERS" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --c-values "$RAW_C_VALUES" \
        --solver liblinear \
        --max-iter 1000 \
        --bootstrap-samples 1000 \
        --output "$output"
}

train_and_probe_split() {
    local split_family="$1"
    local crosscoder_id="local_topk_layers_16_31_40_53_w${D_SAE}_k${LATENT_TOP_K}_${split_family}"
    local output="docs/qwen_scope_local_crosscoder_probe_27b_layers_16_31_40_53_w${D_SAE}_k${LATENT_TOP_K}_${split_family}.json"

    echo "Training local crosscoder ${crosscoder_id}"
    python -u scripts/stage2_train_local_topk_crosscoder.py \
        --activation-dir "$ACTIVATION_DIR" \
        --out-dir "$FEATURE_DIR" \
        --dictionary-dir "$DICT_DIR" \
        --model-key qwen35_27b \
        --tasks infer_property infer_subtype \
        --layers "$LAYERS" \
        --crosscoder-id "$crosscoder_id" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --d-sae "$D_SAE" \
        --latent-top-k "$LATENT_TOP_K" \
        --feature-top-k "$FEATURE_TOP_K" \
        --steps "$STEPS" \
        --batch-size "$BATCH_SIZE" \
        --encode-batch-size "$ENCODE_BATCH_SIZE" \
        --device cuda:0 \
        --amp-dtype bfloat16 \
        --log-every 100

    echo "Probing local crosscoder ${crosscoder_id}: ${split_family}"
    python -u scripts/stage2_probe_crosscoder.py \
        --feature-dir "$FEATURE_DIR" \
        --model-key qwen35_27b \
        --tasks infer_property infer_subtype \
        --crosscoder-id "$crosscoder_id" \
        --top-k "$FEATURE_TOP_K" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --c-values "$RAW_C_VALUES" \
        --solver liblinear \
        --max-iter 1000 \
        --bootstrap-samples 1000 \
        --output "$output"
}

train_and_probe_split s1
train_and_probe_split s3
run_raw_concat s1 docs/qwen_scope_raw_concat_probe_27b_layers_16_31_40_53_s1.json
run_raw_concat s3 docs/qwen_scope_raw_concat_probe_27b_layers_16_31_40_53_s3.json

nvidia-smi
echo "Finished: $(date)"
