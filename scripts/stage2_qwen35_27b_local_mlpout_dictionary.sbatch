#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_local_mlpout_dict
#SBATCH --output=slurm_logs/qwen35_local_mlpout_dict_%j.out
#SBATCH --error=slurm_logs/qwen35_local_mlpout_dict_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/sae_features results/stage2/local_dictionaries docs

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

OUT_DIR=results/stage2/activations
FEATURE_DIR=results/stage2/sae_features
DICT_DIR=results/stage2/local_dictionaries
LAYER="${LAYER:-53}"
SITE="${SITE:-mlp_out}"
D_SAE="${D_SAE:-4096}"
LATENT_TOP_K="${LATENT_TOP_K:-64}"
FEATURE_TOP_K="${FEATURE_TOP_K:-64}"
STEPS="${STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-512}"
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10

for task in infer_property infer_subtype; do
    prefix="$OUT_DIR/qwen35_27b_${task}_L${LAYER}_${SITE}"
    test -f "${prefix}.safetensors"
    test -f "${prefix}.meta.json"
    test -f "${prefix}.example_ids.jsonl"
done

train_and_probe_split() {
    local split_family="$1"
    local dict_id="local_topk_${SITE}_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${split_family}"
    local output="docs/qwen_scope_local_mlpout_topk_probe_27b_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${split_family}.json"

    echo "Training local dictionary ${dict_id}"
    python -u scripts/stage2_train_local_topk_dictionary.py \
        --activation-prefix "$OUT_DIR/qwen35_27b_infer_property_L${LAYER}_${SITE}" \
        --activation-prefix "$OUT_DIR/qwen35_27b_infer_subtype_L${LAYER}_${SITE}" \
        --out-dir "$FEATURE_DIR" \
        --dictionary-dir "$DICT_DIR" \
        --dictionary-id "$dict_id" \
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

    echo "Probing local dictionary ${dict_id}: ${split_family}"
    python -u scripts/stage2_probe_sae.py \
        --feature-dir "$FEATURE_DIR" \
        --activation-site "$SITE" \
        --model-key qwen35_27b \
        --tasks infer_property infer_subtype \
        --layer "$LAYER" \
        --sae-id "$dict_id" \
        --top-k "$FEATURE_TOP_K" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --c-values "$C_VALUES" \
        --solver liblinear \
        --bootstrap-samples 1000 \
        --output "$output"
}

train_and_probe_split s1
train_and_probe_split s3

nvidia-smi
echo "Finished: $(date)"
