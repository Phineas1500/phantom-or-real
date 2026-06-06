#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --job-name=gemma3_local_tc
#SBATCH --output=slurm_logs/gemma3_local_tc_%j.out
#SBATCH --error=slurm_logs/gemma3_local_tc_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/sae_features results/stage2/local_dictionaries results/stage2/local_dictionary_components docs

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
COMP_DIR=results/stage2/local_dictionary_components
LAYER="${LAYER:-45}"
SOURCE_SITE="${SOURCE_SITE:-mlp_in_weighted}"
TARGET_SITE="${TARGET_SITE:-mlp_out}"
SPLIT_FAMILY="${SPLIT_FAMILY:-s1}"
D_SAE="${D_SAE:-4096}"
LATENT_TOP_K="${LATENT_TOP_K:-64}"
FEATURE_TOP_K="${FEATURE_TOP_K:-64}"
STEPS="${STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-512}"
C_VALUES="${C_VALUES:-0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10}"
COMP_C_VALUES="${COMP_C_VALUES:-0.0003,0.001,0.003,0.01,0.03,0.1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-200}"

for task in infer_property infer_subtype; do
    source_prefix="$OUT_DIR/gemma3_27b_${task}_L${LAYER}_${SOURCE_SITE}"
    target_prefix="$OUT_DIR/gemma3_27b_${task}_L${LAYER}_${TARGET_SITE}"
    test -f "${source_prefix}.safetensors"
    test -f "${source_prefix}.meta.json"
    test -f "${source_prefix}.example_ids.jsonl"
    test -f "${target_prefix}.safetensors"
    test -f "${target_prefix}.meta.json"
    test -f "${target_prefix}.example_ids.jsonl"
done

dict_id="gemma3_local_topk_tc_${SOURCE_SITE}_to_${TARGET_SITE}_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${SPLIT_FAMILY}"
feature_probe_output="docs/gemma3_local_transcoder_probe_27b_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${SPLIT_FAMILY}.json"
recon_output="docs/gemma3_local_transcoder_component_probe_27b_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${SPLIT_FAMILY}.json"
component_id="gemma3_l${LAYER}_localtc_${dict_id}"

echo "Training local Gemma transcoder ${dict_id}"
python -u scripts/stage2_train_local_topk_transcoder.py \
    --source-activation-prefix "$OUT_DIR/gemma3_27b_infer_property_L${LAYER}_${SOURCE_SITE}" \
    --target-activation-prefix "$OUT_DIR/gemma3_27b_infer_property_L${LAYER}_${TARGET_SITE}" \
    --source-activation-prefix "$OUT_DIR/gemma3_27b_infer_subtype_L${LAYER}_${SOURCE_SITE}" \
    --target-activation-prefix "$OUT_DIR/gemma3_27b_infer_subtype_L${LAYER}_${TARGET_SITE}" \
    --out-dir "$FEATURE_DIR" \
    --dictionary-dir "$DICT_DIR" \
    --dictionary-id "$dict_id" \
    --splits results/stage2/splits.jsonl \
    --split-family "$SPLIT_FAMILY" \
    --d-sae "$D_SAE" \
    --latent-top-k "$LATENT_TOP_K" \
    --feature-top-k "$FEATURE_TOP_K" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --encode-batch-size "$ENCODE_BATCH_SIZE" \
    --device cuda:0 \
    --amp-dtype bfloat16 \
    --log-every 100

echo "Probing local transcoder sparse features ${dict_id}"
python -u scripts/stage2_probe_sae.py \
    --feature-dir "$FEATURE_DIR" \
    --activation-site "$SOURCE_SITE" \
    --model-key gemma3_27b \
    --tasks infer_property infer_subtype \
    --layer "$LAYER" \
    --sae-id "$dict_id" \
    --top-k "$FEATURE_TOP_K" \
    --splits results/stage2/splits.jsonl \
    --split-family "$SPLIT_FAMILY" \
    --c-values "$C_VALUES" \
    --solver liblinear \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --output "$feature_probe_output"

echo "Decoding transcoder target/reconstruction/error components for ${dict_id}"
python -u scripts/stage2_local_dictionary_reconstruction_diagnostics.py \
    --dictionary "$DICT_DIR/${dict_id}.safetensors" \
    --dictionary-meta "$DICT_DIR/${dict_id}.meta.json" \
    --feature-prefix "$FEATURE_DIR/gemma3_27b_infer_property_L${LAYER}_${SOURCE_SITE}_${dict_id}_top${FEATURE_TOP_K}" \
    --feature-prefix "$FEATURE_DIR/gemma3_27b_infer_subtype_L${LAYER}_${SOURCE_SITE}_${dict_id}_top${FEATURE_TOP_K}" \
    --out-dir "$COMP_DIR" \
    --component-id "$component_id" \
    --splits results/stage2/splits.jsonl \
    --split-family "$SPLIT_FAMILY" \
    --device cuda:0 \
    --dtype bfloat16 \
    --output-dtype bfloat16 \
    --chunk-size 256 \
    --c-values "$COMP_C_VALUES" \
    --solver liblinear \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --output "$recon_output"

nvidia-smi
echo "Finished: $(date)"
