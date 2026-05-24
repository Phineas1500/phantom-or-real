#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_local_dict_recon
#SBATCH --output=slurm_logs/qwen35_local_dict_recon_%j.out
#SBATCH --error=slurm_logs/qwen35_local_dict_recon_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs docs results/stage2/local_dictionary_components

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

FEATURE_DIR=results/stage2/sae_features
DICT_DIR=results/stage2/local_dictionaries
COMP_DIR=results/stage2/local_dictionary_components
LAYER="${LAYER:-53}"
D_SAE="${D_SAE:-4096}"
LATENT_TOP_K="${LATENT_TOP_K:-64}"
FEATURE_TOP_K="${FEATURE_TOP_K:-64}"
C_VALUES=0.0003,0.001,0.003,0.01,0.03,0.1

run_mlpout_recon() {
    local split_family="$1"
    local dict_id="local_topk_mlp_out_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${split_family}"
    local component_id="qwen35_l${LAYER}_mlpout_${dict_id}"
    local output="docs/qwen_scope_local_mlpout_reconstruction_probe_27b_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${split_family}.json"

    test -f "$DICT_DIR/${dict_id}.safetensors"
    test -f "$DICT_DIR/${dict_id}.meta.json"

    python -u scripts/stage2_local_dictionary_reconstruction_diagnostics.py \
        --dictionary "$DICT_DIR/${dict_id}.safetensors" \
        --dictionary-meta "$DICT_DIR/${dict_id}.meta.json" \
        --feature-prefix "$FEATURE_DIR/qwen35_27b_infer_property_L${LAYER}_mlp_out_${dict_id}_top${FEATURE_TOP_K}" \
        --feature-prefix "$FEATURE_DIR/qwen35_27b_infer_subtype_L${LAYER}_mlp_out_${dict_id}_top${FEATURE_TOP_K}" \
        --out-dir "$COMP_DIR" \
        --component-id "$component_id" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --device cuda:0 \
        --dtype bfloat16 \
        --output-dtype bfloat16 \
        --chunk-size 256 \
        --c-values "$C_VALUES" \
        --solver liblinear \
        --bootstrap-samples 200 \
        --skip-target-probe \
        --output "$output"
}

run_transcoder_recon() {
    local split_family="$1"
    local source_site="mlp_in_weighted"
    local target_site="mlp_out"
    local dict_id="local_topk_tc_${source_site}_to_${target_site}_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${split_family}"
    local component_id="qwen35_l${LAYER}_localtc_${dict_id}"
    local output="docs/qwen_scope_local_transcoder_component_probe_27b_l${LAYER}_w${D_SAE}_k${LATENT_TOP_K}_${split_family}.json"

    test -f "$DICT_DIR/${dict_id}.safetensors"
    test -f "$DICT_DIR/${dict_id}.meta.json"

    python -u scripts/stage2_local_dictionary_reconstruction_diagnostics.py \
        --dictionary "$DICT_DIR/${dict_id}.safetensors" \
        --dictionary-meta "$DICT_DIR/${dict_id}.meta.json" \
        --feature-prefix "$FEATURE_DIR/qwen35_27b_infer_property_L${LAYER}_${source_site}_${dict_id}_top${FEATURE_TOP_K}" \
        --feature-prefix "$FEATURE_DIR/qwen35_27b_infer_subtype_L${LAYER}_${source_site}_${dict_id}_top${FEATURE_TOP_K}" \
        --out-dir "$COMP_DIR" \
        --component-id "$component_id" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split_family" \
        --device cuda:0 \
        --dtype bfloat16 \
        --output-dtype bfloat16 \
        --chunk-size 256 \
        --c-values "$C_VALUES" \
        --solver liblinear \
        --bootstrap-samples 200 \
        --skip-target-probe \
        --output "$output"
}

run_mlpout_recon s1
run_mlpout_recon s3
run_transcoder_recon s1
run_transcoder_recon s3

nvidia-smi
echo "Finished: $(date)"
