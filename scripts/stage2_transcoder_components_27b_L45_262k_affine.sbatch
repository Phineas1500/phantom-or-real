#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_tc_comp_262k
#SBATCH --output=slurm_logs/stage2_tc_comp_262k_%j.out
#SBATCH --error=slurm_logs/stage2_tc_comp_262k_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/transcoder_components docs

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

COMMON_ARGS=(
    --input-activation-dir results/stage2/activations
    --target-activation-dir results/stage2/activations
    --feature-dir results/stage2/sae_features
    --out-dir results/stage2/transcoder_components
    --model-key gemma3_27b
    --tasks infer_property infer_subtype
    --layer 45
    --input-activation-site mlp_in
    --target-activation-site mlp_out
    --transcoder-release gemma-scope-2-27b-it-transcoders-all
    --transcoder-id layer_45_width_262k_l0_small_affine
    --top-k 128
    --splits results/stage2/splits.jsonl
    --device cuda:0
    --dtype bfloat16
    --output-dtype bfloat16
    --chunk-size 128
    --verify-dense-rows 4
    --solver liblinear
    --c-values 1.0
    --bootstrap-samples 0
    --skip-target-probe
)

python -u scripts/stage2_transcoder_component_diagnostics.py \
    "${COMMON_ARGS[@]}" \
    --split-family s1 \
    --output docs/transcoder_component_probe_27b_l45_262k_affine_s1.json

python -u scripts/stage2_transcoder_component_diagnostics.py \
    "${COMMON_ARGS[@]}" \
    --reuse-existing \
    --split-family s3 \
    --output docs/transcoder_component_probe_27b_l45_262k_affine_s3_target_symbol.json

nvidia-smi
echo "Finished: $(date)"
