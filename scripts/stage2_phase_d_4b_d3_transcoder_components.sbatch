#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage2_d3_4b_tccomp
#SBATCH --output=slurm_logs/stage2_d3_4b_tccomp_%j.out
#SBATCH --error=slurm_logs/stage2_d3_4b_tccomp_%j.err
#SBATCH --mail-user=semani@purdue.edu
#SBATCH --mail-type=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p slurm_logs docs results/stage2/transcoder_components_exact

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
nvidia-smi

COMMON_ARGS=(
    --input-activation-dir results/stage2/activations
    --target-activation-dir results/stage2/activations
    --feature-dir results/stage2/sae_features
    --out-dir results/stage2/transcoder_components_exact
    --model-key gemma3_4b
    --tasks infer_property infer_subtype
    --layer 22
    --input-activation-site mlp_in_weighted
    --target-activation-site mlp_out_hook
    --transcoder-release gemma-scope-2-4b-it-transcoders-all
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
    --skip-orientation untransposed
)

for width in 16k 262k; do
    transcoder_id="layer_22_width_${width}_l0_small_affine"
    python -u scripts/stage2_transcoder_component_diagnostics.py \
        "${COMMON_ARGS[@]}" \
        --transcoder-id "$transcoder_id" \
        --split-family s1 \
        --output "docs/transcoder_component_probe_4b_l22_${width}_affine_exact_s1.json"

    python -u scripts/stage2_transcoder_component_diagnostics.py \
        "${COMMON_ARGS[@]}" \
        --transcoder-id "$transcoder_id" \
        --reuse-existing \
        --split-family s3 \
        --output "docs/transcoder_component_probe_4b_l22_${width}_affine_exact_s3_target_symbol.json"
done

nvidia-smi
echo "Finished: $(date)"
