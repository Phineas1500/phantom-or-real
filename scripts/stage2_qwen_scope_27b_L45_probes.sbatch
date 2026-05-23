#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen_scope_27b_l45_probes
#SBATCH --output=slurm_logs/qwen_scope_27b_l45_probes_%j.out
#SBATCH --error=slurm_logs/qwen_scope_27b_l45_probes_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs docs results/stage2/probes results/stage2/sae_reconstructions

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

LAYER="${LAYER:-45}"
LAYER_TAG="l${LAYER}"
SPLIT_ARGS=(${SPLITS:-s1 s3})
TASKS=(infer_property infer_subtype)

for split in "${SPLIT_ARGS[@]}"; do
    python scripts/stage2_probe_raw.py \
        --activation-dir results/stage2/activations \
        --model-key qwen35_27b \
        --tasks "${TASKS[@]}" \
        --layers "$LAYER" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --output "docs/qwen_scope_raw_probe_27b_${LAYER_TAG}_${split}.json" \
        --save-probes-dir "results/stage2/probes/qwen35_27b_${LAYER_TAG}_${split}" \
        --bootstrap-samples 1000

    python scripts/stage2_probe_sae.py \
        --feature-dir results/stage2/sae_features \
        --model-key qwen35_27b \
        --tasks "${TASKS[@]}" \
        --layer "$LAYER" \
        --sae-id qwenscope_qwen35_27b_w80k_l0_50 \
        --top-k 50 \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --output "docs/qwen_scope_sae_probe_27b_${LAYER_TAG}_w80k_l0_50_${split}.json" \
        --bootstrap-samples 1000

    python scripts/stage2_probe_sae.py \
        --feature-dir results/stage2/sae_features \
        --model-key qwen35_27b \
        --tasks "${TASKS[@]}" \
        --layer "$LAYER" \
        --sae-id qwenscope_qwen35_27b_w80k_l0_100 \
        --top-k 100 \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --output "docs/qwen_scope_sae_probe_27b_${LAYER_TAG}_w80k_l0_100_${split}.json" \
        --bootstrap-samples 1000

    python scripts/stage2_qwen_scope_reconstruction_diagnostics.py \
        --activation-dir results/stage2/activations \
        --feature-dir results/stage2/sae_features \
        --out-dir results/stage2/sae_reconstructions \
        --model-key qwen35_27b \
        --tasks "${TASKS[@]}" \
        --layer "$LAYER" \
        --sae-repo-id Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
        --sae-id qwenscope_qwen35_27b_w80k_l0_50 \
        --top-k 50 \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --output "docs/qwen_scope_reconstruction_probe_27b_${LAYER_TAG}_w80k_l0_50_${split}.json" \
        --device cuda:0 \
        --dtype bfloat16 \
        --output-dtype bfloat16 \
        --chunk-size 512 \
        --verify-dense-rows 8 \
        --solver lbfgs \
        --bootstrap-samples 1000

    python scripts/stage2_qwen_scope_reconstruction_diagnostics.py \
        --activation-dir results/stage2/activations \
        --feature-dir results/stage2/sae_features \
        --out-dir results/stage2/sae_reconstructions \
        --model-key qwen35_27b \
        --tasks "${TASKS[@]}" \
        --layer "$LAYER" \
        --sae-repo-id Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100 \
        --sae-id qwenscope_qwen35_27b_w80k_l0_100 \
        --top-k 100 \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --output "docs/qwen_scope_reconstruction_probe_27b_${LAYER_TAG}_w80k_l0_100_${split}.json" \
        --device cuda:0 \
        --dtype bfloat16 \
        --output-dtype bfloat16 \
        --chunk-size 512 \
        --verify-dense-rows 8 \
        --solver lbfgs \
        --bootstrap-samples 1000
done

echo "Finished: $(date)"
