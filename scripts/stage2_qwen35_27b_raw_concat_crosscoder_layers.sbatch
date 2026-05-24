#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --job-name=qwen35_raw_concat_xcoder
#SBATCH --output=slurm_logs/qwen35_raw_concat_xcoder_%j.out
#SBATCH --error=slurm_logs/qwen35_raw_concat_xcoder_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export TOKENIZERS_PARALLELISM=false
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
LAYERS="${LAYERS:-16,31,40,53}"
RAW_C_VALUES="${RAW_C_VALUES:-0.0003,0.001,0.003,0.01}"

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

run_raw_concat s1 docs/qwen_scope_raw_concat_probe_27b_layers_16_31_40_53_s1.json
run_raw_concat s3 docs/qwen_scope_raw_concat_probe_27b_layers_16_31_40_53_s3.json

nvidia-smi
echo "Finished: $(date)"
