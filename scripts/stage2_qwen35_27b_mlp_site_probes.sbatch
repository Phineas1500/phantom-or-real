#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen35_mlp_site_probes
#SBATCH --output=slurm_logs/qwen35_mlp_site_probes_%j.out
#SBATCH --error=slurm_logs/qwen35_mlp_site_probes_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/activations docs

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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
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

python scripts/stage2_validate_qwen_outputs.py     results/full/with_errortype/qwen35_27b_infer_property.jsonl     results/full/with_errortype/qwen35_27b_infer_subtype.jsonl

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
LAYERS="${LAYERS:-45,53}"
SITES="${SITES:-mlp_in_weighted,mlp_out}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LAYER_TAG="l$(printf '%s' "$LAYERS" | tr ',' '_' | tr -d '[:space:]')"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

if [[ "${SKIP_EXTRACTION:-0}" != "1" ]]; then
for TASK in infer_property infer_subtype; do
    JSONL="results/full/with_errortype/qwen35_27b_${TASK}.jsonl"
    python scripts/stage2_extract_hf.py         --jsonl "$JSONL"         --model "$MODEL"         --model-key qwen35_27b         --layers "$LAYERS"         --activation-sites "$SITES"         --batch-size "$BATCH_SIZE"         --n-ctx 4096         --drop-parse-failed         --disable-thinking         --out-dir results/stage2/activations         "${LIMIT_ARGS[@]}"
    done
fi

IFS=',' read -ra SITE_ARRAY <<< "$SITES"
for SITE in "${SITE_ARRAY[@]}"; do
    SITE="$(printf '%s' "$SITE" | tr '-' '_' | xargs)"
    python scripts/stage2_probe_raw.py         --activation-dir results/stage2/activations         --activation-site "$SITE"         --model-key qwen35_27b         --tasks infer_property infer_subtype         --layers "$LAYERS"         --splits results/stage2/splits.jsonl         --split-family s1         --output "docs/qwen_scope_raw_probe_27b_${LAYER_TAG}_${SITE}_s1.json"

    python scripts/stage2_probe_raw.py         --activation-dir results/stage2/activations         --activation-site "$SITE"         --model-key qwen35_27b         --tasks infer_property infer_subtype         --layers "$LAYERS"         --splits results/stage2/splits.jsonl         --split-family s3         --output "docs/qwen_scope_raw_probe_27b_${LAYER_TAG}_${SITE}_s3.json"
done

echo "Finished: $(date)"
