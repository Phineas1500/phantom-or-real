#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --job-name=qwen35_dtype_sanity
#SBATCH --output=slurm_logs/qwen35_dtype_sanity_%j.out
#SBATCH --error=slurm_logs/qwen35_dtype_sanity_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/dtype_sanity_features/qwen_scope docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

OUT_DIR=results/stage2/dtype_sanity_features/qwen_scope
COMMON_ARGS=(
  --out-dir "$OUT_DIR"
  --layer 45
  --local-files-only
  --device cuda:0
  --dtype float32
  --chunk-size 64
  --skip 0
  --limit 512
)

for TASK in infer_property infer_subtype; do
  python scripts/stage2_extract_qwen_scope_features.py     --activation-prefix "results/stage2/activations/qwen35_27b_${TASK}_L45"     --sae-repo-id Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50     "${COMMON_ARGS[@]}"

  python scripts/stage2_extract_qwen_scope_features.py     --activation-prefix "results/stage2/activations/qwen35_27b_${TASK}_L45"     --sae-repo-id Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100     "${COMMON_ARGS[@]}"
done

python scripts/stage2_sparse_dtype_sanity_report.py   --preset qwen35_27b   --stage2-root results/stage2   --dtype-root "$OUT_DIR"   --output docs/qwen_scope_sparse_dtype_sanity_27b.json
