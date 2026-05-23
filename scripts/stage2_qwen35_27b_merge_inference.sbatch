#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=qwen35_27b_merge
#SBATCH --output=slurm_logs/qwen35_27b_merge_%j.out
#SBATCH --error=slurm_logs/qwen35_27b_merge_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs docs results/full/with_errortype results/stage2

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export BD_PATH="${BD_PATH:-/scratch/scholar/$USER/beyond-deduction}"

SHARD_ROOT="${SHARD_ROOT:-results/full/with_errortype/qwen35_27b_shards}"

python scripts/stage2_merge_inference_shards.py \
    --shard-root "$SHARD_ROOT" \
    --examples-dir data/full \
    --output-dir results/full/with_errortype \
    --model-slug qwen35_27b \
    --summary docs/qwen35_27b_inference_merge_summary.json

python scripts/stage2_validate_qwen_outputs.py \
    results/full/with_errortype/qwen35_27b_infer_property.jsonl \
    results/full/with_errortype/qwen35_27b_infer_subtype.jsonl

python scripts/stage2_make_splits.py \
    --jsonl-dir results/full/with_errortype \
    --output results/stage2/splits.jsonl \
    --summary docs/stage2_splits_summary.json
