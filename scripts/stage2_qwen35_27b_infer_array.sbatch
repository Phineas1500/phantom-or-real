#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_27b_infer_array
#SBATCH --array=0-21%2
#SBATCH --output=slurm_logs/qwen35_27b_infer_array_%A_%a.out
#SBATCH --error=slurm_logs/qwen35_27b_infer_array_%A_%a.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs

# Full data counts are 1000/2000/3000/5000 per task for heights 1..4.
# SHARD_SIZE=1000 gives 22 array tasks total across property and ontology.
SPECS=(
  "property 1 0 1000"
  "property 2 0 1000"
  "property 2 1000 1000"
  "property 3 0 1000"
  "property 3 1000 1000"
  "property 3 2000 1000"
  "property 4 0 1000"
  "property 4 1000 1000"
  "property 4 2000 1000"
  "property 4 3000 1000"
  "property 4 4000 1000"
  "ontology 1 0 1000"
  "ontology 2 0 1000"
  "ontology 2 1000 1000"
  "ontology 3 0 1000"
  "ontology 3 1000 1000"
  "ontology 3 2000 1000"
  "ontology 4 0 1000"
  "ontology 4 1000 1000"
  "ontology 4 2000 1000"
  "ontology 4 3000 1000"
  "ontology 4 4000 1000"
)

read -r TASK HEIGHT SKIP LIMIT <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

export TASKS="$TASK"
export HEIGHTS="$HEIGHT"
export SKIP="$SKIP"
export LIMIT="$LIMIT"
export OUTPUT_DIR="${OUTPUT_DIR:-results/full/with_errortype/qwen35_27b_shards}"
export MODEL_SLUG="qwen35_27b_${TASK}_h${HEIGHT}_s${SKIP}_n${LIMIT}"
export MAKE_SPLITS=0
export MAX_TOKENS="${MAX_TOKENS:-256}"
export CONCURRENCY="${CONCURRENCY:-32}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: TASK=$TASK HEIGHT=$HEIGHT SKIP=$SKIP LIMIT=$LIMIT"
echo "MODEL_SLUG=$MODEL_SLUG OUTPUT_DIR=$OUTPUT_DIR"

bash scripts/stage2_qwen35_27b_infer_vllm.sbatch
