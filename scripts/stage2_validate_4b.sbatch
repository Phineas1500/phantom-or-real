#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --job-name=stage2_4b_validate
#SBATCH --output=slurm_logs/stage2_4b_validate_%j.out
#SBATCH --error=slurm_logs/stage2_4b_validate_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/scratch/scholar/$USER/587_proj/phantom-or-real}"
mkdir -p slurm_logs docs results/stage2/pilots

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

echo "Node: $(hostname)"
echo "Started: $(date)"

# Validate the A.0 smoke extraction artifact.
python scripts/validate_activations.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --output docs/stage2_pilot_validation_4b_infer_property_L17.json \
  --hf-cache "$HF_HOME" \
  --n-ctx 4096 \
  --limit 10 \
  --layers 17 \
  --activation-dir results/stage2/pilots \
  --strict

# Validate full property JSONL/model/tokenizer/prompt invariants.
python scripts/validate_activations.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --output docs/stage2_equivalence_4b_infer_property.json \
  --hf-cache "$HF_HOME" \
  --n-ctx 4096 \
  --strict

# Validate full subtype JSONL/model/tokenizer/prompt invariants.
python scripts/validate_activations.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_subtype.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --output docs/stage2_equivalence_4b_infer_subtype.json \
  --hf-cache "$HF_HOME" \
  --n-ctx 4096 \
  --strict

echo "Finished: $(date)"