#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --job-name=ns4b_infer_all
#SBATCH --output=slurm_logs/ns4b_infer_all_%j.out
#SBATCH --error=slurm_logs/ns4b_infer_all_%j.err
#SBATCH --mail-user=semani@purdue.edu
#SBATCH --mail-type=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p slurm_logs results/stage2/namescramble_infer

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

for cond in nonce natural; do
  for task in infer_property infer_subtype; do
    for h in 1 2 3 4; do
      echo "Running cond=${cond} task=${task} h=${h}"
      python scripts/stage2_namescramble_infer.py \
        --input-jsonl "data/stage2/namescramble/${cond}/${task}_h${h}.jsonl" \
        --output-jsonl "results/stage2/namescramble_infer/${cond}/${task}_h${h}.jsonl" \
        --model google/gemma-3-4b-it \
        --concurrency 16 \
        --max-tokens 512 \
        --temperature 0.0 \
        --max-attempts 4
    done
  done
done

echo "Finished: $(date)"
