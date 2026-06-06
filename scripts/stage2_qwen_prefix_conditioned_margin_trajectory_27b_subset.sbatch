#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen_prefix_margin
#SBATCH --output=slurm_logs/qwen_prefix_margin_%j.out
#SBATCH --error=slurm_logs/qwen_prefix_margin_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/decode_time docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH="${BD_PATH:-/scratch/scholar/$USER/beyond-deduction}"
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

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
LIMIT="${LIMIT:-14}"
CHECKPOINTS="${CHECKPOINTS:-0,1,4,8,16,32,64,final}"

python -u scripts/stage2_qwen_prefix_conditioned_margin_trajectory.py \
    --recognition-jsonl "${RECOGNITION_JSONL:-results/stage2/qwen_causal/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.jsonl}" \
    --source-jsonl "${SOURCE_JSONL:-results/full/with_errortype/qwen35_27b_infer_subtype.jsonl}" \
    --model "$MODEL" \
    --model-key qwen35_27b \
    --task infer_subtype \
    --height 4 \
    --limit "$LIMIT" \
    --require-recognition-gap \
    --checkpoints "$CHECKPOINTS" \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --device-map auto \
    --disable-thinking \
    --out-jsonl "${OUT_JSONL:-results/stage2/decode_time/qwen_prefix_conditioned_margin_trajectory_h4_subset.jsonl}" \
    --output "${OUTPUT_JSON:-docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.json}" \
    --md-output "${OUTPUT_MD:-docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.md}"

nvidia-smi
echo "Finished: $(date)"
