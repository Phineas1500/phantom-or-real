#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen35_spbundle
#SBATCH --output=slurm_logs/qwen35_spbundle_%j.out
#SBATCH --error=slurm_logs/qwen35_spbundle_%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/steering docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export BD_PATH=/scratch/scholar/$USER/beyond-deduction
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
TASK="${TASK:-infer_property}"
TASK_LABEL="${TASK#infer_}"
SAE_L0="${SAE_L0:-100}"
TOP_K="${TOP_K:-100}"
TOP_POSITIVE="${TOP_POSITIVE:-25}"
TOP_NEGATIVE="${TOP_NEGATIVE:-25}"
LAYER_TAG="l${LAYER}"
SAE_ID="qwenscope_qwen35_27b_w80k_l0_${SAE_L0}"
SAE_REPO_ID="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_${SAE_L0}"
JSONL="results/full/with_errortype/qwen35_27b_${TASK}.jsonl"
OUT_STEM="qwen35_scope_${LAYER_TAG}_l0_${SAE_L0}_sparse_bundle_${TASK_LABEL}_smoke"

python -u scripts/stage2_qwen_steer_sparse_probe_bundle.py \
    --jsonl "$JSONL" \
    --model Qwen/Qwen3.5-27B \
    --model-key qwen35_27b \
    --task "$TASK" \
    --layer "$LAYER" \
    --activation-site resid_post \
    --projection-activation-site resid_post \
    --activation-dir results/stage2/activations \
    --feature-dir results/stage2/sae_features \
    --sae-repo-id "$SAE_REPO_ID" \
    --sae-id "$SAE_ID" \
    --top-k "$TOP_K" \
    --top-positive "$TOP_POSITIVE" \
    --top-negative "$TOP_NEGATIVE" \
    --min-density 0.02 \
    --max-density 0.50 \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights 3,4 \
    --per-height-label 2 \
    --strengths=-0.5,0.5 \
    --controls shuffled,random,orthogonal \
    --control-seed 20260553 \
    --selection-seed 20260523 \
    --intervention-scope last_token_each_forward \
    --max-new-tokens 128 \
    --temperature 0 \
    --dtype bfloat16 \
    --device-map auto \
    --disable-thinking \
    --local-files-only \
    --out-jsonl "results/stage2/steering/${OUT_STEM}.jsonl" \
    --direction-output "results/stage2/steering/${OUT_STEM}_direction.npz" \
    --output "docs/qwen35_scope_sparse_bundle_steering_27b_${LAYER_TAG}_l0_${SAE_L0}_${TASK_LABEL}_smoke.json"

nvidia-smi
echo "Finished: $(date)"
