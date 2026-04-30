#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_feat_steer_sub
#SBATCH --output=slurm_logs/stage2_feat_steer_sub_%j.out
#SBATCH --error=slurm_logs/stage2_feat_steer_sub_%j.err

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

python -u scripts/stage2_steer_transcoder_features.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_subtype.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_subtype \
    --layer 45 \
    --hook-name blocks.45.hook_mlp_out \
    --activation-site mlp_in_weighted \
    --feature-dir results/stage2/sae_features \
    --sae-id layer_45_width_262k_l0_big_affine \
    --top-k 512 \
    --features 35036,187589,72374 \
    --scale-stat mean_nonzero \
    --control-mode random_feature \
    --hf-repo-id google/gemma-scope-2-27b-it \
    --hf-subfolder transcoder_all \
    --hf-revision main \
    --local-files-only \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights 3,4 \
    --per-height-label 2 \
    --strengths=-0.25,0.25 \
    --intervention-scope last_token_each_forward \
    --max-new-tokens 96 \
    --temperature 0 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl results/stage2/steering/big_l0_feature_steering_27b_l45_subtype_pilot.jsonl \
    --output docs/big_l0_feature_steering_27b_l45_subtype_pilot.json

nvidia-smi
echo "Finished: $(date)"
