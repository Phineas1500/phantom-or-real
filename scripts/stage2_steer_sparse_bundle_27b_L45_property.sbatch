#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_bundle_prop
#SBATCH --output=slurm_logs/stage2_bundle_prop_%j.out
#SBATCH --error=slurm_logs/stage2_bundle_prop_%j.err

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

python -u scripts/stage2_steer_sparse_probe_bundle.py \
    --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
    --model google/gemma-3-27b-it \
    --model-key gemma3_27b \
    --task infer_property \
    --layer 45 \
    --hook-name blocks.45.hook_mlp_out \
    --activation-site mlp_in_weighted \
    --projection-activation-site mlp_out_hook \
    --activation-dir results/stage2/activations \
    --feature-dir results/stage2/sae_features \
    --sae-id layer_45_width_262k_l0_big_affine \
    --top-k 512 \
    --top-positive 25 \
    --top-negative 25 \
    --min-density 0.02 \
    --max-density 0.50 \
    --weight-key standardized_coef \
    --c-values 0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10 \
    --max-iter 2000 \
    --solver liblinear \
    --controls shuffled,random,orthogonal \
    --hf-repo-id google/gemma-scope-2-27b-it \
    --hf-subfolder transcoder_all \
    --hf-revision main \
    --local-files-only \
    --splits results/stage2/splits.jsonl \
    --split-family s1 \
    --heights 3,4 \
    --per-height-label 2 \
    --strengths=-0.25,0.25,-0.5,0.5 \
    --intervention-scope last_token_each_forward \
    --max-new-tokens 96 \
    --temperature 0 \
    --n-devices 2 \
    --n-ctx 4096 \
    --dtype bfloat16 \
    --load-mode no-processing \
    --out-jsonl results/stage2/steering/sparse_probe_bundle_27b_l45_property.jsonl \
    --direction-output results/stage2/steering/sparse_probe_bundle_27b_l45_property_directions.npz \
    --output docs/sparse_probe_bundle_steering_27b_l45_property.json

nvidia-smi
echo "Finished: $(date)"
