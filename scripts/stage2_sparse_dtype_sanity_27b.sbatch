#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J s2-dtype-sanity
#SBATCH -o slurm_logs/stage2_sparse_dtype_sanity_27b.%j.out
#SBATCH -e slurm_logs/stage2_sparse_dtype_sanity_27b.%j.err

set -euo pipefail

cd /scratch/scholar/$USER/phantom-or-real
mkdir -p slurm_logs results/stage2/dtype_sanity_features docs

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

COMMON=(--out-dir results/stage2/dtype_sanity_features --device cuda:0 --dtype float32 --chunk-size 256 --top-k 128 --limit 512 --local-files-only)

python scripts/stage2_extract_sae_features.py \
  --activation-prefix results/stage2/activations/gemma3_27b_infer_property_L45 \
  --sae-release gemma-scope-2-27b-it-res-all \
  --hf-subfolder resid_post_all \
  --sae-id layer_45_width_16k_l0_small \
  "${COMMON[@]}"

python scripts/stage2_extract_sae_features.py \
  --activation-prefix results/stage2/activations/gemma3_27b_infer_subtype_L45 \
  --sae-release gemma-scope-2-27b-it-res-all \
  --hf-subfolder resid_post_all \
  --sae-id layer_45_width_16k_l0_small \
  "${COMMON[@]}"

python scripts/stage2_extract_sae_features.py \
  --activation-prefix results/stage2/activations/gemma3_27b_infer_property_L45_mlp_out \
  --sae-release gemma-scope-2-27b-it-mlp-all \
  --hf-subfolder mlp_out_all \
  --sae-id layer_45_width_16k_l0_small \
  "${COMMON[@]}"

python scripts/stage2_extract_sae_features.py \
  --activation-prefix results/stage2/activations/gemma3_27b_infer_subtype_L45_mlp_out \
  --sae-release gemma-scope-2-27b-it-mlp-all \
  --hf-subfolder mlp_out_all \
  --sae-id layer_45_width_16k_l0_small \
  "${COMMON[@]}"

python scripts/stage2_extract_sae_features.py \
  --activation-prefix results/stage2/activations/gemma3_27b_infer_property_L45_mlp_in \
  --sae-release gemma-scope-2-27b-it-transcoders-all \
  --hf-subfolder transcoder_all \
  --sae-id layer_45_width_16k_l0_small_affine \
  "${COMMON[@]}"

python scripts/stage2_extract_sae_features.py \
  --activation-prefix results/stage2/activations/gemma3_27b_infer_subtype_L45_mlp_in \
  --sae-release gemma-scope-2-27b-it-transcoders-all \
  --hf-subfolder transcoder_all \
  --sae-id layer_45_width_16k_l0_small_affine \
  "${COMMON[@]}"

python scripts/stage2_extract_crosscoder_features.py \
  --activation-dir results/stage2/activations \
  --model-key gemma3_27b \
  --task infer_property \
  --layers 16,31,40,53 \
  --out-dir results/stage2/dtype_sanity_features \
  --crosscoder-id layer_16_31_40_53_width_65k_l0_medium \
  --device cuda:0 \
  --dtype float32 \
  --chunk-size 256 \
  --top-k 128 \
  --limit 512 \
  --local-files-only

python scripts/stage2_extract_crosscoder_features.py \
  --activation-dir results/stage2/activations \
  --model-key gemma3_27b \
  --task infer_subtype \
  --layers 16,31,40,53 \
  --out-dir results/stage2/dtype_sanity_features \
  --crosscoder-id layer_16_31_40_53_width_65k_l0_medium \
  --device cuda:0 \
  --dtype float32 \
  --chunk-size 256 \
  --top-k 128 \
  --limit 512 \
  --local-files-only

python scripts/stage2_sparse_dtype_sanity_report.py \
  --stage2-root results/stage2 \
  --dtype-root results/stage2/dtype_sanity_features \
  --output docs/sparse_dtype_sanity_27b.json
