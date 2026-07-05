#!/bin/bash
#SBATCH -p gorman-gpu
#SBATCH --gres=gpu:tesla_v100_sxm2_32gb:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --job-name=phantom_gorman
#SBATCH --output=phantom_gorman_%j.out
#SBATCH --error=phantom_gorman_%j.err

# Self-staging runner for the gorman-gpu lane (no persistent user storage).
# Pulls the packed conda env, HF weights, repo code, and declared inputs
# from Scholar into /dev/shm, runs $GORMAN_CMD with fp32 defaults, pushes
# declared outputs back to Scholar, and cleans up.
#
# Required env vars at submit time:
#   GORMAN_CMD      — command to run, relative to repo root
#                     (e.g. "python -u scripts/stage2_rank_k_guard_v2.py ...")
# Optional:
#   GORMAN_INPUTS   — space-separated repo-relative paths to pull (results/... etc.)
#   GORMAN_OUTPUTS  — space-separated repo-relative paths to push back
#
# Comparability note: V100 has no bf16; runs here use --dtype float32 and
# are NOT stitchable with Scholar bf16 shards. Calibration gate: baselines
# must reproduce Scholar's within CI before any gorman number is used.

set -euo pipefail
SCHOLAR=skiron@scholar.rcac.purdue.edu
SBASE=/scratch/scholar/skiron
SHM=/dev/shm/skiron_${SLURM_JOB_ID}
trap 'rm -rf "$SHM"' EXIT
mkdir -p "$SHM"

echo "Node: $(hostname)  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

echo "[stage] env tarball"
scp -q -o BatchMode=yes "$SCHOLAR:$SBASE/phantom_env.tar.gz" "$SHM/"
mkdir -p "$SHM/env" && tar -xzf "$SHM/phantom_env.tar.gz" -C "$SHM/env" && rm "$SHM/phantom_env.tar.gz"
source "$SHM/env/bin/activate" && conda-unpack || true

echo "[stage] HF weights"
mkdir -p "$SHM/hf/hub"
rsync -a -e "ssh -o BatchMode=yes" "$SCHOLAR:$SBASE/hf-cache/hub/models--google--gemma-3-27b-it" "$SHM/hf/hub/"
export HF_HOME="$SHM/hf" HF_HUB_CACHE="$SHM/hf/hub" HF_HUB_OFFLINE=1

echo "[stage] repo + beyond-deduction"
rsync -a -e "ssh -o BatchMode=yes" --exclude results --exclude slurm_logs --exclude .git \
    "$SCHOLAR:$SBASE/phantom-or-real/" "$SHM/repo/"
rsync -a -e "ssh -o BatchMode=yes" --exclude .git "$SCHOLAR:$SBASE/beyond-deduction/" "$SHM/bd/"
export BD_PATH="$SHM/bd" TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$SHM/repo"

for path in ${GORMAN_INPUTS:-}; do
    echo "[stage] input $path"
    mkdir -p "$(dirname "$path")"
    rsync -a -e "ssh -o BatchMode=yes" "$SCHOLAR:$SBASE/phantom-or-real/$path" "$(dirname "$path")/"
done

echo "[run] $GORMAN_CMD"
eval "$GORMAN_CMD"

for path in ${GORMAN_OUTPUTS:-}; do
    echo "[push] $path"
    rsync -a -e "ssh -o BatchMode=yes" "$path" "$SCHOLAR:$SBASE/phantom-or-real/$(dirname "$path")/"
done
echo "Finished: $(date)"
