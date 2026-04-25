#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=j2_a40_probe
#SBATCH --output=slurm_logs/j2_a40_probe_%j.out
#SBATCH --error=slurm_logs/j2_a40_probe_%j.err

set -u

echo "Started: $(date)"
echo "Submitted from: $SLURM_SUBMIT_DIR"

module load conda/2024.09 || true
PHANTOM_ENV=/scratch/scholar/$USER/conda-envs/phantom
if [ -d "$PHANTOM_ENV" ] && command -v conda >/dev/null 2>&1; then
    conda activate "$PHANTOM_ENV"
elif [ ! -d "$PHANTOM_ENV" ]; then
    echo "Phantom conda environment not found at $PHANTOM_ENV; continuing with default python."
else
    echo "conda command not found; continuing with default python."
fi

python3 "$SLURM_SUBMIT_DIR/probe_j2_a40.py"

echo "Finished: $(date)"
