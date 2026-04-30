#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=stage2_d8_4b_inv
#SBATCH --output=slurm_logs/stage2_d8_4b_inv_%j.out
#SBATCH --error=slurm_logs/stage2_d8_4b_inv_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p slurm_logs docs

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import list_repo_files

repo_id = "google/gemma-scope-2-4b-it"
needle = "transcoder_all/layer_22_width_262k_l0_big_affine/"
out = Path("docs/gemma_scope_4b_l22_big_affine_inventory.md")

files = list_repo_files(repo_id, repo_type="model")
matches = sorted(path for path in files if needle in path)
transcoder_l22 = sorted(path for path in files if path.startswith("transcoder_all/layer_22_"))

lines = [
    "# Gemma Scope 2 4B L22 Big-Affine Inventory",
    "",
    f"Checked: {datetime.now(timezone.utc).isoformat()}",
    f"Repository: `{repo_id}`",
    f"Target artifact: `{needle.rstrip('/')}`",
    "",
    f"Exists: `{bool(matches)}`",
    "",
]
if matches:
    lines.extend(["## Matching Files", ""])
    lines.extend(f"- `{path}`" for path in matches)
else:
    lines.extend([
        "No `layer_22_width_262k_l0_big_affine` files were found. D.8 big-affine rerun should be skipped unless this repo changes.",
        "",
        "## Available L22 Transcoder Files",
        "",
    ])
    lines.extend(f"- `{path}`" for path in transcoder_l22[:100])

out.write_text("\\n".join(lines) + "\\n")
print(out)
print(f"exists={bool(matches)} matches={len(matches)}")
PY

echo "Finished: $(date)"
