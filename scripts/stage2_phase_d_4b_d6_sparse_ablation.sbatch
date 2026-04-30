#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage2_d6_4b_ablate
#SBATCH --output=slurm_logs/stage2_d6_4b_ablate_%j.out
#SBATCH --error=slurm_logs/stage2_d6_4b_ablate_%j.err
#SBATCH --mail-user=semani@purdue.edu
#SBATCH --mail-type=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p slurm_logs docs docs/ablation_4b_l20_l22_all_sparse

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

export HF_HOME=/scratch/scholar/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

TASKS=(infer_property infer_subtype)
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10

BLOCK_NAMES=(l20_resid16k l20_resid262k l22_resid16k l22_resid262k l22_tc16k l22_tc262k l22_mlpout16k)
BLOCK_PATTERNS=(
    "results/stage2/sae_features/gemma3_4b_{task}_L20_layer_20_width_16k_l0_small_top128"
    "results/stage2/sae_features/gemma3_4b_{task}_L20_layer_20_width_262k_l0_small_top128"
    "results/stage2/sae_features/gemma3_4b_{task}_L22_layer_22_width_16k_l0_small_top128"
    "results/stage2/sae_features/gemma3_4b_{task}_L22_layer_22_width_262k_l0_small_top128"
    "results/stage2/sae_features/gemma3_4b_{task}_L22_mlp_in_weighted_layer_22_width_16k_l0_small_affine_top128"
    "results/stage2/sae_features/gemma3_4b_{task}_L22_mlp_in_weighted_layer_22_width_262k_l0_small_affine_top128"
    "results/stage2/sae_features/gemma3_4b_{task}_L22_mlp_out_hook_layer_22_width_16k_l0_small_top128"
)

run_ablation() {
    local split="$1"
    local label="$2"
    local drop_index="$3"
    local args=()
    for idx in "${!BLOCK_PATTERNS[@]}"; do
        if [ "$drop_index" != "$idx" ]; then
            args+=(--feature-pattern "${BLOCK_PATTERNS[$idx]}")
        fi
    done
    python -u scripts/stage2_probe_sparse_concat.py \
        --combo-name "l20_l22_all_sparse_${label}" \
        "${args[@]}" \
        --tasks "${TASKS[@]}" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --c-values "$C_VALUES" \
        --bootstrap-samples 0 \
        --output "docs/ablation_4b_l20_l22_all_sparse/${label}_${split}.json"
}

for split in s1 s3; do
    run_ablation "$split" full -1
    for idx in "${!BLOCK_NAMES[@]}"; do
        run_ablation "$split" "drop_${BLOCK_NAMES[$idx]}" "$idx"
    done
done

python - <<'PY'
import json
from pathlib import Path

base = Path("docs/ablation_4b_l20_l22_all_sparse")
summary = {
    "schema_version": 1,
    "combo_name": "l20_l22_all_sparse",
    "source_dir": str(base),
    "splits": {},
}
for split in ("s1", "s3"):
    summary["splits"][split] = {}
    for path in sorted(base.glob(f"*_{split}.json")):
        report = json.load(path.open())
        label = path.name[: -len(f"_{split}.json")]
        summary["splits"][split][label] = {
            "path": str(path),
            "feature_count": len(report["feature_patterns"]),
            "tasks": {
                task: {
                    "test_auc": result.get("test_auc"),
                    "val_auc": result.get("val_auc"),
                    "best_c": result.get("best_c"),
                    "status": result.get("status"),
                }
                for task, result in report["results"].items()
            },
        }
json.dump(summary, open("docs/sparse_concat_ablation_4b_l20_l22_all_sparse_summary.json", "w"), indent=2, sort_keys=True)
print("docs/sparse_concat_ablation_4b_l20_l22_all_sparse_summary.json")
PY

echo "Finished: $(date)"
