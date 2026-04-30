#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage2_d8_4b_big
#SBATCH --output=slurm_logs/stage2_d8_4b_big_%j.out
#SBATCH --error=slurm_logs/stage2_d8_4b_big_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p slurm_logs docs results/stage2/sae_features results/stage2/transcoder_components_exact

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

BIG_ID=layer_22_width_262k_l0_big_affine
RELEASE=gemma-scope-2-4b-it-transcoders-all
TOP_K=512
TASKS=(infer_property infer_subtype)
C_VALUES=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10

if ! python - <<'PY'
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(
        repo_id="google/gemma-scope-2-4b-it",
        filename="transcoder_all/layer_22_width_262k_l0_big_affine/config.json",
        revision="main",
    )
except Exception as exc:
    print(f"SKIP_D8_BIG_AFFINE: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
then
    cat > docs/transcoder_probe_4b_l22_262k_big_affine_exact_top512_skipped.json <<'JSON'
{
  "schema_version": 1,
  "status": "skipped",
  "reason": "Gemma Scope 2 4B layer_22_width_262k_l0_big_affine was not found or could not be resolved."
}
JSON
    echo "D.8 big-affine artifact unavailable; wrote skipped marker."
    exit 0
fi

for task in "${TASKS[@]}"; do
    python -u scripts/stage2_extract_sae_features.py \
        --activation-prefix "results/stage2/activations/gemma3_4b_${task}_L22_mlp_in_weighted" \
        --out-dir results/stage2/sae_features \
        --sae-release "$RELEASE" \
        --sae-id "$BIG_ID" \
        --hf-repo-id google/gemma-scope-2-4b-it \
        --hf-subfolder transcoder_all \
        --hf-revision main \
        --device cuda:0 \
        --dtype bfloat16 \
        --chunk-size 128 \
        --top-k "$TOP_K"
done

python - <<'PY'
import json
from pathlib import Path

inv_path = Path("docs/stage2_invariants.4b.semani.json")
inv = json.load(inv_path.open())
meta = json.load(open("results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.meta.json"))
cfg = meta["sae_cfg"]
entry = {
    "architecture": cfg["architecture"],
    "d_in": int(cfg["d_in"]),
    "d_out": int(cfg["d_out"]),
    "d_sae": int(cfg["d_sae"]),
    "hf_repo_id": meta["hf_repo_id"],
    "hf_snapshot_revision": meta["hf_snapshot_revision"],
    "hf_subfolder": meta["hf_subfolder"],
    "sae_config_sha256": meta["sae_config_sha256"],
    "sae_id": meta["sae_id"],
    "sae_params_sha256": meta["sae_params_sha256"],
    "sae_release": meta["sae_release"],
}
inv.setdefault("sae_releases", {}).setdefault("entries", {})[f"{meta['sae_release']}/{meta['sae_id']}"] = entry
inv["sae_releases"]["status"] = "pinned_4b_l20_l22_residual_l22_mlp_skip_transcoder_and_big_affine_artifacts"
json.dump(inv, inv_path.open("w"), indent=2, sort_keys=True)
inv_path.write_text(inv_path.read_text() + "\\n")
print(inv_path)
PY

for split in s1 s3; do
    suffix="$split"
    if [ "$split" = "s3" ]; then
        suffix=s3_target_symbol
    fi
    python -u scripts/stage2_probe_sae.py \
        --feature-dir results/stage2/sae_features \
        --activation-site mlp_in_weighted \
        --model-key gemma3_4b \
        --tasks "${TASKS[@]}" \
        --layer 22 \
        --sae-id "$BIG_ID" \
        --top-k "$TOP_K" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --c-values "$C_VALUES" \
        --bootstrap-samples 1000 \
        --output "docs/transcoder_probe_4b_l22_262k_big_affine_exact_top512_${suffix}.json"

    python -u scripts/stage2_transcoder_component_diagnostics.py \
        --input-activation-dir results/stage2/activations \
        --target-activation-dir results/stage2/activations \
        --feature-dir results/stage2/sae_features \
        --out-dir results/stage2/transcoder_components_exact \
        --model-key gemma3_4b \
        --tasks "${TASKS[@]}" \
        --layer 22 \
        --input-activation-site mlp_in_weighted \
        --target-activation-site mlp_out_hook \
        --transcoder-release "$RELEASE" \
        --transcoder-id "$BIG_ID" \
        --top-k "$TOP_K" \
        --splits results/stage2/splits.jsonl \
        --split-family "$split" \
        --device cuda:0 \
        --dtype bfloat16 \
        --output-dtype bfloat16 \
        --chunk-size 64 \
        --verify-dense-rows 4 \
        --solver liblinear \
        --c-values 1.0 \
        --bootstrap-samples 0 \
        --skip-target-probe \
        --skip-orientation untransposed \
        --output "docs/transcoder_component_probe_4b_l22_262k_big_affine_exact_top512_${suffix}.json"
done

echo "Finished: $(date)"
