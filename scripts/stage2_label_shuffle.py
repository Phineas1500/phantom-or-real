"""Label-shuffle sanity probe (Phase 0.5 / Step A.5).

Run after Phase A.3 activations exist. Trains a logistic probe on raw
residual activations at a chosen layer with randomly shuffled labels on
S1/S2/S3. Expect AUC ≈ 0.50 ± 0.03. Any clear deviation from 0.50 indicates
a leak in the splits or an activation-alignment bug.

Writes results/stage2/baselines/label_shuffle_{model_slug}_{task}_L{layer}.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_sidecar(sidecar_path: Path) -> list[dict]:
    rows = []
    with sidecar_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_splits(
    splits_path: Path,
    *,
    model: str,
    task: str,
    source_file: str,
) -> dict[str, dict[int, str]]:
    """Return {split_name: {row_index: partition}} for one model/task/source_file."""
    out: dict[str, dict[int, str]] = {"s1": {}, "s2": {}, "s3": {}}
    with splits_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("model") != model or row.get("task") != task:
                continue
            if row.get("source_file") != source_file:
                continue
            row_index = int(row["row_index"])
            out["s1"][row_index] = row["s1_split"]
            out["s2"][row_index] = row["s2_split"]
            out["s3"][row_index] = row["s3_split"]
    return out


def run_label_shuffle(
    X: np.ndarray,
    y_true: np.ndarray,
    split_assignment: dict[int, str],
    row_indices: list[int],
    *,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    row_to_idx = {row_idx: i for i, row_idx in enumerate(row_indices)}
    train_idx = np.array(
        [row_to_idx[row_idx] for row_idx in row_indices if split_assignment.get(row_idx) == "train"],
        dtype=np.intp,
    )
    test_idx = np.array(
        [row_to_idx[row_idx] for row_idx in row_indices if split_assignment.get(row_idx) == "test"],
        dtype=np.intp,
    )

    if len(train_idx) == 0 or len(test_idx) == 0:
        return {
            "auc": None,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "shuffle_seed": seed,
            "note": "skipped_no_train_or_test_partition",
        }

    y_shuffled = y_true.copy()
    rng.shuffle(y_shuffled)

    X_train, y_train = X[train_idx], y_shuffled[train_idx]
    X_test,  y_test  = X[test_idx],  y_shuffled[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auc": None, "note": "only one class present after shuffle — retry with different seed"}

    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=int(seed))
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_prob))

    return {
        "auc": auc,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "shuffle_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to .safetensors activation file, e.g. results/stage2/activations/gemma3_4b_infer_property_L17.safetensors",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Path to .example_ids.jsonl sidecar (inferred from --activations if omitted)",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "results" / "stage2" / "splits.jsonl",
    )
    parser.add_argument("--task", required=True, help="e.g. infer_property")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model-slug", default="gemma3_4b")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "stage2" / "baselines",
    )
    parser.add_argument(
    "--source-file",
    required=True,
    help="Exact source_file string used in splits.jsonl (e.g. results/full/with_errortype/gemma3_4b_infer_property.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gemma3-4b",
        help="Model name as stored in splits.jsonl (e.g. gemma3-4b)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load activations ─────────────────────────────────────────────────────
    from safetensors.torch import load_file
    print(f"Loading activations from {args.activations} ...")
    tensors = load_file(str(args.activations))
    X_torch = tensors["activations"]
    X = X_torch.to(dtype=float).numpy().astype(np.float32)
    print(f"  Activation shape: {X.shape}")

    # ── Load sidecar ──────────────────────────────────────────────────────────
    sidecar_path = args.sidecar or args.activations.with_suffix(".example_ids.jsonl")
    print(f"Loading sidecar from {sidecar_path} ...")
    sidecar_rows = load_sidecar(sidecar_path)
    row_indices = [int(r["row_index"]) for r in sidecar_rows]
    y_true = np.array([int(bool(r.get("is_correct_strong", False))) for r in sidecar_rows], dtype=np.int32)
    print(f"  {len(row_indices)} rows; {y_true.sum()} positive")

    if X.shape[0] != len(row_indices):
        sys.exit(f"Row count mismatch: activations={X.shape[0]}, sidecar={len(row_indices)}")

    # ── Load splits ───────────────────────────────────────────────────────────
    print(
        f"Loading splits from {args.splits} "
        f"(model={args.model}, task={args.task}, source_file={args.source_file}) ..."
    )
    splits = load_splits(
        args.splits,
        model=args.model,
        task=args.task,
        source_file=args.source_file,
    )
    for split_name, assignments in splits.items():
        print(f"  {split_name}: {len(assignments)} assignment rows")

    # ── Run label shuffle for S1/S2/S3 ───────────────────────────────────────
    results = {}
    for split_name, split_assignment in splits.items():
        print(f"  Running label-shuffle probe on {split_name} ...", end=" ", flush=True)
        res = run_label_shuffle(X, y_true, split_assignment, row_indices, seed=args.seed)
        results[split_name] = res
        auc_str = f"{res['auc']:.4f}" if res["auc"] is not None else "N/A"
        print(f"AUC={auc_str} (expected ≈0.50±0.03)")

    # ── Write output ──────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model_slug": args.model_slug,
        "model": args.model,
        "task": args.task,
        "source_file": args.source_file,
        "layer": args.layer,
        "activations_file": str(args.activations),
        "sidecar_file": str(sidecar_path),
        "splits_file": str(args.splits),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_auc_range": [0.47, 0.53],
        "results": results,
    }

    out_path = args.out_dir / f"label_shuffle_{args.model_slug}_{args.task}_L{args.layer}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {out_path}")

    # ── Sanity alert ──────────────────────────────────────────────────────────
    for split_name, res in results.items():
        auc = res.get("auc")
        if auc is not None and (auc < 0.47 or auc > 0.53):
            print(
                f"WARNING: {split_name} AUC={auc:.4f} is outside 0.47–0.53. "
                "Investigate for split leakage or activation-alignment bugs."
            )


if __name__ == "__main__":
    main()