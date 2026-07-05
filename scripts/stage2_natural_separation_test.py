#!/usr/bin/env python3
"""Item F(i): does the rank-8 subspace separate NATURAL success from failure?

Pre-registered in docs/causal_handle_directions.md (F(i)) before capture.
Features: per-row mean over gold-concept positions of the unhinted L30
state, projected onto a frozen rank-8 basis (8 features). Logistic
regression, stratified 5-fold CV AUC, vs 200 seeded random orthonormal
rank-8 subspace nulls. Bases: (a) dev/composite; (b) guard-v2 full 26-row.
Full-dimensional logistic AUC reported as ceiling.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.stage2_rank_k_guard import fit_pca_basis  # noqa: E402


def load_deltas(npz_path: Path, layer: int) -> dict[int, np.ndarray]:
    data = np.load(npz_path)
    out = {}
    for key in data.files:
        match = re.match(rf"L{layer}_row(\d+)_concept_delta$", key)
        if match:
            out[int(match.group(1))] = data[key].astype(np.float64)
    return out


def cv_auc(x: np.ndarray, y: np.ndarray, seed: int, c: float = 0.01) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(y))
    for train, test in skf.split(x, y):
        mu, sd = x[train].mean(axis=0), x[train].std(axis=0) + 1e-8
        clf = LogisticRegression(C=c, max_iter=2000, random_state=seed)
        clf.fit((x[train] - mu) / sd, y[train])
        scores[test] = clf.decision_function((x[test] - mu) / sd)
    return float(roc_auc_score(y, scores))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-npz", type=Path, default=Path("results/stage2/erasure/natural_state_capture_27b_property_L30.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage2/erasure/natural_state_capture_27b_property_L30.manifest.jsonl"))
    parser.add_argument("--dev-states-npz", type=Path, default=Path("results/stage2/erasure/focus_state_composite_27b_property_states.npz"))
    parser.add_argument("--guard-states-npz", nargs=2, type=Path, default=[
        Path("results/stage2/erasure/rank_k_guard_v2_27b_property_shard0of2_states.npz"),
        Path("results/stage2/erasure/rank_k_guard_v2_27b_property_shard1of2_states.npz"),
    ])
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--output", type=Path, default=Path("docs/natural_separation_test_27b_property.json"))
    args = parser.parse_args()

    labels = {}
    heights = {}
    with args.manifest.open() as f:
        for line in f:
            row = json.loads(line)
            labels[row["source_row_index"]] = bool(row["is_correct_strong"])
            heights[row["source_row_index"]] = row["height"]
    states = np.load(args.capture_npz)
    rows = sorted(labels)
    x_full = np.stack([
        states[f"L{args.layer}_row{r}_unhinted_concept_states"].astype(np.float64).mean(axis=0)
        for r in rows
    ])
    y = np.array([labels[r] for r in rows], dtype=int)
    h = np.array([heights[r] for r in rows])
    print(f"rows={len(rows)} P(correct)={y.mean():.3f} dims={x_full.shape[1]}")

    dev_basis = fit_pca_basis(load_deltas(args.dev_states_npz, args.layer), args.rank)
    guard_deltas: dict[int, np.ndarray] = {}
    for path in args.guard_states_npz:
        guard_deltas.update(load_deltas(path, args.layer))
    guard_basis = fit_pca_basis(guard_deltas, args.rank)

    rng = np.random.default_rng(args.seed)
    results: dict = {"schema_version": 1, "rows": len(rows), "rank": args.rank, "n_null": args.n_null, "seed": args.seed}
    results["full_dim_auc"] = cv_auc(x_full, y, args.seed)
    print(f"full-dim ceiling AUC = {results['full_dim_auc']:.3f}")

    null_aucs = []
    for i in range(args.n_null):
        q, _ = np.linalg.qr(rng.standard_normal((x_full.shape[1], args.rank)))
        null_aucs.append(cv_auc(x_full @ q, y, args.seed))
    null_aucs = np.array(null_aucs)
    results["null_median"] = float(np.median(null_aucs))
    results["null_p95"] = float(np.percentile(null_aucs, 95))
    print(f"random-subspace null: median {results['null_median']:.3f}, p95 {results['null_p95']:.3f}")

    for name, basis in (("dev_basis", dev_basis), ("guard_basis", guard_basis)):
        auc = cv_auc(x_full @ basis["components"].T, y, args.seed)
        supported = auc > results["null_p95"]
        results[name] = {
            "auc": auc,
            "exceeds_null_p95": bool(supported),
            "margin_over_null_median": float(auc - results["null_median"]),
        }
        print(f"{name}: AUC = {auc:.3f}  > null p95? {supported}")
        for height in sorted(set(h)):
            mask = h == height
            if len(set(y[mask])) == 2:
                results[name][f"h{height}_auc"] = cv_auc((x_full @ basis["components"].T)[mask], y[mask], args.seed)

    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
