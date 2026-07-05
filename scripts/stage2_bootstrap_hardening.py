#!/usr/bin/env python3
"""Robustness companion for near-zero CI bounds (review-response plan W2).

For a (condition, reference) pair over row-level generation JSONLs, reports
side by side: percentile bootstrap CI, BCa CI, leave-one-row-out point-
estimate range, minimum detectable effect (95% percentile half-width), and a
TOST-style equivalence check (90% CI within a declared margin). Rows are
clustered by source_row_index across all input files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scipy.stats import norm


def load_row_deltas(paths: list[Path], condition: str, reference: str) -> np.ndarray:
    samples: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if row["condition"] in (condition, reference):
                    samples[row["source_row_index"]][row["condition"]].append(bool(row["is_correct_strong"]))
    deltas = []
    for row_index in sorted(samples):
        cell = samples[row_index]
        if condition in cell and reference in cell:
            deltas.append(float(np.mean(cell[condition])) - float(np.mean(cell[reference])))
    return np.array(deltas)


def percentile_ci(deltas: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, np.ndarray]:
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), means


def bca_ci(deltas: np.ndarray, boot_means: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    theta = deltas.mean()
    z0 = norm.ppf(np.clip((boot_means < theta).mean(), 1e-9, 1 - 1e-9))
    n = len(deltas)
    jack = np.array([np.delete(deltas, i).mean() for i in range(n)])
    diffs = jack.mean() - jack
    denom = 6.0 * (diffs**2).sum() ** 1.5
    a = (diffs**3).sum() / denom if denom > 0 else 0.0
    lo_q, hi_q = (
        norm.cdf(z0 + (z0 + norm.ppf(q)) / (1 - a * (z0 + norm.ppf(q))))
        for q in (alpha / 2, 1 - alpha / 2)
    )
    return float(np.percentile(boot_means, 100 * lo_q)), float(np.percentile(boot_means, 100 * hi_q))


def analyze(label: str, paths: list[Path], condition: str, reference: str, *, seed: int, n_boot: int, margin: float) -> dict:
    deltas = load_row_deltas(paths, condition, reference)
    rng = np.random.default_rng(seed)
    lo, hi, boot = percentile_ci(deltas, rng, n_boot)
    blo, bhi = bca_ci(deltas, boot)
    loo = np.array([np.delete(deltas, i).mean() for i in range(len(deltas))])
    lo90, hi90 = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    return {
        "label": label,
        "condition": condition,
        "reference": reference,
        "n_rows": int(len(deltas)),
        "point": float(deltas.mean()),
        "percentile_ci95": [lo, hi],
        "bca_ci95": [blo, bhi],
        "loo_point_range": [float(loo.min()), float(loo.max())],
        "mde_95_halfwidth": float((hi - lo) / 2),
        "ci90": [lo90, hi90],
        "equivalent_within_margin": bool(-margin <= lo90 and hi90 <= margin),
        "margin": margin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", action="append", required=True,
                        help="label=condition:reference:path1,path2,...")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for spec in args.spec:
        label, rest = spec.split("=", 1)
        condition, reference, paths = rest.split(":", 2)
        result = analyze(
            label, [Path(p) for p in paths.split(",")], condition, reference,
            seed=args.seed, n_boot=args.n_boot, margin=args.margin,
        )
        results.append(result)
        print(
            f"{label:28s} n={result['n_rows']:3d} dP={result['point']:+.3f} "
            f"pct[{result['percentile_ci95'][0]:+.3f},{result['percentile_ci95'][1]:+.3f}] "
            f"BCa[{result['bca_ci95'][0]:+.3f},{result['bca_ci95'][1]:+.3f}] "
            f"LOO[{result['loo_point_range'][0]:+.3f},{result['loo_point_range'][1]:+.3f}] "
            f"MDE={result['mde_95_halfwidth']:.3f} "
            f"equiv@{args.margin}: {result['equivalent_within_margin']}",
            flush=True,
        )
    if args.output:
        args.output.write_text(json.dumps({"seed": args.seed, "n_boot": args.n_boot, "results": results}, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
