#!/usr/bin/env python3
"""Probe-on-erased-activations check for the subspace erasure result.

Quantifies the direction-vs-information caveat: after mean-ablating the raw
correctness probe direction from stored activations, can a freshly trained
probe still decode correctness? Iterates INLP-style to estimate how many
directions carry linearly decodable signal at each layer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_probes import load_probe_dataset, read_split_assignments, split_indices_from_assignments  # noqa: E402
from src.stage2_steering import parse_int_list  # noqa: E402


def fit_probe(
    x: np.ndarray,
    labels: list[int],
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    c_values: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels)
    best = None
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=seed),
        )
        model.fit(x[train_indices], y[train_indices])
        val_auc = roc_auc_score(y[val_indices], model.predict_proba(x[val_indices])[:, 1])
        if best is None or val_auc > best["val_auc"]:
            best = {"model": model, "c": c_value, "val_auc": float(val_auc)}
    assert best is not None
    model = best["model"]
    test_auc = roc_auc_score(y[test_indices], model.predict_proba(x[test_indices])[:, 1])
    scaler = model[0]
    logreg = model[-1]
    raw_coef = logreg.coef_[0] / np.asarray(scaler.scale_)
    unit = raw_coef / np.linalg.norm(raw_coef)
    return {
        "best_c": best["c"],
        "val_auc": best["val_auc"],
        "test_auc": float(test_auc),
        "unit_direction": unit.astype(np.float64),
    }


def mean_ablate(x: np.ndarray, unit: np.ndarray, train_indices: list[int]) -> np.ndarray:
    projections = x @ unit
    mean = float(projections[train_indices].mean())
    return x - np.outer(projections - mean, unit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="results/full/with_errortype/gemma3_27b_infer_property.jsonl")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="15,30,40,45,53")
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--c-values", default="0.01,0.1,1.0")
    parser.add_argument("--iter-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260472)
    parser.add_argument("--output", type=Path, default=Path("docs/probe_on_erased_activations_27b_property.json"))
    parser.add_argument("--directions-output", type=Path, default=None, help="Optional npz path for the per-round INLP unit directions (the readable-subspace stack).")
    args = parser.parse_args()
    started = time.time()

    layers = parse_int_list(args.layers)
    c_values = tuple(float(part) for part in args.c_values.split(","))
    assignments = read_split_assignments(args.splits)
    by_layer: dict[str, Any] = {}
    for layer in layers:
        prefix = args.activation_dir / f"{args.model_key}_{args.task}_L{layer}"
        dataset = load_probe_dataset(
            activation_path=prefix.with_suffix(".safetensors"),
            sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
            drop_parse_failed=True,
        )
        x = dataset["x"].astype(np.float64)
        labels = dataset["labels"]
        splits = split_indices_from_assignments(
            dataset["sidecar"],
            assignments=assignments,
            source_file=args.jsonl,
            split_field=f"{args.split_family}_split",
        )
        train, val, test = splits["train"], splits["val"], splits["test"]

        rounds = []
        round_directions = []
        current = x
        for round_index in range(args.rounds + 1):
            round_c = c_values if round_index <= 1 else (args.iter_c,)
            probe = fit_probe(current, labels, train, val, test, round_c, args.seed)
            rounds.append(
                {
                    "round": round_index,
                    "test_auc": probe["test_auc"],
                    "val_auc": probe["val_auc"],
                    "best_c": probe["best_c"],
                }
            )
            print(
                f"L{layer} round {round_index}: test_auc={probe['test_auc']:.4f} "
                f"(val={probe['val_auc']:.4f}, c={probe['best_c']})",
                flush=True,
            )
            round_directions.append(probe["unit_direction"].astype(np.float32))
            if round_index < args.rounds:
                current = mean_ablate(current, probe["unit_direction"], train)
        if args.directions_output is not None:
            args.directions_output.parent.mkdir(parents=True, exist_ok=True)
            existing = dict(np.load(args.directions_output)) if args.directions_output.exists() else {}
            existing[f"L{layer}_inlp_stack"] = np.stack(round_directions)
            np.savez_compressed(args.directions_output, **existing)
        by_layer[f"L{layer}"] = {
            "baseline_test_auc": rounds[0]["test_auc"],
            "after_first_erasure_test_auc": rounds[1]["test_auc"] if len(rounds) > 1 else None,
            "rounds": rounds,
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "script": "scripts/stage2_probe_on_erased_check.py",
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "free_form_correctness",
        "representation_type": "raw_direction",
        "method": "probe_on_erased_activations_inlp",
        "split_family": args.split_family,
        "layers": layers,
        "rounds": args.rounds,
        "by_layer": by_layer,
        "interpretation_note": (
            "Scopes the runtime erasure result: if a retrained probe recovers high AUC after "
            "the first mean-ablation, the runtime null means the readout axis is not "
            "load-bearing even though correctness information remains linearly present in "
            "other directions. The AUC decay across INLP rounds estimates how many directions "
            "carry the signal."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
