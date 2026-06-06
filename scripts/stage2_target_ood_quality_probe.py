#!/usr/bin/env python3
"""Run continuous quality-score probes for target/OOD analysis."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    load_activation_matrix,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    write_json,
)


DEFAULT_MODEL_LAYERS = {
    "gemma3_27b": 45,
    "qwen35_27b": 53,
}
DEFAULT_TASKS = ("infer_property", "infer_subtype")
DEFAULT_SPLIT_FAMILIES = ("s1", "s3", "height_h12_to_h34")
DEFAULT_ALPHAS = (10.0, 100.0, 1000.0, 10000.0)


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_model_layer(value: str) -> tuple[str, int]:
    if ":" not in value:
        model_key = value
        if model_key not in DEFAULT_MODEL_LAYERS:
            raise ValueError(f"layer required for unknown model key: {value}")
        return model_key, DEFAULT_MODEL_LAYERS[model_key]
    model_key, layer_text = value.split(":", 1)
    return model_key.strip(), int(layer_text)


def read_full_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open() as f:
        for row_index, line in enumerate(f):
            if line.strip():
                rows[row_index] = json.loads(line)
    return rows


def load_joined_dataset(
    *,
    activation_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    activation_site: str,
    drop_parse_failed: bool,
) -> dict[str, Any]:
    prefix = activation_dir / activation_stem(
        model_key=model_key,
        task=task,
        layer=layer,
        activation_site=activation_site,
    )
    meta = read_json(prefix.with_suffix(".meta.json"))
    source_file = str(meta["jsonl_path"])
    source_rows = read_full_rows(Path(source_file))
    x_all = load_activation_matrix(prefix.with_suffix(".safetensors"))
    sidecar_all = read_jsonl(prefix.with_suffix(".example_ids.jsonl"))
    if x_all.shape[0] != len(sidecar_all):
        raise ValueError(f"{prefix}.safetensors rows {x_all.shape[0]} != sidecar rows {len(sidecar_all)}")

    keep_indices = [
        idx
        for idx, row in enumerate(sidecar_all)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    sidecar = []
    joined_rows = []
    for idx in keep_indices:
        sidecar_row = dict(sidecar_all[idx])
        source_row = source_rows.get(int(sidecar_row["row_index"]))
        if source_row is None:
            raise KeyError(f"missing source row {sidecar_row['row_index']} in {source_file}")
        if bool(sidecar_row.get("is_correct_strong")) != bool(source_row.get("is_correct_strong")):
            raise ValueError(
                f"strong label mismatch for {source_file}:{sidecar_row['row_index']}"
            )
        sidecar_row["is_correct_weak"] = bool(source_row.get("is_correct_weak"))
        sidecar_row["quality_score"] = float(source_row["quality_score"])
        sidecar_row["source_parse_failed"] = bool(source_row.get("parse_failed"))
        sidecar.append(sidecar_row)
        joined_rows.append(source_row)

    x = x_all[keep_indices]
    return {
        "x": x,
        "sidecar": sidecar,
        "joined_rows": joined_rows,
        "source_file": source_file,
        "activation_prefix": str(prefix),
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
        "d_model": int(x.shape[1]),
        "source_total_rows": len(source_rows),
        "source_parse_failed_rows": sum(1 for row in source_rows.values() if row.get("parse_failed")),
    }


def make_height_extrapolation_splits(
    sidecar_rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> dict[str, list[int]]:
    low_indices = []
    test_indices = []
    for idx, row in enumerate(sidecar_rows):
        height = int(row["height"])
        if height in (1, 2):
            low_indices.append(idx)
        elif height in (3, 4):
            test_indices.append(idx)
        else:
            raise ValueError(f"unexpected height {height!r}")

    rng = random.Random(seed)
    shuffled = list(low_indices)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    return {
        "train": sorted(shuffled[:-n_val]),
        "val": sorted(shuffled[-n_val:]),
        "test": sorted(test_indices),
    }


def split_for_family(
    *,
    split_family: str,
    sidecar_rows: list[dict[str, Any]],
    assignments: dict[tuple[str, int], dict[str, Any]],
    source_file: str,
    seed: int,
    height_val_fraction: float,
) -> dict[str, list[int]]:
    if split_family in {"s1", "s2", "s3"}:
        return split_indices_from_assignments(
            sidecar_rows,
            assignments=assignments,
            source_file=source_file,
            split_field=f"{split_family}_split",
        )
    if split_family == "height_h12_to_h34":
        return make_height_extrapolation_splits(
            sidecar_rows,
            seed=seed,
            val_fraction=height_val_fraction,
        )
    raise ValueError(f"unknown split family: {split_family}")


def _rankdata(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return np.asarray(rankdata(values), dtype=np.float64)
    except Exception:  # noqa: BLE001
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            rank = (start + end - 1) / 2.0 + 1.0
            ranks[order[start:end]] = rank
            start = end
        return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "n": int(len(y_true)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true)),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "pearson": _corr(y_true, y_pred),
        "spearman": _corr(_rankdata(y_true), _rankdata(y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any] | None:
    if samples <= 0 or len(y_true) < 2:
        return None
    rng = np.random.default_rng(seed)
    metric_samples: dict[str, list[float]] = {"pearson": [], "spearman": [], "r2": []}
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        metrics = regression_metrics(y_true[indices], y_pred[indices])
        for metric_name in metric_samples:
            value = metrics.get(metric_name)
            if value is not None and np.isfinite(value):
                metric_samples[metric_name].append(float(value))
    out: dict[str, Any] = {
        "samples_requested": samples,
        "seed": seed,
    }
    for metric_name, values in metric_samples.items():
        if values:
            low, high = np.quantile(values, [0.025, 0.975])
            out[metric_name] = {
                "samples_used": len(values),
                "low": float(low),
                "high": float(high),
            }
    return out


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    *,
    alphas: tuple[float, ...],
    max_iter: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    best = None
    for alpha in alphas:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, max_iter=max_iter),
        )
        model.fit(x[train], y[train])
        val_pred = np.asarray(model.predict(x[val]), dtype=np.float64)
        val_metrics = regression_metrics(y[val], val_pred)
        rank = val_metrics["spearman"] if val_metrics["spearman"] is not None else -np.inf
        if best is None or rank > best["rank"]:
            best = {
                "alpha": alpha,
                "rank": rank,
                "model": model,
                "val_metrics": val_metrics,
            }
    assert best is not None
    test_pred = np.asarray(best["model"].predict(x[test]), dtype=np.float64)
    test_metrics = regression_metrics(y[test], test_pred)
    return {
        "status": "ok",
        "best_alpha": best["alpha"],
        "alphas": list(alphas),
        "selection_metric": "val_spearman",
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "test_metric_ci": bootstrap_metric_ci(
            y[test],
            test_pred,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }


def binary_mean_baseline(
    y: np.ndarray,
    binary_labels: np.ndarray,
    splits: dict[str, list[int]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    train = splits["train"]
    train_mean = float(np.mean(y[train]))
    means = {}
    for label in (0, 1):
        label_indices = [idx for idx in train if int(binary_labels[idx]) == label]
        means[str(label)] = float(np.mean(y[label_indices])) if label_indices else train_mean

    out = {
        "train_means": means,
        "fallback_mean": train_mean,
        "splits": {},
    }
    for split, indices in splits.items():
        pred = np.asarray([means.get(str(int(binary_labels[idx])), train_mean) for idx in indices], dtype=np.float64)
        split_y = y[indices]
        out["splits"][split] = {
            "metrics": regression_metrics(split_y, pred),
            "metric_ci": (
                bootstrap_metric_ci(
                    split_y,
                    pred,
                    seed=bootstrap_seed,
                    samples=bootstrap_samples,
                )
                if split == "test"
                else None
            ),
        }
    return out


def residualize_by_binary(
    y: np.ndarray,
    binary_labels: np.ndarray,
    splits: dict[str, list[int]],
) -> tuple[np.ndarray, dict[str, float]]:
    train = splits["train"]
    fallback_mean = float(np.mean(y[train]))
    means = {}
    for label in (0, 1):
        label_indices = [idx for idx in train if int(binary_labels[idx]) == label]
        means[str(label)] = float(np.mean(y[label_indices])) if label_indices else fallback_mean
    residual = np.asarray(
        [float(y[idx]) - means.get(str(int(binary_labels[idx])), fallback_mean) for idx in range(len(y))],
        dtype=np.float64,
    )
    return residual, means


def split_counts(y: np.ndarray, labels: dict[str, np.ndarray], splits: dict[str, list[int]]) -> dict[str, Any]:
    out = {}
    for split, indices in splits.items():
        split_y = y[indices]
        out[split] = {
            "n": len(indices),
            "quality_mean": float(np.mean(split_y)),
            "quality_std": float(np.std(split_y)),
            "quality_min": float(np.min(split_y)),
            "quality_max": float(np.max(split_y)),
            "strong_positive_n": int(np.sum(labels["is_correct_strong"][indices])),
            "weak_positive_n": int(np.sum(labels["is_correct_weak"][indices])),
        }
    return out


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    assignments = read_split_assignments(args.splits)
    model_layers = [parse_model_layer(value) for value in args.model_layers]
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "raw_activation_quality_regression",
        "target_variable": "quality_score",
        "representation_type": "raw activation",
        "activation_dir": str(args.activation_dir),
        "activation_site": args.activation_site,
        "splits_path": str(args.splits),
        "model_layers": [{"model_key": model_key, "layer": layer} for model_key, layer in model_layers],
        "tasks": list(args.tasks),
        "split_families": list(args.split_families),
        "drop_parse_failed": args.drop_parse_failed,
        "seed": args.seed,
        "alphas": list(args.alphas),
        "max_iter": args.max_iter,
        "bootstrap_samples": args.bootstrap_samples,
        "height_val_fraction": args.height_val_fraction,
        "causal_abstraction_claim": {
            "target_variable": "quality_score",
            "tested_representation": "raw residual activation",
            "claim_type": "predictive",
            "note": "Continuous target analysis; no causal claim is made.",
        },
        "results": {},
    }

    for model_key, layer in model_layers:
        report["results"].setdefault(model_key, {})
        for task in args.tasks:
            dataset = load_joined_dataset(
                activation_dir=args.activation_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                activation_site=args.activation_site,
                drop_parse_failed=args.drop_parse_failed,
            )
            y = np.asarray([float(row["quality_score"]) for row in dataset["joined_rows"]], dtype=np.float64)
            labels = {
                "is_correct_strong": np.asarray([int(bool(row["is_correct_strong"])) for row in dataset["joined_rows"]], dtype=np.int64),
                "is_correct_weak": np.asarray([int(bool(row["is_correct_weak"])) for row in dataset["joined_rows"]], dtype=np.int64),
            }
            task_result = {
                "model": model_key,
                "task": task,
                "site_or_layer": f"L{layer}/{args.activation_site}",
                "source_file": dataset["source_file"],
                "activation_prefix": dataset["activation_prefix"],
                "n": dataset["kept_rows"],
                "parse_fail_rate": (
                    dataset["source_parse_failed_rows"] / dataset["source_total_rows"]
                    if dataset["source_total_rows"]
                    else None
                ),
                "quality_unique_count": int(len(set(float(value) for value in y))),
                "splits": {},
            }
            for split_family in args.split_families:
                split_seed = args.seed + layer + sum(ord(ch) for ch in f"{model_key}:{task}:quality_score:{split_family}")
                splits = split_for_family(
                    split_family=split_family,
                    sidecar_rows=dataset["sidecar"],
                    assignments=assignments,
                    source_file=dataset["source_file"],
                    seed=split_seed,
                    height_val_fraction=args.height_val_fraction,
                )
                raw = fit_ridge(
                    dataset["x"],
                    y,
                    splits,
                    alphas=args.alphas,
                    max_iter=args.max_iter,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=split_seed,
                )
                strong_baseline = binary_mean_baseline(
                    y,
                    labels["is_correct_strong"],
                    splits,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=split_seed + 1009,
                )
                weak_baseline = binary_mean_baseline(
                    y,
                    labels["is_correct_weak"],
                    splits,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=split_seed + 2003,
                )
                residual, residual_means = residualize_by_binary(y, labels["is_correct_strong"], splits)
                residual_raw = fit_ridge(
                    dataset["x"],
                    residual,
                    splits,
                    alphas=args.alphas,
                    max_iter=args.max_iter,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=split_seed + 3001,
                )
                test_raw = raw["test_metrics"]
                test_strong = strong_baseline["splits"]["test"]["metrics"]
                task_result["splits"][split_family] = {
                    "split": split_family,
                    "target_variable": "quality_score",
                    "split_counts": split_counts(y, labels, splits),
                    "raw_regression": raw,
                    "strong_label_mean_baseline": strong_baseline,
                    "weak_label_mean_baseline": weak_baseline,
                    "raw_delta_vs_strong_baseline": {
                        "pearson": (
                            test_raw["pearson"] - test_strong["pearson"]
                            if test_raw["pearson"] is not None and test_strong["pearson"] is not None
                            else None
                        ),
                        "spearman": (
                            test_raw["spearman"] - test_strong["spearman"]
                            if test_raw["spearman"] is not None and test_strong["spearman"] is not None
                            else None
                        ),
                        "r2": (
                            test_raw["r2"] - test_strong["r2"]
                            if test_raw["r2"] is not None and test_strong["r2"] is not None
                            else None
                        ),
                    },
                    "quality_residual_after_strong": {
                        "residualization_means": residual_means,
                        "raw_regression": residual_raw,
                    },
                }
                print(
                    model_key,
                    task,
                    split_family,
                    "raw_spearman",
                    raw["test_metrics"]["spearman"],
                    "strong_spearman",
                    strong_baseline["splits"]["test"]["metrics"]["spearman"],
                    "resid_spearman",
                    residual_raw["test_metrics"]["spearman"],
                )
            report["results"][model_key][task] = task_result
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--model-layers", nargs="+", default=("gemma3_27b:45", "qwen35_27b:53"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split-families", nargs="+", default=DEFAULT_SPLIT_FAMILIES)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--drop-parse-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alphas", type=parse_float_list, default=DEFAULT_ALPHAS)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--height-val-fraction", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=Path("docs/target_ood_quality_probe_27b_main.json"))
    args = parser.parse_args()

    report = run_grid(args)
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
