#!/usr/bin/env python3
"""Analyze top-weight stability for sparse SAE logistic probes."""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_feature_stability import (  # noqa: E402
    best_correlation_matches,
    coefficient_weights,
    dense_feature_columns,
    feature_activation_stats,
    feature_overlap,
    pairwise_column_correlations,
    summarize_correlation_matches,
)
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    write_json,
)
from src.stage2_paths import DEFAULT_ACTIVATION_SITE, activation_stem  # noqa: E402
from src.stage2_sae import topk_tensors_to_csr  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _predict_scores(model: Any, x: Any, indices: list[int]) -> list[float]:
    if not indices:
        return []
    return [float(score) for score in model.predict_proba(x[indices])[:, 1]]


def load_sae_dataset(*, feature_prefix: Path, drop_parse_failed: bool) -> dict[str, Any]:
    meta = read_json(feature_prefix.with_suffix(".meta.json"))
    tensors = load_file(feature_prefix.with_suffix(".safetensors"))
    x_all = topk_tensors_to_csr(
        tensors["top_indices"],
        tensors["top_values"],
        d_sae=int(meta["sae_cfg"]["d_sae"]),
    )
    x_all.eliminate_zeros()
    sidecar_all = read_jsonl(feature_prefix.with_suffix(".example_ids.jsonl"))
    if x_all.shape[0] != len(sidecar_all):
        raise ValueError(f"{feature_prefix} rows {x_all.shape[0]} != sidecar rows {len(sidecar_all)}")
    keep_indices = [
        idx
        for idx, row in enumerate(sidecar_all)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    x = x_all[keep_indices]
    sidecar = [sidecar_all[idx] for idx in keep_indices]
    return {
        "x": x,
        "labels": [int(row["is_correct_strong"]) for row in sidecar],
        "sidecar": sidecar,
        "meta": meta,
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
    }


def fit_best_logistic_model(
    x: Any,
    labels: list[int],
    *,
    splits: dict[str, list[int]],
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
) -> dict[str, Any]:
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    if len({labels[idx] for idx in train_indices}) < 2:
        raise ValueError("train split has fewer than two classes")
    if len({labels[idx] for idx in val_indices}) < 2 or len({labels[idx] for idx in test_indices}) < 2:
        raise ValueError("val/test split has fewer than two classes")

    if sparse.issparse(x):
        active_feature_ids = np.unique(x[train_indices].nonzero()[1]).astype(np.int64, copy=False)
    else:
        active_feature_ids = np.arange(x.shape[1], dtype=np.int64)
    if active_feature_ids.size == 0:
        raise ValueError("train split has no active features")
    x_fit = x[:, active_feature_ids]

    best = None
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(with_mean=not sparse.issparse(x_fit)),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=max_iter, solver=solver),
        )
        model.fit(x_fit[train_indices], [labels[idx] for idx in train_indices])
        val_scores = _predict_scores(model, x_fit, val_indices)
        val_auc = _safe_auc([labels[idx] for idx in val_indices], val_scores)
        rank_auc = -np.inf if val_auc is None else val_auc
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "best_c": float(c_value),
                "val_auc": val_auc,
                "rank_auc": rank_auc,
            }
    if best is None:
        raise ValueError("no C values provided")

    model = best["model"]
    test_scores = _predict_scores(model, x_fit, test_indices)
    best["test_auc"] = _safe_auc([labels[idx] for idx in test_indices], test_scores)
    best["active_feature_ids"] = active_feature_ids
    best["active_feature_n"] = int(active_feature_ids.size)
    return best


def remap_feature_rows(
    features: list[dict[str, Any]],
    *,
    active_feature_ids: np.ndarray,
) -> list[dict[str, Any]]:
    remapped = []
    for row in features:
        fit_feature_index = int(row["feature"])
        remapped.append(
            {
                **row,
                "fit_feature_index": fit_feature_index,
                "feature": int(active_feature_ids[fit_feature_index]),
            }
        )
    return remapped


def add_activation_stats(
    features: list[dict[str, Any]],
    *,
    x: Any,
    train_indices: list[int],
) -> list[dict[str, Any]]:
    feature_ids = [int(row["feature"]) for row in features]
    all_stats = feature_activation_stats(x, feature_ids)
    train_stats = feature_activation_stats(x, feature_ids, row_indices=train_indices)
    enriched = []
    for row in features:
        feature = int(row["feature"])
        enriched.append(
            {
                **row,
                "activation_all": all_stats[feature],
                "activation_train": train_stats[feature],
            }
        )
    return enriched


def row_alignment(left_sidecar: list[dict[str, Any]], right_sidecar: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    left_by_row = {int(row["row_index"]): idx for idx, row in enumerate(left_sidecar)}
    right_by_row = {int(row["row_index"]): idx for idx, row in enumerate(right_sidecar)}
    common_rows = sorted(set(left_by_row) & set(right_by_row))
    return [left_by_row[row] for row in common_rows], [right_by_row[row] for row in common_rows]


def load_best_c_values(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    best_c: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        report = read_json(path)
        sae_id = report["sae_id"]
        for task, result in report["results"].items():
            if result.get("status") == "ok" and result.get("best_c") is not None:
                best_c[(sae_id, task)] = {
                    "best_c": float(result["best_c"]),
                    "source": str(path),
                }
    return best_c


def run_analysis(
    *,
    feature_dir: Path,
    activation_site: str,
    model_key: str,
    tasks: list[str],
    layer: int,
    sae_ids: list[str],
    top_k: int,
    splits_path: Path,
    split_family: str,
    output_top_n: int,
    overlap_top_ns: tuple[int, ...],
    correlation_top_n: int,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    drop_parse_failed: bool,
    best_c_values: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    datasets: dict[str, dict[str, dict[str, Any]]] = {}
    models: dict[str, dict[str, dict[str, Any]]] = {}

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_dir": str(feature_dir),
        "activation_site": activation_site,
        "model_key": model_key,
        "tasks": tasks,
        "layer": layer,
        "sae_ids": sae_ids,
        "top_k": top_k,
        "splits_path": str(splits_path),
        "split_family": split_family,
        "drop_parse_failed": drop_parse_failed,
        "c_values": list(c_values),
        "best_c_values": {
            f"{sae_id}/{task}": value
            for (sae_id, task), value in (best_c_values or {}).items()
        },
        "max_iter": max_iter,
        "solver": solver,
        "coefficient_weight_space": "standardized_logistic_features",
        "models": {},
        "within_width_task_overlap": {},
        "cross_width_activation_correlation": {},
        "stable_same_width_candidates": {},
    }

    for sae_id in sae_ids:
        datasets[sae_id] = {}
        models[sae_id] = {}
        report["models"][sae_id] = {}
        for task in tasks:
            feature_stem = activation_stem(
                model_key=model_key,
                task=task,
                layer=layer,
                activation_site=activation_site,
            )
            prefix = feature_dir / f"{feature_stem}_{sae_id}_top{top_k}"
            dataset = load_sae_dataset(feature_prefix=prefix, drop_parse_failed=drop_parse_failed)
            source_file = dataset["meta"]["source_activation_meta"]["jsonl_path"]
            splits = split_indices_from_assignments(
                dataset["sidecar"],
                assignments=split_assignments,
                source_file=source_file,
                split_field=f"{split_family}_split",
            )
            best_c_entry = (best_c_values or {}).get((sae_id, task))
            fit_c_values = (float(best_c_entry["best_c"]),) if best_c_entry else c_values
            fitted = fit_best_logistic_model(
                dataset["x"],
                dataset["labels"],
                splits=splits,
                c_values=fit_c_values,
                max_iter=max_iter,
                solver=solver,
            )
            model = fitted["model"]
            scaler = model.named_steps["standardscaler"]
            logistic = model.named_steps["logisticregression"]
            coef = logistic.coef_[0]
            scale = getattr(scaler, "scale_", None)
            active_feature_ids = fitted["active_feature_ids"]
            top_abs = remap_feature_rows(
                coefficient_weights(coef, scaler_scale=scale, top_n=output_top_n, mode="abs"),
                active_feature_ids=active_feature_ids,
            )
            top_pos = remap_feature_rows(
                coefficient_weights(coef, scaler_scale=scale, top_n=output_top_n, mode="positive"),
                active_feature_ids=active_feature_ids,
            )
            top_neg = remap_feature_rows(
                coefficient_weights(coef, scaler_scale=scale, top_n=output_top_n, mode="negative"),
                active_feature_ids=active_feature_ids,
            )
            top_abs = add_activation_stats(top_abs, x=dataset["x"], train_indices=splits["train"])
            top_pos = add_activation_stats(top_pos, x=dataset["x"], train_indices=splits["train"])
            top_neg = add_activation_stats(top_neg, x=dataset["x"], train_indices=splits["train"])

            model_report = {
                "status": "ok",
                "feature_path": str(prefix.with_suffix(".safetensors")),
                "input_rows": dataset["input_rows"],
                "kept_rows": dataset["kept_rows"],
                "d_sae": int(dataset["meta"]["sae_cfg"]["d_sae"]),
                "fit_active_feature_n": fitted["active_feature_n"],
                "fit_active_feature_source": "train_nonzero",
                "fit_c_values": list(fit_c_values),
                "best_c_source": best_c_entry["source"] if best_c_entry else "grid_search",
                "best_c": fitted["best_c"],
                "val_auc": fitted["val_auc"],
                "test_auc": fitted["test_auc"],
                "top_abs_features": top_abs,
                "top_positive_features": top_pos,
                "top_negative_features": top_neg,
            }
            report["models"][sae_id][task] = model_report
            datasets[sae_id][task] = {**dataset, "splits": splits}
            models[sae_id][task] = {
                "model": model,
                "coef": coef,
                "top_abs_features": top_abs,
            }

    if len(tasks) >= 2:
        left_task, right_task = tasks[:2]
        for sae_id in sae_ids:
            report["within_width_task_overlap"][sae_id] = {
                str(top_n): feature_overlap(
                    report["models"][sae_id][left_task]["top_abs_features"],
                    report["models"][sae_id][right_task]["top_abs_features"],
                    top_n=top_n,
                )
                for top_n in overlap_top_ns
                if top_n <= output_top_n
            }
            candidate_overlap = feature_overlap(
                report["models"][sae_id][left_task]["top_abs_features"],
                report["models"][sae_id][right_task]["top_abs_features"],
                top_n=min(correlation_top_n, output_top_n),
            )
            left_by_feature = {
                int(row["feature"]): row
                for row in report["models"][sae_id][left_task]["top_abs_features"]
            }
            right_by_feature = {
                int(row["feature"]): row
                for row in report["models"][sae_id][right_task]["top_abs_features"]
            }
            candidates = []
            for feature in candidate_overlap["overlap_features"]:
                left_row = left_by_feature[feature]
                right_row = right_by_feature[feature]
                candidates.append(
                    {
                        "feature": feature,
                        "left_task": left_task,
                        "right_task": right_task,
                        "left_rank": left_row["rank"],
                        "right_rank": right_row["rank"],
                        "left_weight": left_row["weight"],
                        "right_weight": right_row["weight"],
                        "same_sign": left_row["sign"] == right_row["sign"],
                    }
                )
            candidates.sort(key=lambda row: (not row["same_sign"], max(row["left_rank"], row["right_rank"])))
            report["stable_same_width_candidates"][sae_id] = candidates

    for task in tasks:
        report["cross_width_activation_correlation"][task] = {}
        for left_sae, right_sae in itertools.combinations(sae_ids, 2):
            left_dataset = datasets[left_sae][task]
            right_dataset = datasets[right_sae][task]
            left_rows, right_rows = row_alignment(left_dataset["sidecar"], right_dataset["sidecar"])
            left_features = [
                int(row["feature"])
                for row in report["models"][left_sae][task]["top_abs_features"][:correlation_top_n]
            ]
            right_features = [
                int(row["feature"])
                for row in report["models"][right_sae][task]["top_abs_features"][:correlation_top_n]
            ]
            left_columns = dense_feature_columns(left_dataset["x"], left_features, row_indices=left_rows)
            right_columns = dense_feature_columns(right_dataset["x"], right_features, row_indices=right_rows)
            corr = pairwise_column_correlations(left_columns, right_columns)
            left_to_right = best_correlation_matches(
                corr,
                left_features=left_features,
                right_features=right_features,
            )
            right_to_left = best_correlation_matches(
                corr.T,
                left_features=right_features,
                right_features=left_features,
            )
            key = f"{left_sae}_vs_{right_sae}"
            report["cross_width_activation_correlation"][task][key] = {
                "aligned_rows": len(left_rows),
                "left_sae_id": left_sae,
                "right_sae_id": right_sae,
                "top_n": correlation_top_n,
                "left_to_right_summary": summarize_correlation_matches(left_to_right),
                "right_to_left_summary": summarize_correlation_matches(right_to_left),
                "left_to_right_top_matches": left_to_right[:10],
                "right_to_left_top_matches": right_to_left[:10],
            }

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--activation-site", default=DEFAULT_ACTIVATION_SITE)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-ids", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-top-n", type=int, default=100)
    parser.add_argument("--overlap-top-ns", type=parse_int_list, default=(10, 25, 50, 100))
    parser.add_argument("--correlation-top-n", type=int, default=50)
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--probe-reports", nargs="*", type=Path, default=[])
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--keep-parse-failed", action="store_true")
    args = parser.parse_args()

    best_c_values = load_best_c_values(args.probe_reports)
    report = run_analysis(
        feature_dir=args.feature_dir,
        activation_site=args.activation_site,
        model_key=args.model_key,
        tasks=args.tasks,
        layer=args.layer,
        sae_ids=args.sae_ids,
        top_k=args.top_k,
        splits_path=args.splits,
        split_family=args.split_family,
        output_top_n=args.output_top_n,
        overlap_top_ns=args.overlap_top_ns,
        correlation_top_n=args.correlation_top_n,
        c_values=args.c_values,
        max_iter=args.max_iter,
        solver=args.solver,
        drop_parse_failed=not args.keep_parse_failed,
        best_c_values=best_c_values,
    )
    write_json(args.output, report)
    print(args.output)
    for sae_id, by_task in report["models"].items():
        for task, model in by_task.items():
            print(
                f"{sae_id} {task}: C={model['best_c']} "
                f"val={model['val_auc']:.3f} test={model['test_auc']:.3f}"
            )


if __name__ == "__main__":
    main()
