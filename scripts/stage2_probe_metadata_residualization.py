#!/usr/bin/env python3
"""Residualize raw probe scores against prompt/name-frequency metadata."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_phase0 import (  # noqa: E402
    build_name_counts,
    feature_names,
    feature_vector,
    read_jsonl_rows,
)
from src.stage2_probes import (  # noqa: E402
    _safe_auc,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    write_json,
)
from src.stage2_paths import activation_stem  # noqa: E402


SPLIT_SUFFIX = {
    "s1": "s1",
    "s3": "s3_target_symbol",
}
METADATA_SETS = ("prompt_token_count", "b0_prompt", "b0_namefreq")


def load_raw_dataset(
    *,
    activation_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    split_assignments: dict[tuple[str, int], dict[str, Any]],
    split_family: str,
) -> dict[str, Any]:
    prefix = activation_dir / activation_stem(model_key=model_key, task=task, layer=layer)
    meta = read_json(prefix.with_suffix(".meta.json"))
    source_file = meta["jsonl_path"]
    source_path = Path(source_file)
    source_rows = {row_index: row for row_index, row in read_jsonl_rows(source_path)}

    x_all = load_file(prefix.with_suffix(".safetensors"))["activations"].float().cpu().numpy()
    sidecar_all = read_jsonl(prefix.with_suffix(".example_ids.jsonl"))
    if x_all.shape[0] != len(sidecar_all):
        raise ValueError(f"{prefix}.safetensors rows {x_all.shape[0]} != sidecar rows {len(sidecar_all)}")

    keep_indices = [
        idx
        for idx, row in enumerate(sidecar_all)
        if not row.get("parse_failed")
    ]
    sidecar = [sidecar_all[idx] for idx in keep_indices]
    x = x_all[keep_indices]
    labels = [int(row["is_correct_strong"]) for row in sidecar]
    splits = split_indices_from_assignments(
        sidecar,
        assignments=split_assignments,
        source_file=source_file,
        split_field=f"{split_family}_split",
    )

    records = []
    for row in sidecar:
        source_row = source_rows[int(row["row_index"])]
        split_row = split_assignments[(source_file, int(row["row_index"]))]
        records.append(
            {
                "source_file": source_file,
                "row_index": int(row["row_index"]),
                "row_id": f"{source_file}:{row['row_index']}",
                "model": row["model"],
                "task": row["task"],
                "height": int(row["height"]),
                "example_id": row.get("example_id"),
                "is_correct_strong": bool(row["is_correct_strong"]),
                "parse_failed": bool(row["parse_failed"]),
                "prompt_token_count": int(row["token_count"]),
                "prompt_length_mode": "activation_sidecar_token_count",
                "row": source_row,
                **{key: split_row[key] for key in ("s1_split", "s2_split", "s3_split") if key in split_row},
            }
        )

    return {
        "x": x,
        "labels": labels,
        "sidecar": sidecar,
        "records": records,
        "splits": splits,
        "source_file": source_file,
        "activation_prefix": str(prefix),
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
        "d_model": int(x.shape[1]),
    }


def train_logistic_scores(
    x: np.ndarray,
    labels: list[int],
    splits: dict[str, list[int]],
    *,
    c_value: float,
    max_iter: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=max_iter),
    )
    train_indices = splits["train"]
    model.fit(x[train_indices], [labels[idx] for idx in train_indices])
    scores = model.decision_function(x)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "model": model,
        "scores": scores,
        "train_auc": auc_for_indices(labels, scores, splits["train"]),
        "val_auc": auc_for_indices(labels, scores, splits["val"]),
        "test_auc": auc_for_indices(labels, scores, splits["test"]),
    }


def auc_for_indices(labels: list[int], scores: np.ndarray, indices: list[int]) -> float | None:
    return _safe_auc([labels[idx] for idx in indices], [float(scores[idx]) for idx in indices])


def pearson_for_indices(labels: list[int], scores: np.ndarray, indices: list[int]) -> float | None:
    if not indices:
        return None
    y = np.asarray([labels[idx] for idx in indices], dtype=np.float64)
    x = scores[indices].astype(np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metadata_matrix(
    records: list[dict[str, Any]],
    indices: list[int],
    metadata_set: str,
    *,
    name_counts: dict[bool, Any] | None,
) -> np.ndarray:
    if metadata_set == "prompt_token_count":
        return np.asarray([[float(records[idx]["prompt_token_count"])] for idx in indices], dtype=np.float64)
    return np.asarray(
        [
            feature_vector(records[idx], metadata_set, name_counts=name_counts)
            for idx in indices
        ],
        dtype=np.float64,
    )


def metadata_feature_names(metadata_set: str) -> list[str]:
    if metadata_set == "prompt_token_count":
        return ["prompt_token_count"]
    return feature_names(metadata_set)


def train_metadata_scores(
    records: list[dict[str, Any]],
    labels: list[int],
    splits: dict[str, list[int]],
    metadata_set: str,
    *,
    c_values: tuple[float, ...],
    max_iter: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_records = [records[idx] for idx in splits["train"]]
    name_counts = build_name_counts(train_records) if metadata_set == "b0_namefreq" else None
    x_train = metadata_matrix(records, splits["train"], metadata_set, name_counts=name_counts)
    x_val = metadata_matrix(records, splits["val"], metadata_set, name_counts=name_counts)
    x_test = metadata_matrix(records, splits["test"], metadata_set, name_counts=name_counts)
    y_train = [labels[idx] for idx in splits["train"]]

    best = None
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=max_iter),
        )
        model.fit(x_train, y_train)
        val_scores = np.asarray(model.decision_function(x_val), dtype=np.float64)
        val_auc = _safe_auc([labels[idx] for idx in splits["val"]], [float(score) for score in val_scores])
        rank_auc = val_auc if val_auc is not None else -math.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "c": c_value,
                "val_auc": val_auc,
                "rank_auc": rank_auc,
            }
    assert best is not None
    test_scores = np.asarray(best["model"].decision_function(x_test), dtype=np.float64)
    return {
        "best_c": best["c"],
        "val_auc": best["val_auc"],
        "test_auc": _safe_auc([labels[idx] for idx in splits["test"]], [float(score) for score in test_scores]),
        "feature_names": metadata_feature_names(metadata_set),
        "feature_count": len(metadata_feature_names(metadata_set)),
    }


def train_metadata_plus_raw_scores(
    records: list[dict[str, Any]],
    labels: list[int],
    splits: dict[str, list[int]],
    metadata_set: str,
    raw_scores: np.ndarray,
    *,
    c_values: tuple[float, ...],
    max_iter: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_records = [records[idx] for idx in splits["train"]]
    name_counts = build_name_counts(train_records) if metadata_set == "b0_namefreq" else None

    def combined_matrix(indices: list[int]) -> np.ndarray:
        x_meta = metadata_matrix(records, indices, metadata_set, name_counts=name_counts)
        raw_column = raw_scores[indices].reshape(-1, 1)
        return np.hstack([x_meta, raw_column])

    x_train = combined_matrix(splits["train"])
    x_val = combined_matrix(splits["val"])
    x_test = combined_matrix(splits["test"])
    y_train = [labels[idx] for idx in splits["train"]]

    best = None
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=max_iter),
        )
        model.fit(x_train, y_train)
        val_scores = np.asarray(model.decision_function(x_val), dtype=np.float64)
        val_auc = _safe_auc([labels[idx] for idx in splits["val"]], [float(score) for score in val_scores])
        rank_auc = val_auc if val_auc is not None else -math.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "c": c_value,
                "val_auc": val_auc,
                "rank_auc": rank_auc,
            }
    assert best is not None
    test_scores = np.asarray(best["model"].decision_function(x_test), dtype=np.float64)
    return {
        "best_c": best["c"],
        "val_auc": best["val_auc"],
        "test_auc": _safe_auc([labels[idx] for idx in splits["test"]], [float(score) for score in test_scores]),
        "feature_names": [*metadata_feature_names(metadata_set), "raw_probe_score"],
        "feature_count": len(metadata_feature_names(metadata_set)) + 1,
    }


def residualize_scores(
    records: list[dict[str, Any]],
    labels: list[int],
    splits: dict[str, list[int]],
    raw_scores: np.ndarray,
    metadata_set: str,
) -> dict[str, Any]:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_records = [records[idx] for idx in splits["train"]]
    name_counts = build_name_counts(train_records) if metadata_set == "b0_namefreq" else None
    all_indices = list(range(len(records)))
    x_all = metadata_matrix(records, all_indices, metadata_set, name_counts=name_counts)
    train_indices = splits["train"]
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_all[train_indices], raw_scores[train_indices])
    predicted = np.asarray(model.predict(x_all), dtype=np.float64)
    residual = raw_scores - predicted
    return {
        "metadata_set": metadata_set,
        "feature_names": metadata_feature_names(metadata_set),
        "feature_count": len(metadata_feature_names(metadata_set)),
        "train_r2_raw_score": float(model.score(x_all[train_indices], raw_scores[train_indices])),
        "val_auc": auc_for_indices(labels, residual, splits["val"]),
        "test_auc": auc_for_indices(labels, residual, splits["test"]),
        "test_pearson": pearson_for_indices(labels, residual, splits["test"]),
    }


def reported_raw_c(docs_dir: Path, split_family: str, task: str, layer: int) -> float:
    report = read_json(docs_dir / f"raw_probe_27b_{SPLIT_SUFFIX[split_family]}.json")
    layer_report = report["results"][task][f"L{layer}"]
    return float(layer_report["best_c"])


def reported_raw_auc(docs_dir: Path, split_family: str, task: str, layer: int) -> float:
    report = read_json(docs_dir / f"raw_probe_27b_{SPLIT_SUFFIX[split_family]}.json")
    layer_report = report["results"][task][f"L{layer}"]
    return float(layer_report["test_auc"])


def run_diagnostic(
    *,
    docs_dir: Path,
    activation_dir: Path,
    splits_path: Path,
    model_key: str,
    tasks: list[str],
    layer: int,
    split_families: list[str],
    c_values: tuple[float, ...],
    max_iter: int,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic": "raw_score_metadata_residualization",
        "model_key": model_key,
        "layer": layer,
        "activation_dir": str(activation_dir),
        "splits_path": str(splits_path),
        "tasks": tasks,
        "split_families": split_families,
        "metadata_sets": list(METADATA_SETS),
        "metadata_c_values": list(c_values),
        "max_iter": max_iter,
        "results": {},
    }

    for split_family in split_families:
        report["results"][split_family] = {}
        for task in tasks:
            dataset = load_raw_dataset(
                activation_dir=activation_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                split_assignments=split_assignments,
                split_family=split_family,
            )
            raw_c = reported_raw_c(docs_dir, split_family, task, layer)
            raw = train_logistic_scores(
                dataset["x"],
                dataset["labels"],
                dataset["splits"],
                c_value=raw_c,
                max_iter=max_iter,
            )
            task_result: dict[str, Any] = {
                "source_file": dataset["source_file"],
                "activation_prefix": dataset["activation_prefix"],
                "input_rows": dataset["input_rows"],
                "kept_rows": dataset["kept_rows"],
                "d_model": dataset["d_model"],
                "raw": {
                    "best_c_from_report": raw_c,
                    "reported_test_auc": reported_raw_auc(docs_dir, split_family, task, layer),
                    "refit_train_auc": raw["train_auc"],
                    "refit_val_auc": raw["val_auc"],
                    "refit_test_auc": raw["test_auc"],
                },
                "metadata_only": {},
                "metadata_plus_raw_score": {},
                "raw_residualized": {},
            }
            for metadata_set in METADATA_SETS:
                metadata_scores = train_metadata_scores(
                    dataset["records"],
                    dataset["labels"],
                    dataset["splits"],
                    metadata_set,
                    c_values=c_values,
                    max_iter=max_iter,
                )
                combined_scores = train_metadata_plus_raw_scores(
                    dataset["records"],
                    dataset["labels"],
                    dataset["splits"],
                    metadata_set,
                    raw["scores"],
                    c_values=c_values,
                    max_iter=max_iter,
                )
                residual = residualize_scores(
                    dataset["records"],
                    dataset["labels"],
                    dataset["splits"],
                    raw["scores"],
                    metadata_set,
                )
                raw_auc = float(raw["test_auc"])
                residual_auc = residual["test_auc"]
                task_result["metadata_only"][metadata_set] = metadata_scores
                task_result["metadata_plus_raw_score"][metadata_set] = {
                    **combined_scores,
                    "test_auc_delta_vs_metadata": (
                        float(combined_scores["test_auc"]) - float(metadata_scores["test_auc"])
                        if combined_scores["test_auc"] is not None and metadata_scores["test_auc"] is not None
                        else None
                    ),
                    "test_auc_delta_vs_raw": (
                        float(combined_scores["test_auc"]) - raw_auc
                        if combined_scores["test_auc"] is not None
                        else None
                    ),
                }
                task_result["raw_residualized"][metadata_set] = {
                    **residual,
                    "test_auc_drop_vs_raw": raw_auc - float(residual_auc) if residual_auc is not None else None,
                    "test_auc_delta_vs_metadata": (
                        float(residual_auc) - float(metadata_scores["test_auc"])
                        if residual_auc is not None and metadata_scores["test_auc"] is not None
                        else None
                    ),
                }
            report["results"][split_family][task] = task_result
            print(
                split_family,
                task,
                {
                    "raw": task_result["raw"]["refit_test_auc"],
                    "resid_namefreq": task_result["raw_residualized"]["b0_namefreq"]["test_auc"],
                },
            )
    return report


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--tasks", nargs="+", default=["infer_property", "infer_subtype"])
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--split-families", nargs="+", default=["s1", "s3"])
    parser.add_argument("--c-values", type=parse_float_list, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("docs/raw_probe_metadata_residualization_27b_l45.json"))
    args = parser.parse_args()

    report = run_diagnostic(
        docs_dir=args.docs_dir,
        activation_dir=args.activation_dir,
        splits_path=args.splits,
        model_key=args.model_key,
        tasks=args.tasks,
        layer=args.layer,
        split_families=args.split_families,
        c_values=args.c_values,
        max_iter=args.max_iter,
    )
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
