#!/usr/bin/env python3
"""Train logistic probes on concatenated sparse Stage 2 feature artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_dense_active import (  # noqa: E402
    dense_active_matrix,
    sparse_feature_source_file,
    sparse_feature_width,
    train_active_feature_ids,
)
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    train_logistic_probe_with_splits,
    write_json,
)
from src.stage2_sae import topk_tensors_to_csr  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def expand_task_pattern(pattern: str, *, task: str) -> Path:
    try:
        expanded = pattern.format(task=task)
    except KeyError as exc:
        raise ValueError(f"unknown pattern placeholder in {pattern!r}: {exc}") from exc
    return Path(expanded)


def load_sparse_feature_dataset(
    feature_prefix: Path,
    *,
    drop_parse_failed: bool,
) -> dict[str, Any]:
    meta = read_json(feature_prefix.with_suffix(".meta.json"))
    tensors = load_file(feature_prefix.with_suffix(".safetensors"))
    x_all = topk_tensors_to_csr(
        tensors["top_indices"],
        tensors["top_values"],
        d_sae=sparse_feature_width(meta),
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
    sidecar = [sidecar_all[idx] for idx in keep_indices]
    return {
        "x": x_all[keep_indices],
        "labels": [int(row["is_correct_strong"]) for row in sidecar],
        "sidecar": sidecar,
        "meta": meta,
        "source_file": sparse_feature_source_file(meta),
        "prefix": feature_prefix,
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
        "d_sparse": sparse_feature_width(meta),
        "nnz": int(x_all[keep_indices].nnz),
    }


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_file"),
        row.get("row_index"),
        row.get("example_id"),
        row.get("task"),
        row.get("height"),
        row.get("is_correct_strong"),
        row.get("parse_failed"),
    )


def assert_aligned_datasets(datasets: list[dict[str, Any]]) -> None:
    if len(datasets) < 2:
        return
    reference = datasets[0]
    reference_keys = [_row_key(row) for row in reference["sidecar"]]
    reference_labels = reference["labels"]
    reference_source = reference["source_file"]
    for dataset in datasets[1:]:
        if dataset["source_file"] != reference_source:
            raise ValueError(
                "all concatenated features must come from the same source file; "
                f"got {reference_source!r} and {dataset['source_file']!r}"
            )
        if dataset["labels"] != reference_labels:
            raise ValueError(f"labels differ between {reference['prefix']} and {dataset['prefix']}")
        keys = [_row_key(row) for row in dataset["sidecar"]]
        if keys != reference_keys:
            for idx, (left, right) in enumerate(zip(reference_keys, keys, strict=False)):
                if left != right:
                    raise ValueError(
                        f"sidecar mismatch at kept row {idx}: {reference['prefix']} has {left}, "
                        f"{dataset['prefix']} has {right}"
                    )
            if len(keys) != len(reference_keys):
                raise ValueError(
                    f"sidecar length mismatch: {reference['prefix']} has {len(reference_keys)}, "
                    f"{dataset['prefix']} has {len(keys)}"
                )


def concatenate_sparse_datasets(datasets: list[dict[str, Any]]) -> Any:
    from scipy import sparse

    assert_aligned_datasets(datasets)
    return sparse.hstack([dataset["x"] for dataset in datasets], format="csr")


def run_sparse_concat_probe_grid(
    *,
    combo_name: str,
    feature_patterns: list[str],
    tasks: list[str],
    splits_path: Path,
    split_family: str,
    seed: int,
    drop_parse_failed: bool,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
    dense_active: bool,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "combo_name": combo_name,
        "feature_patterns": feature_patterns,
        "tasks": tasks,
        "splits_path": str(splits_path),
        "split_family": split_family,
        "seed": seed,
        "drop_parse_failed": drop_parse_failed,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "probe_variant": "dense_train_active_sparse_hstack_centered"
        if dense_active
        else "sparse_hstack_uncentered",
        "results": {},
        "best_by_task": {},
    }

    for task in tasks:
        prefixes = [expand_task_pattern(pattern, task=task) for pattern in feature_patterns]
        datasets = [load_sparse_feature_dataset(prefix, drop_parse_failed=drop_parse_failed) for prefix in prefixes]
        x = concatenate_sparse_datasets(datasets)
        reference = datasets[0]
        splits = split_indices_from_assignments(
            reference["sidecar"],
            assignments=split_assignments,
            source_file=reference["source_file"],
            split_field=f"{split_family}_split",
        )
        active_feature_ids = None
        x_probe = x
        if dense_active:
            active_feature_ids = train_active_feature_ids(x, splits["train"])
            x_probe = dense_active_matrix(x, active_feature_ids)
        result = train_logistic_probe_with_splits(
            x_probe,
            reference["labels"],
            reference["sidecar"],
            splits=splits,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed,
        )
        feature_blocks = [
            {
                "feature_prefix": str(dataset["prefix"]),
                "feature_path": str(dataset["prefix"].with_suffix(".safetensors")),
                "sidecar_path": str(dataset["prefix"].with_suffix(".example_ids.jsonl")),
                "d_sparse": dataset["d_sparse"],
                "kept_nnz": dataset["nnz"],
                "kept_mean_l0": dataset["nnz"] / dataset["kept_rows"] if dataset["kept_rows"] else None,
                "sae_release": dataset["meta"].get("sae_release"),
                "sae_id": dataset["meta"].get("sae_id"),
                "hf_snapshot_revision": dataset["meta"].get("hf_snapshot_revision"),
            }
            for dataset in datasets
        ]
        result.update(
            {
                "source_file": reference["source_file"],
                "input_rows": reference["input_rows"],
                "kept_rows": reference["kept_rows"],
                "d_sparse_concat": int(x.shape[1]),
                "kept_nnz_concat": int(x.nnz),
                "kept_mean_l0_concat": int(x.nnz) / reference["kept_rows"] if reference["kept_rows"] else None,
                "feature_blocks": feature_blocks,
            }
        )
        if active_feature_ids is not None:
            result.update(
                {
                    "active_feature_n": int(active_feature_ids.size),
                    "active_feature_source": "train_nonzero_concat_columns",
                    "active_feature_ids_sample": [int(feature) for feature in active_feature_ids[:25]],
                }
            )
        report["results"][task] = result
        report["best_by_task"][task] = {
            "combo_name": combo_name,
            "val_auc": result.get("val_auc"),
            "test_auc": result.get("test_auc"),
            "best_c": result.get("best_c"),
        } if result.get("status") == "ok" else None
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo-name", required=True)
    parser.add_argument(
        "--feature-pattern",
        action="append",
        required=True,
        help="Feature prefix pattern. Use {task} to expand over --tasks.",
    )
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--keep-parse-failed", action="store_true")
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--dense-active",
        action="store_true",
        help="Materialize train-active concat columns as dense centered features before probing.",
    )
    args = parser.parse_args()

    report = run_sparse_concat_probe_grid(
        combo_name=args.combo_name,
        feature_patterns=args.feature_pattern,
        tasks=args.tasks,
        splits_path=args.splits,
        split_family=args.split_family,
        seed=args.seed,
        drop_parse_failed=not args.keep_parse_failed,
        c_values=args.c_values,
        max_iter=args.max_iter,
        solver=args.solver,
        bootstrap_samples=args.bootstrap_samples,
        dense_active=args.dense_active,
    )
    write_json(args.output, report)
    print(args.output)
    for task, best in report["best_by_task"].items():
        print(f"{task}: {best}")


if __name__ == "__main__":
    main()
