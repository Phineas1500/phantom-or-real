#!/usr/bin/env python3
"""Probe sparse SAE/crosscoder artifacts after dense active-column materialization."""

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


def run_dense_active_probe_grid(
    *,
    feature_prefixes: list[Path],
    splits_path: Path,
    split_family: str,
    seed: int,
    drop_parse_failed: bool,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    assignments = read_split_assignments(splits_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_prefixes": [str(prefix) for prefix in feature_prefixes],
        "splits_path": str(splits_path),
        "split_family": split_family,
        "seed": seed,
        "drop_parse_failed": drop_parse_failed,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "probe_variant": "dense_train_active_columns_centered",
        "results": {},
    }
    for prefix in feature_prefixes:
        dataset = load_sparse_feature_dataset(prefix, drop_parse_failed=drop_parse_failed)
        source_file = sparse_feature_source_file(dataset["meta"])
        splits = split_indices_from_assignments(
            dataset["sidecar"],
            assignments=assignments,
            source_file=source_file,
            split_field=f"{split_family}_split",
        )
        active_feature_ids = train_active_feature_ids(dataset["x"], splits["train"])
        dense_x = dense_active_matrix(dataset["x"], active_feature_ids)
        result = train_logistic_probe_with_splits(
            dense_x,
            dataset["labels"],
            dataset["sidecar"],
            splits=splits,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed,
        )
        result.update(
            {
                "feature_path": str(prefix.with_suffix(".safetensors")),
                "sidecar_path": str(prefix.with_suffix(".example_ids.jsonl")),
                "source_file": source_file,
                "input_rows": dataset["input_rows"],
                "kept_rows": dataset["kept_rows"],
                "d_sparse": sparse_feature_width(dataset["meta"]),
                "active_feature_n": int(active_feature_ids.size),
                "active_feature_source": "train_nonzero",
                "active_feature_ids_sample": [int(feature) for feature in active_feature_ids[:25]],
            }
        )
        report["results"][prefix.name] = result
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-prefix", type=Path, action="append", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--keep-parse-failed", action="store_true")
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    report = run_dense_active_probe_grid(
        feature_prefixes=args.feature_prefix,
        splits_path=args.splits,
        split_family=args.split_family,
        seed=args.seed,
        drop_parse_failed=not args.keep_parse_failed,
        c_values=args.c_values,
        max_iter=args.max_iter,
        solver=args.solver,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_json(args.output, report)
    print(args.output)
    for name, result in report["results"].items():
        print(
            f"{name}: active={result.get('active_feature_n')} "
            f"val={result.get('val_auc')} test={result.get('test_auc')}"
        )


if __name__ == "__main__":
    main()
