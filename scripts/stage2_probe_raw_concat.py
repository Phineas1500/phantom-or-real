#!/usr/bin/env python3
"""Train logistic probes on concatenated raw activation layers."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list  # noqa: E402
from src.stage2_crosscoder import verify_matching_sidecars  # noqa: E402
from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    load_activation_matrix,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    train_logistic_probe_with_splits,
    write_json,
)


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def load_concat_dataset(
    *,
    activation_dir: Path,
    model_key: str,
    task: str,
    layers: list[int],
    drop_parse_failed: bool,
) -> dict[str, Any]:
    matrices = []
    metas = []
    sidecars_by_layer = []
    activation_paths = []
    for layer in layers:
        prefix = activation_dir / activation_stem(model_key=model_key, task=task, layer=layer)
        activation_path = prefix.with_suffix(".safetensors")
        activation_paths.append(activation_path)
        matrices.append(load_activation_matrix(activation_path))
        sidecars_by_layer.append(read_jsonl(prefix.with_suffix(".example_ids.jsonl")))
        metas.append(read_json(prefix.with_suffix(".meta.json")))
    reference_rows = sidecars_by_layer[0]
    for rows in sidecars_by_layer[1:]:
        verify_matching_sidecars(reference_rows, rows)
    keep_indices = [
        idx
        for idx, row in enumerate(reference_rows)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    x = np.concatenate([matrix[keep_indices] for matrix in matrices], axis=1)
    sidecar = [reference_rows[idx] for idx in keep_indices]
    return {
        "x": x,
        "labels": [int(row["is_correct_strong"]) for row in sidecar],
        "sidecar": sidecar,
        "metas": metas,
        "activation_paths": activation_paths,
        "input_rows": len(reference_rows),
        "kept_rows": len(sidecar),
        "d_model_concat": int(x.shape[1]),
        "source_file": metas[0].get("jsonl_path"),
    }


def run_raw_concat_probe_grid(
    *,
    activation_dir: Path,
    model_key: str,
    tasks: list[str],
    layers: list[int],
    splits_path: Path,
    split_family: str,
    seed: int,
    drop_parse_failed: bool,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    layer_key = "concat_" + "_".join(f"L{layer}" for layer in layers)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_dir": str(activation_dir),
        "model_key": model_key,
        "tasks": tasks,
        "layers": layers,
        "seed": seed,
        "drop_parse_failed": drop_parse_failed,
        "splits_path": str(splits_path),
        "split_family": split_family,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "results": {},
        "best_by_task": {},
    }
    for task in tasks:
        dataset = load_concat_dataset(
            activation_dir=activation_dir,
            model_key=model_key,
            task=task,
            layers=layers,
            drop_parse_failed=drop_parse_failed,
        )
        splits = split_indices_from_assignments(
            dataset["sidecar"],
            assignments=split_assignments,
            source_file=dataset["source_file"],
            split_field=f"{split_family}_split",
        )
        result = train_logistic_probe_with_splits(
            dataset["x"],
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
                "activation_paths": [str(path) for path in dataset["activation_paths"]],
                "source_file": dataset["source_file"],
                "input_rows": dataset["input_rows"],
                "kept_rows": dataset["kept_rows"],
                "d_model_concat": dataset["d_model_concat"],
            }
        )
        report["results"][task] = {layer_key: result}
        report["best_by_task"][task] = {
            "layer": layer_key,
            "val_auc": result.get("val_auc"),
            "test_auc": result.get("test_auc"),
        } if result.get("status") == "ok" else None
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--keep-parse-failed", action="store_true")
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    report = run_raw_concat_probe_grid(
        activation_dir=args.activation_dir,
        model_key=args.model_key,
        tasks=args.tasks,
        layers=args.layers,
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
    for task, best in report["best_by_task"].items():
        print(f"{task}: {best}")


if __name__ == "__main__":
    main()
