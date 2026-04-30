#!/usr/bin/env python3
"""Train logistic probes on sparse top-k SAE feature files."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    train_logistic_probe_with_splits,
    write_json,
)
from src.stage2_paths import DEFAULT_ACTIVATION_SITE, activation_stem  # noqa: E402
from src.stage2_sae import topk_tensors_to_csr  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def load_sae_dataset(
    *,
    feature_prefix: Path,
    drop_parse_failed: bool,
) -> dict[str, Any]:
    meta = read_json(feature_prefix.with_suffix(".meta.json"))
    tensors = load_file(feature_prefix.with_suffix(".safetensors"))
    x_all = topk_tensors_to_csr(
        tensors["top_indices"],
        tensors["top_values"],
        d_sae=int(meta["sae_cfg"]["d_sae"]),
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


def run_sae_probe_grid(
    *,
    feature_dir: Path,
    model_key: str,
    tasks: list[str],
    layer: int,
    sae_id: str,
    top_k: int,
    splits_path: Path,
    split_family: str,
    seed: int,
    drop_parse_failed: bool,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
    activation_site: str = DEFAULT_ACTIVATION_SITE,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_dir": str(feature_dir),
        "activation_site": activation_site,
        "model_key": model_key,
        "tasks": tasks,
        "layer": layer,
        "sae_id": sae_id,
        "top_k": top_k,
        "splits_path": str(splits_path),
        "split_family": split_family,
        "seed": seed,
        "drop_parse_failed": drop_parse_failed,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "results": {},
        "best_by_task": {},
    }
    for task in tasks:
        activation_name = activation_stem(
            model_key=model_key,
            task=task,
            layer=layer,
            activation_site=activation_site,
        )
        prefix = feature_dir / f"{activation_name}_{sae_id}_top{top_k}"
        dataset = load_sae_dataset(feature_prefix=prefix, drop_parse_failed=drop_parse_failed)
        source_file = dataset["meta"]["source_activation_meta"]["jsonl_path"]
        splits = split_indices_from_assignments(
            dataset["sidecar"],
            assignments=split_assignments,
            source_file=source_file,
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
            bootstrap_seed=seed + layer,
        )
        result = {
            key: value
            for key, value in result.items()
            if not key.startswith("_artifact_")
        }
        result.update(
            {
                "feature_path": str(prefix.with_suffix(".safetensors")),
                "sidecar_path": str(prefix.with_suffix(".example_ids.jsonl")),
                "source_file": source_file,
                "input_rows": dataset["input_rows"],
                "kept_rows": dataset["kept_rows"],
                "d_sae": int(dataset["meta"]["sae_cfg"]["d_sae"]),
                "sae_release": dataset["meta"]["sae_release"],
                "sae_id": dataset["meta"]["sae_id"],
                "hf_snapshot_revision": dataset["meta"].get("hf_snapshot_revision"),
            }
        )
        report["results"][task] = result
        report["best_by_task"][task] = {
            "layer": f"L{layer}",
            "sae_id": sae_id,
            "val_auc": result.get("val_auc"),
            "test_auc": result.get("test_auc"),
        } if result.get("status") == "ok" else None
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--activation-site", default=DEFAULT_ACTIVATION_SITE)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--keep-parse-failed", action="store_true")
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    report = run_sae_probe_grid(
        feature_dir=args.feature_dir,
        activation_site=args.activation_site,
        model_key=args.model_key,
        tasks=args.tasks,
        layer=args.layer,
        sae_id=args.sae_id,
        top_k=args.top_k,
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
