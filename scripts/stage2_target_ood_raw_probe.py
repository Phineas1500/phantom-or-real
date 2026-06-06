#!/usr/bin/env python3
"""Run raw-activation probes for alternate targets and OOD splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    _class_counts,
    load_activation_matrix,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    train_logistic_probe_with_splits,
    write_json,
)


DEFAULT_MODEL_LAYERS = {
    "gemma3_27b": 45,
    "qwen35_27b": 53,
}
DEFAULT_TASKS = ("infer_property", "infer_subtype")
DEFAULT_TARGETS = ("is_correct_weak",)
DEFAULT_SPLIT_FAMILIES = ("s1", "s3", "height_h12_to_h34")


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


def target_value(row: dict[str, Any], target: str) -> int:
    if target == "is_correct_strong":
        return int(bool(row.get("is_correct_strong")))
    if target == "is_correct_weak":
        return int(bool(row.get("is_correct_weak")))
    if target == "quality_score_perfect":
        return int(row.get("quality_score") == 1.0)
    if target == "weak_not_strong":
        return int(bool(row.get("is_correct_weak")) and not bool(row.get("is_correct_strong")))
    raise ValueError(f"unknown target: {target}")


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
        sidecar_row["quality_score"] = source_row.get("quality_score")
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
        "meta": meta,
        "source_total_rows": len(source_rows),
        "source_parse_failed_rows": sum(1 for row in source_rows.values() if row.get("parse_failed")),
    }


def make_height_extrapolation_splits(
    labels: list[int],
    sidecar_rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> dict[str, list[int]]:
    low_by_label: dict[int, list[int]] = {0: [], 1: []}
    test_indices = []
    for idx, row in enumerate(sidecar_rows):
        height = int(row["height"])
        if height in (1, 2):
            low_by_label[int(labels[idx])].append(idx)
        elif height in (3, 4):
            test_indices.append(idx)
        else:
            raise ValueError(f"unexpected height {height!r}")

    train_indices = []
    val_indices = []
    for label, indices in sorted(low_by_label.items()):
        rng = random.Random(seed + label * 9176)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) >= 3:
            n_val = max(1, round(len(shuffled) * val_fraction))
            n_train = len(shuffled) - n_val
        else:
            n_train = len(shuffled)
            n_val = 0
        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train : n_train + n_val])

    return {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }


def split_for_family(
    *,
    split_family: str,
    sidecar_rows: list[dict[str, Any]],
    labels: list[int],
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
            labels,
            sidecar_rows,
            seed=seed,
            val_fraction=height_val_fraction,
        )
    raise ValueError(f"unknown split family: {split_family}")


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    assignments = read_split_assignments(args.splits)
    model_layers = [parse_model_layer(value) for value in args.model_layers]
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "raw_activation_logistic_probe",
        "representation_type": "raw direction",
        "activation_dir": str(args.activation_dir),
        "activation_site": args.activation_site,
        "splits_path": str(args.splits),
        "model_layers": [{"model_key": model_key, "layer": layer} for model_key, layer in model_layers],
        "tasks": list(args.tasks),
        "targets": list(args.targets),
        "split_families": list(args.split_families),
        "drop_parse_failed": args.drop_parse_failed,
        "seed": args.seed,
        "c_values": list(args.c_values),
        "max_iter": args.max_iter,
        "solver": args.solver,
        "bootstrap_samples": args.bootstrap_samples,
        "height_val_fraction": args.height_val_fraction,
        "include_split_indices": args.include_split_indices,
        "causal_abstraction_claim": {
            "target_variables": list(args.targets),
            "tested_representation": "raw residual activation",
            "claim_type": "predictive",
            "note": "No causal claim is made by this report; it tests target/OOD readout feasibility.",
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
            task_result: dict[str, Any] = {
                "model": model_key,
                "task": task,
                "site_or_layer": f"L{layer}/{args.activation_site}",
                "source_file": dataset["source_file"],
                "activation_prefix": dataset["activation_prefix"],
                "n": dataset["kept_rows"],
                "input_rows": dataset["input_rows"],
                "source_total_rows": dataset["source_total_rows"],
                "parse_fail_rate": (
                    dataset["source_parse_failed_rows"] / dataset["source_total_rows"]
                    if dataset["source_total_rows"]
                    else None
                ),
                "targets": {},
            }
            for target in args.targets:
                labels = [target_value(row, target) for row in dataset["joined_rows"]]
                target_result: dict[str, Any] = {}
                for split_family in args.split_families:
                    split_seed = args.seed + layer + sum(ord(ch) for ch in f"{model_key}:{task}:{target}:{split_family}")
                    splits = split_for_family(
                        split_family=split_family,
                        sidecar_rows=dataset["sidecar"],
                        labels=labels,
                        assignments=assignments,
                        source_file=dataset["source_file"],
                        seed=split_seed,
                        height_val_fraction=args.height_val_fraction,
                    )
                    probe = train_logistic_probe_with_splits(
                        dataset["x"],
                        labels,
                        dataset["sidecar"],
                        splits=splits,
                        c_values=args.c_values,
                        max_iter=args.max_iter,
                        solver=args.solver,
                        bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=split_seed,
                    )
                    probe.pop("_artifact_model", None)
                    if not args.include_split_indices:
                        probe.pop("split_indices", None)
                    target_result[split_family] = {
                        "model": model_key,
                        "task": task,
                        "target_variable": target,
                        "split": split_family,
                        "site_or_layer": f"L{layer}/{args.activation_site}",
                        "method": "logistic_probe",
                        "representation_type": "raw activation",
                        "controls": {
                            "drop_parse_failed": args.drop_parse_failed,
                            "split_source": "stage2_splits" if split_family in {"s1", "s2", "s3"} else "height_h12_to_h34",
                            "source_join": "activation sidecar row_index to full JSONL",
                        },
                        "n": dataset["kept_rows"],
                        "baseline_metrics": {
                            "target_class_counts": {
                                split: _class_counts(labels, split_indices)
                                for split, split_indices in splits.items()
                            },
                        },
                        "intervention_metrics": None,
                        "paired_flips": None,
                        "parse_fail_rate": task_result["parse_fail_rate"],
                        "matched_noise_summary": None,
                        "causal_abstraction_claim": {
                            "variable": target,
                            "representation": "raw residual activation",
                            "type": "predictive_only",
                        },
                        **probe,
                    }
                    print(
                        model_key,
                        task,
                        target,
                        split_family,
                        target_result[split_family].get("status"),
                        target_result[split_family].get("test_auc"),
                    )
                task_result["targets"][target] = target_result
            report["results"][model_key][task] = task_result
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument(
        "--model-layers",
        nargs="+",
        default=("gemma3_27b:45", "qwen35_27b:53"),
        help="Model/layer pairs, e.g. gemma3_27b:45 qwen35_27b:53.",
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--split-families", nargs="+", default=DEFAULT_SPLIT_FAMILIES)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--drop-parse-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--height-val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--include-split-indices",
        action="store_true",
        help="Include full split-index arrays in the JSON report. Disabled by default to keep validation reports compact.",
    )
    parser.add_argument("--output", type=Path, default=Path("docs/target_ood_raw_probe_27b_main.json"))
    args = parser.parse_args()

    report = run_grid(args)
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
