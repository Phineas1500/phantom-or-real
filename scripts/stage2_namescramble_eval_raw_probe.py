#!/usr/bin/env python3
"""Evaluate original raw probes on name-scrambled regenerated activations."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    _class_counts,
    _safe_auc,
    _safe_balanced_accuracy,
    load_probe_dataset,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    train_logistic_probe_with_splits,
    write_json,
)
from src.stage2_steering import parse_float_list  # noqa: E402


def parse_task_layers(value: str, tasks: list[str]) -> dict[str, int]:
    if "=" not in value:
        layer = int(value)
        return {task: layer for task in tasks}
    out: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        task, layer_text = part.split("=", 1)
        out[task.strip()] = int(layer_text)
    missing = [task for task in tasks if task not in out]
    if missing:
        raise ValueError(f"--layers missing task(s): {missing}")
    return out


def positive_probs(model: Any, x: np.ndarray) -> list[float]:
    logreg = model[-1]
    positive_index = int(np.where(logreg.classes_ == 1)[0][0])
    return [float(v) for v in model.predict_proba(x)[:, positive_index]]


def summarize_scores(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    labels = [int(row["is_correct_strong"]) for row in rows]
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    by_height: dict[str, Any] = {}
    for height in sorted({int(row["height"]) for row in rows}):
        idxs = [idx for idx, row in enumerate(rows) if int(row["height"]) == height]
        h_labels = [labels[idx] for idx in idxs]
        h_scores = [scores[idx] for idx in idxs]
        h_preds = [predictions[idx] for idx in idxs]
        by_height[f"h{height}"] = {
            **_class_counts(labels, idxs),
            "auc": _safe_auc(h_labels, h_scores),
            "strong_accuracy": sum(h_labels) / len(h_labels) if h_labels else None,
            "balanced_accuracy_at_0p5": _safe_balanced_accuracy(h_labels, h_preds),
        }
    return {
        "n": len(rows),
        "auc": _safe_auc(labels, scores),
        "strong_accuracy": sum(labels) / len(labels) if labels else None,
        "balanced_accuracy_at_0p5": _safe_balanced_accuracy(labels, predictions),
        "by_height": by_height,
    }


def finite_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return float(a - b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--namescramble-infer-dir", type=Path, default=Path("results/stage2/namescramble_infer"))
    parser.add_argument(
        "--namescramble-activations-dir",
        type=Path,
        default=Path("results/stage2/activations_namescramble"),
    )
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--tasks", nargs="+", default=("infer_property", "infer_subtype"))
    parser.add_argument("--layers", default="45", help="Either one layer or task=layer pairs, e.g. infer_property=45,infer_subtype=45")
    parser.add_argument("--conditions", nargs="+", default=("nonce", "natural"))
    parser.add_argument("--split-family", default="s1", choices=("s1", "s3"))
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--drop-parse-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("docs/namescramble_raw_probe_eval.json"))
    args = parser.parse_args()

    tasks = list(args.tasks)
    task_layers = parse_task_layers(args.layers, tasks)
    assignments = read_split_assignments(args.splits)
    c_values = parse_float_list(args.c_values)

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model_key,
        "tasks": tasks,
        "task_layers": task_layers,
        "conditions": list(args.conditions),
        "split_family": args.split_family,
        "c_values": list(c_values),
        "results": {},
    }

    for task in tasks:
        layer = task_layers[task]
        base_stem = activation_stem(model_key=args.model_key, task=task, layer=layer)
        base_prefix = args.activation_dir / base_stem
        base_meta = read_json(base_prefix.with_suffix(".meta.json"))
        source_file = base_meta["jsonl_path"]
        base_dataset = load_probe_dataset(
            activation_path=base_prefix.with_suffix(".safetensors"),
            sidecar_path=base_prefix.with_suffix(".example_ids.jsonl"),
            drop_parse_failed=args.drop_parse_failed,
        )
        base_sidecar = base_dataset["sidecar"]
        base_x = base_dataset["x"]
        base_labels = base_dataset["labels"]
        splits = split_indices_from_assignments(
            base_sidecar,
            assignments=assignments,
            source_file=source_file,
            split_field=f"{args.split_family}_split",
        )
        probe = train_logistic_probe_with_splits(
            base_x,
            base_labels,
            base_sidecar,
            splits=splits,
            c_values=c_values,
            max_iter=args.max_iter,
            solver=args.solver,
            bootstrap_samples=0,
            bootstrap_seed=0,
        )
        if probe.get("status") != "ok":
            report["results"][task] = {"status": probe.get("status"), "probe": probe}
            continue
        model = probe.pop("_artifact_model")
        base_scores = positive_probs(model, base_x)
        base_by_example = {
            str(row["example_id"]): (row, base_scores[idx])
            for idx, row in enumerate(base_sidecar)
        }

        task_report: dict[str, Any] = {
            "status": "ok",
            "layer": layer,
            "source_file": source_file,
            "base_activation_prefix": str(base_prefix),
            "original_probe": probe,
            "conditions": {},
        }

        for condition in args.conditions:
            infer_path = args.namescramble_infer_dir / condition / f"{task}.jsonl"
            ns_rows_all = read_jsonl(infer_path)
            ns_stem = activation_stem(model_key=f"{args.model_key}_{condition}", task=task, layer=layer)
            ns_prefix = args.namescramble_activations_dir / ns_stem
            ns_dataset = load_probe_dataset(
                activation_path=ns_prefix.with_suffix(".safetensors"),
                sidecar_path=ns_prefix.with_suffix(".example_ids.jsonl"),
                drop_parse_failed=args.drop_parse_failed,
            )
            ns_sidecar = ns_dataset["sidecar"]
            ns_x = ns_dataset["x"]
            ns_scores = positive_probs(model, ns_x)

            source_ids: list[str] = []
            missing_source_ids: list[int] = []
            ns_eval_rows: list[dict[str, Any]] = []
            for sidecar_row in ns_sidecar:
                row_index = int(sidecar_row["row_index"])
                ns_row = ns_rows_all[row_index]
                ns_eval_rows.append(ns_row)
                source_id = (
                    ns_row.get("source_example_id")
                    or ns_row.get("namescramble", {}).get("source_example_id")
                )
                if source_id is None:
                    missing_source_ids.append(row_index)
                else:
                    source_ids.append(str(source_id))

            base_subset_rows: list[dict[str, Any]] = []
            base_subset_scores: list[float] = []
            unmatched_source_ids: list[str] = []
            for source_id in source_ids:
                match = base_by_example.get(source_id)
                if match is None:
                    unmatched_source_ids.append(source_id)
                    continue
                row, score = match
                base_subset_rows.append(row)
                base_subset_scores.append(score)

            scrambled_summary = summarize_scores(ns_eval_rows, ns_scores)
            baseline_summary = summarize_scores(base_subset_rows, base_subset_scores)
            task_report["conditions"][condition] = {
                "status": "ok",
                "infer_jsonl": str(infer_path),
                "activation_prefix": str(ns_prefix),
                "input_rows": ns_dataset["input_rows"],
                "kept_rows": ns_dataset["kept_rows"],
                "missing_source_id_rows": missing_source_ids[:20],
                "missing_source_id_count": len(missing_source_ids),
                "unmatched_source_id_count": len(unmatched_source_ids),
                "scrambled": scrambled_summary,
                "matched_original_subset": baseline_summary,
                "auc_drop_vs_matched_original": finite_delta(
                    baseline_summary["auc"],
                    scrambled_summary["auc"],
                ),
                "strong_accuracy_delta_vs_matched_original": finite_delta(
                    scrambled_summary["strong_accuracy"],
                    baseline_summary["strong_accuracy"],
                ),
            }

        report["results"][task] = task_report

    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
