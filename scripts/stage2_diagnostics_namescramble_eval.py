#!/usr/bin/env python3
"""Evaluate fixed 4B logistic probes on namescramble activations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import load_activation_matrix, read_json, read_jsonl, read_split_assignments, split_indices_from_assignments  # noqa: E402


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) == 0 or len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _fit_headline_logistic(
    *,
    x: np.ndarray,
    labels: list[int],
    split_indices: dict[str, list[int]],
    c_value: float,
    max_iter: int,
) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=max_iter, solver="lbfgs"),
    )
    train = split_indices["train"]
    model.fit(x[train], [labels[idx] for idx in train])
    return model


def _collect_original_subset_scores(
    *,
    source_example_ids: list[str],
    original_sidecar: list[dict[str, Any]],
    original_scores_all: list[float],
    original_labels_all: list[int],
) -> tuple[list[float], list[int]]:
    idx_by_example = {row.get("example_id"): idx for idx, row in enumerate(original_sidecar)}
    scores: list[float] = []
    labels: list[int] = []
    for ex_id in source_example_ids:
        idx = idx_by_example.get(ex_id)
        if idx is None:
            continue
        scores.append(float(original_scores_all[idx]))
        labels.append(int(original_labels_all[idx]))
    return scores, labels


def _detect_headline_layers(probe_auc: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for task, by_layer in probe_auc["results"].items():
        best = None
        best_val = -1.0
        for layer, by_probe in by_layer.items():
            s1 = by_probe.get("logistic", {}).get("s1")
            if not s1 or s1.get("status") != "ok" or s1.get("val_auc") is None:
                continue
            if float(s1["val_auc"]) > best_val:
                best_val = float(s1["val_auc"])
                best = layer
        if best is None:
            raise ValueError(f"no headline layer for task={task}")
        out[task] = best
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-auc", type=Path, default=Path("results/stage2/probe_auc.json"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--namescramble-infer-dir", type=Path, default=Path("results/stage2/namescramble_infer"))
    parser.add_argument("--namescramble-activations-dir", type=Path, default=Path("results/stage2/activations_namescramble"))
    parser.add_argument("--model-key", default="gemma3_4b")
    parser.add_argument("--model-name", default="gemma3-4b")
    parser.add_argument("--condition", choices=("nonce", "natural"), required=True)
    parser.add_argument("--split-family", default="s1", choices=("s1", "s3"))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/stage2/probe_diagnostics_namescramble_4b.json"))
    args = parser.parse_args()

    probe_auc = read_json(args.probe_auc)
    assignments = read_split_assignments(args.splits)
    headline_layers = _detect_headline_layers(probe_auc)

    out: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model_key,
        "condition": args.condition,
        "split_family": args.split_family,
        "headline_layers": headline_layers,
        "results": {},
    }

    for task, layer_key in headline_layers.items():
        layer = int(layer_key.lstrip("L"))
        base_stem = activation_stem(model_key=args.model_key, task=task, layer=layer)
        base_prefix = args.activation_dir / base_stem
        base_meta = read_json(base_prefix.with_suffix(".meta.json"))
        source_file = base_meta["jsonl_path"]

        base_sidecar_all = read_jsonl(base_prefix.with_suffix(".example_ids.jsonl"))
        base_x_all = load_activation_matrix(base_prefix.with_suffix(".safetensors"))
        keep_base = [idx for idx, row in enumerate(base_sidecar_all) if not row.get("parse_failed")]
        base_sidecar = [base_sidecar_all[idx] for idx in keep_base]
        base_x = base_x_all[keep_base]
        base_labels = [int(row["is_correct_strong"]) for row in base_sidecar]
        split_indices = split_indices_from_assignments(
            base_sidecar,
            assignments=assignments,
            source_file=source_file,
            split_field=f"{args.split_family}_split",
        )

        c_value = float(probe_auc["results"][task][layer_key]["logistic"][args.split_family]["best_c"])
        model = _fit_headline_logistic(
            x=base_x,
            labels=base_labels,
            split_indices=split_indices,
            c_value=c_value,
            max_iter=args.max_iter,
        )
        base_probs = [float(p) for p in model.predict_proba(base_x)[:, 1]]

        cond_task = {}
        total_scrambled_scores: list[float] = []
        total_scrambled_labels: list[int] = []
        total_baseline_scores: list[float] = []
        total_baseline_labels: list[int] = []

        for h in (1, 2, 3, 4):
            infer_jsonl = args.namescramble_infer_dir / args.condition / f"{task}_h{h}.jsonl"
            if not infer_jsonl.exists():
                cond_task[f"h{h}"] = {"status": "missing_inference_jsonl", "path": str(infer_jsonl)}
                continue
            rows = read_jsonl(infer_jsonl)
            source_example_ids = [row.get("source_example_id") or row.get("namescramble", {}).get("source_example_id") for row in rows]

            ns_stem = activation_stem(model_key=f"{args.model_key}_{args.condition}", task=task, layer=layer)
            ns_prefix = args.namescramble_activations_dir / ns_stem
            ns_sidecar_path = ns_prefix.with_suffix(".example_ids.jsonl")
            ns_safetensor_path = ns_prefix.with_suffix(".safetensors")
            if not ns_sidecar_path.exists() or not ns_safetensor_path.exists():
                cond_task[f"h{h}"] = {
                    "status": "missing_activation_artifacts",
                    "sidecar": str(ns_sidecar_path),
                    "safetensors": str(ns_safetensor_path),
                }
                continue

            ns_sidecar = read_jsonl(ns_sidecar_path)
            ns_x = load_activation_matrix(ns_safetensor_path)
            ns_labels = [int(row["is_correct_strong"]) for row in ns_sidecar]
            ns_probs = [float(p) for p in model.predict_proba(ns_x)[:, 1]]

            ns_h_idx = [i for i, row in enumerate(ns_sidecar) if int(row.get("height", -1)) == h]
            ns_h_scores = [ns_probs[i] for i in ns_h_idx]
            ns_h_labels = [ns_labels[i] for i in ns_h_idx]
            base_h_scores, base_h_labels = _collect_original_subset_scores(
                source_example_ids=source_example_ids,
                original_sidecar=base_sidecar,
                original_scores_all=base_probs,
                original_labels_all=base_labels,
            )

            auc_scrambled = _safe_auc(ns_h_labels, ns_h_scores)
            auc_baseline = _safe_auc(base_h_labels, base_h_scores)
            cond_task[f"h{h}"] = {
                "status": "ok",
                "n_scrambled": len(ns_h_scores),
                "n_baseline": len(base_h_scores),
                "auc_scrambled": auc_scrambled,
                "auc_baseline": auc_baseline,
                "auc_drop": (None if auc_scrambled is None or auc_baseline is None else float(auc_baseline - auc_scrambled)),
            }

            total_scrambled_scores.extend(ns_h_scores)
            total_scrambled_labels.extend(ns_h_labels)
            total_baseline_scores.extend(base_h_scores)
            total_baseline_labels.extend(base_h_labels)

        task_summary = {
            "headline_layer": layer_key,
            "c_value": c_value,
            "per_height": cond_task,
            "auc_scrambled_all": _safe_auc(total_scrambled_labels, total_scrambled_scores),
            "auc_baseline_all": _safe_auc(total_baseline_labels, total_baseline_scores),
        }
        if task_summary["auc_scrambled_all"] is not None and task_summary["auc_baseline_all"] is not None:
            task_summary["auc_drop_all"] = float(task_summary["auc_baseline_all"] - task_summary["auc_scrambled_all"])
        else:
            task_summary["auc_drop_all"] = None
        out["results"][task] = task_summary

    write_json(args.output, out)
    print(args.output)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
