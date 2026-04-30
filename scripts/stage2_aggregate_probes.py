#!/usr/bin/env python3
"""Aggregate Stage 2 probe raw outputs into consolidated deliverables.

Builds:
- results/stage2/probe_auc.json
- results/stage2/probe_transfer.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def discover_probe_jsons(docs_dir: Path, model_tag: str) -> list[Path]:
    files = []
    for split in ("s1", "s2", "s3"):
        for probe_type in ("logistic", "diffmeans"):
            candidate = docs_dir / f"raw_probe_{model_tag}_{split}.{probe_type}.json"
            if candidate.exists():
                files.append(candidate)
    return files


def discover_transfer_jsons(docs_dir: Path, model_tag: str) -> list[Path]:
    return sorted(docs_dir.glob(f"raw_probe_transfer_{model_tag}_*.json"))


def _model_name_for_b0(model_key: str) -> str:
    # e.g. gemma3_4b -> gemma3-4b
    return model_key.replace("_", "-")


def _delta(test_auc: float | None, b0_auc: float | None) -> float | None:
    if test_auc is None or b0_auc is None:
        return None
    return float(test_auc - b0_auc)


def aggregate_probe_auc(
    *,
    probe_files: list[Path],
    b0_summary: dict[str, Any],
) -> dict[str, Any]:
    if not probe_files:
        raise ValueError("no probe files provided")

    first = read_json(probe_files[0])
    model_key = first["model_key"]
    model_name = _model_name_for_b0(model_key)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "model_name": model_name,
        "source_probe_files": [str(path) for path in probe_files],
        "b0_summary_source": b0_summary.get("source"),
        "results": {},
    }

    best_b0 = b0_summary["best_pre_output_baseline"]

    for probe_path in probe_files:
        report = read_json(probe_path)
        split = report["split_family"]
        probe_type = report.get("probe_type", "logistic")
        if report["model_key"] != model_key:
            raise ValueError(f"model mismatch in {probe_path}: {report['model_key']} != {model_key}")

        for task, by_layer in report["results"].items():
            task_out = payload["results"].setdefault(task, {})
            b0_key = f"{model_name}__{task}"
            b0_cell = best_b0.get(b0_key, {})
            b0_for_split = b0_cell.get(split)
            b0_auc = None if b0_for_split is None else b0_for_split.get("test_auc")
            b0_feature = None if b0_for_split is None else b0_for_split.get("feature_set")

            for layer, cell in by_layer.items():
                layer_out = task_out.setdefault(layer, {})
                probe_out = layer_out.setdefault(probe_type, {})
                if split in probe_out:
                    raise ValueError(
                        f"duplicate entry task={task} layer={layer} probe_type={probe_type} split={split}"
                    )

                split_counts = cell.get("split_counts", {})
                entry = {
                    "status": cell.get("status"),
                    "val_auc": cell.get("val_auc"),
                    "test_auc": cell.get("test_auc"),
                    "test_auc_ci": cell.get("test_auc_ci"),
                    "per_height": cell.get("per_height"),
                    "n_train": split_counts.get("train", {}).get("n"),
                    "n_val": split_counts.get("val", {}).get("n"),
                    "n_test": split_counts.get("test", {}).get("n"),
                    "positive_n_train": split_counts.get("train", {}).get("positive_n"),
                    "positive_n_val": split_counts.get("val", {}).get("positive_n"),
                    "positive_n_test": split_counts.get("test", {}).get("positive_n"),
                    "delta_over_best_b0": _delta(cell.get("test_auc"), b0_auc),
                    "b0_reference": {
                        "feature_set": b0_feature,
                        "test_auc": b0_auc,
                    },
                }

                if probe_type == "logistic":
                    entry["best_c"] = cell.get("best_c")
                if probe_type == "diffmeans":
                    entry["threshold"] = cell.get("threshold")
                    entry["val_balanced_accuracy"] = cell.get("val_balanced_accuracy")
                    entry["test_balanced_accuracy"] = cell.get("test_balanced_accuracy")

                probe_out[split] = entry

    return payload


def aggregate_transfer(*, transfer_files: list[Path]) -> dict[str, Any]:
    if not transfer_files:
        raise ValueError("no transfer files provided")

    first = read_json(transfer_files[0])
    model_key = first["model_key"]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "source_transfer_files": [str(path) for path in transfer_files],
        "results": {},
    }

    for transfer_path in transfer_files:
        report = read_json(transfer_path)
        split = report["split_family"]
        if report["model_key"] != model_key:
            raise ValueError(f"model mismatch in {transfer_path}: {report['model_key']} != {model_key}")

        for transfer_key, by_layer in report["results"].items():
            transfer_out = payload["results"].setdefault(transfer_key, {})
            for layer, cell in by_layer.items():
                layer_out = transfer_out.setdefault(layer, {})
                if split in layer_out:
                    raise ValueError(
                        f"duplicate transfer entry transfer={transfer_key} layer={layer} split={split}"
                    )
                layer_out[split] = {
                    "status": cell.get("status"),
                    "source_val_auc": cell.get("source_val_auc"),
                    "source_test_auc": cell.get("source_test_auc"),
                    "source_test_auc_ci": cell.get("source_test_auc_ci"),
                    "target_test_auc": cell.get("target_test_auc"),
                    "target_test_auc_ci": cell.get("target_test_auc_ci"),
                    "target_per_height": cell.get("target_per_height"),
                    "source_split_counts": cell.get("source_split_counts"),
                    "target_split_counts": cell.get("target_split_counts"),
                }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--model-tag", default="4b", help="filename tag in docs/raw_probe_<tag>_*.json")
    parser.add_argument(
        "--b0-summary",
        type=Path,
        default=Path("docs/stage2_b0_summary_4b.json"),
    )
    parser.add_argument(
        "--probe-files",
        nargs="*",
        type=Path,
        default=None,
        help="optional explicit probe JSON files; if omitted, auto-discover from --docs-dir/--model-tag",
    )
    parser.add_argument(
        "--transfer-files",
        nargs="*",
        type=Path,
        default=None,
        help="optional explicit transfer JSON files; if omitted, auto-discover from --docs-dir/--model-tag",
    )
    parser.add_argument(
        "--probe-out",
        type=Path,
        default=Path("results/stage2/probe_auc.json"),
    )
    parser.add_argument(
        "--transfer-out",
        type=Path,
        default=Path("results/stage2/probe_transfer.json"),
    )
    args = parser.parse_args()

    probe_files = args.probe_files or discover_probe_jsons(args.docs_dir, args.model_tag)
    transfer_files = args.transfer_files or discover_transfer_jsons(args.docs_dir, args.model_tag)
    if not probe_files:
        raise FileNotFoundError("no raw probe JSON files found")
    if not transfer_files:
        raise FileNotFoundError("no raw transfer JSON files found")

    b0_summary = read_json(args.b0_summary)
    probe_payload = aggregate_probe_auc(probe_files=probe_files, b0_summary=b0_summary)
    transfer_payload = aggregate_transfer(transfer_files=transfer_files)

    write_json(args.probe_out, probe_payload)
    write_json(args.transfer_out, transfer_payload)
    print(args.probe_out)
    print(args.transfer_out)


if __name__ == "__main__":
    main()
