#!/usr/bin/env python3
"""Build Phase B cross-model comparison JSON (4B vs 27B)."""

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


def _safe_delta(test_auc: float | None, b0_auc: float | None) -> float | None:
    if test_auc is None or b0_auc is None:
        return None
    return float(test_auc - b0_auc)


def _best_4b_layer_for_task(probe_auc_4b: dict[str, Any], task: str, split: str) -> tuple[str | None, dict[str, Any] | None]:
    best_layer = None
    best_cell = None
    best_auc = -1.0
    for layer, by_probe in probe_auc_4b["results"][task].items():
        cell = by_probe.get("logistic", {}).get(split)
        if not cell or cell.get("status") != "ok" or cell.get("test_auc") is None:
            continue
        auc = float(cell["test_auc"])
        if auc > best_auc:
            best_auc = auc
            best_layer = layer
            best_cell = cell
    return best_layer, best_cell


def _best_27b_layer_for_task(raw_probe_27b: dict[str, Any], task: str) -> tuple[str | None, dict[str, Any] | None]:
    best_layer = None
    best_cell = None
    best_auc = -1.0
    for layer, cell in raw_probe_27b["results"][task].items():
        if cell.get("status") != "ok" or cell.get("test_auc") is None:
            continue
        auc = float(cell["test_auc"])
        if auc > best_auc:
            best_auc = auc
            best_layer = layer
            best_cell = cell
    return best_layer, best_cell


def _depth_fraction(layer_key: str | None, total_layers: int) -> float | None:
    if not layer_key:
        return None
    layer_num = int(layer_key.lstrip("L"))
    return float(layer_num / total_layers)


def _normalize_task_key(task: str) -> str:
    return f"gemma3-27b__{task}"


def _b0_auc(b0_summary: dict[str, Any], model_task_key: str, split: str) -> float | None:
    cell = b0_summary.get("best_pre_output_baseline", {}).get(model_task_key, {}).get(split)
    if cell is None:
        return None
    return cell.get("test_auc")


def _per_height_profile(cell: dict[str, Any] | None) -> dict[str, Any] | None:
    if not cell:
        return None
    return cell.get("per_height")


def build_cross_model_comparison(
    *,
    probe_auc_4b: dict[str, Any],
    raw_27b_s1: dict[str, Any],
    raw_27b_s3: dict[str, Any],
    b0_4b: dict[str, Any],
    b0_27b_s1: dict[str, Any],
    b0_27b_s3: dict[str, Any],
) -> dict[str, Any]:
    tasks = ["infer_property", "infer_subtype"]
    model_rows: dict[str, Any] = {}

    # 4B row
    four_b = {
        "model_key": "gemma3_4b",
        "model_name": "gemma3-4b",
        "total_layers": 34,
        "splits": {},
        "strategy_frac_annotation": {
            "available": False,
            "reason": "output_strategy field not present in current shipped JSONLs",
            "infer_subtype_h3_h4": {"h3": None, "h4": None},
        },
    }
    best_depth_candidates = []
    for split in ("s1", "s3"):
        split_block = {}
        for task in tasks:
            layer, cell = _best_4b_layer_for_task(probe_auc_4b, task, split)
            if layer:
                best_depth_candidates.append(int(layer.lstrip("L")))
            b0_key = f"gemma3-4b__{task}"
            b0_auc = _b0_auc(b0_4b, b0_key, split)
            split_block[task] = {
                "best_layer": layer,
                "best_layer_depth_fraction": _depth_fraction(layer, 34),
                "test_auc": None if cell is None else cell.get("test_auc"),
                "delta_over_best_b0": _safe_delta(None if cell is None else cell.get("test_auc"), b0_auc),
                "b0_test_auc": b0_auc,
                "per_height_auc_profile": _per_height_profile(cell),
            }
        four_b["splits"][split] = split_block

    if best_depth_candidates:
        global_best_layer = f"L{max(best_depth_candidates)}"
    else:
        global_best_layer = None
    four_b["best_layer_global"] = global_best_layer
    four_b["best_layer_global_depth_fraction"] = _depth_fraction(global_best_layer, 34)
    model_rows["gemma3_4b"] = four_b

    # 27B row
    twenty_seven_b = {
        "model_key": "gemma3_27b",
        "model_name": "gemma3-27b",
        "total_layers": 62,
        "splits": {},
        "strategy_frac_annotation": {
            "available": False,
            "reason": "output_strategy field not present in current shipped JSONLs",
            "infer_subtype_h3_h4": {"h3": None, "h4": None},
        },
    }
    best_depth_candidates = []
    split_to_raw = {"s1": raw_27b_s1, "s3": raw_27b_s3}
    split_to_b0 = {"s1": b0_27b_s1, "s3": b0_27b_s3}
    for split in ("s1", "s3"):
        split_block = {}
        raw = split_to_raw[split]
        b0 = split_to_b0[split]
        for task in tasks:
            layer, cell = _best_27b_layer_for_task(raw, task)
            if layer:
                best_depth_candidates.append(int(layer.lstrip("L")))
            b0_key = _normalize_task_key(task)
            b0_auc = _b0_auc(b0, b0_key, split)
            split_block[task] = {
                "best_layer": layer,
                "best_layer_depth_fraction": _depth_fraction(layer, 62),
                "test_auc": None if cell is None else cell.get("test_auc"),
                "delta_over_best_b0": _safe_delta(None if cell is None else cell.get("test_auc"), b0_auc),
                "b0_test_auc": b0_auc,
                "per_height_auc_profile": _per_height_profile(cell),
            }
        twenty_seven_b["splits"][split] = split_block

    if best_depth_candidates:
        global_best_layer = f"L{max(best_depth_candidates)}"
    else:
        global_best_layer = None
    twenty_seven_b["best_layer_global"] = global_best_layer
    twenty_seven_b["best_layer_global_depth_fraction"] = _depth_fraction(global_best_layer, 62)
    model_rows["gemma3_27b"] = twenty_seven_b

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "probe_auc_4b": "results/stage2/probe_auc.json",
            "raw_probe_27b_s1": "docs/raw_probe_27b_s1.json",
            "raw_probe_27b_s3": "docs/raw_probe_27b_s3_target_symbol.json",
            "b0_4b": "docs/stage2_b0_summary_4b.json",
            "b0_27b_s1": "docs/stage2_b0_summary_27b_s1.json",
            "b0_27b_s3": "docs/stage2_b0_summary_27b_s3.json",
        },
        "models": model_rows,
        "notes": [
            "S2 omitted from cross-model rows: current S2 split is non-evaluable in this dataset snapshot.",
            "strategy_frac annotation not computed because output_strategy is absent in shipped JSONLs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-auc-4b", type=Path, default=Path("results/stage2/probe_auc.json"))
    parser.add_argument("--raw-27b-s1", type=Path, default=Path("docs/raw_probe_27b_s1.json"))
    parser.add_argument("--raw-27b-s3", type=Path, default=Path("docs/raw_probe_27b_s3_target_symbol.json"))
    parser.add_argument("--b0-4b", type=Path, default=Path("docs/stage2_b0_summary_4b.json"))
    parser.add_argument("--b0-27b-s1", type=Path, default=Path("docs/stage2_b0_summary_27b_s1.json"))
    parser.add_argument("--b0-27b-s3", type=Path, default=Path("docs/stage2_b0_summary_27b_s3.json"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2/probe_comparison.json"))
    args = parser.parse_args()

    payload = build_cross_model_comparison(
        probe_auc_4b=read_json(args.probe_auc_4b),
        raw_27b_s1=read_json(args.raw_27b_s1),
        raw_27b_s3=read_json(args.raw_27b_s3),
        b0_4b=read_json(args.b0_4b),
        b0_27b_s1=read_json(args.b0_27b_s1),
        b0_27b_s3=read_json(args.b0_27b_s3),
    )
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
