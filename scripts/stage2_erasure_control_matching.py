#!/usr/bin/env python3
"""Erasure control-matching follow-up.

Reviewer-insurance test for the multi-layer correctness-direction erasure result.
It keeps the landed erasure intervention family fixed, but adds two diagnostics:

1. richer hook telemetry for the pre-erasure projection deltas, including
   within-forward positional variance rather than only mean absolute delta; and
2. a height/difficulty control direction trained per layer as a deliberately
   between-run direction, plus dose-response controls for orthogonal/Gaussian
   erasures.

See docs/causal_handle_directions.md, "Erasure control matching".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import (  # noqa: E402
    generate_one,
    json_default,
    package_version,
    serializable_direction_summary,
    torch_dtype,
)
from scripts.stage2_subspace_erasure import summarize_erasure_rows  # noqa: E402
from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    _class_counts,
    _has_two_classes,
    _make_logistic_pipeline,
    _safe_auc,
    load_probe_dataset,
    read_split_assignments,
    split_indices_from_assignments,
)
from src.stage2_steering import (  # noqa: E402
    make_gaussian_unit_direction,
    make_orthogonal_unit_direction,
    parse_int_list,
    score_reply,
    select_balanced_stage1_rows,
    train_raw_probe_direction,
)


@dataclass(frozen=True)
class ControlCondition:
    label: str
    vector_kind: str | None
    scale: float


def scale_label(scale: float) -> str:
    return f"s{scale:g}".replace(".", "p").replace("-", "m")


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one scale")
    if any(scale <= 0 for scale in parsed):
        raise ValueError("all scales must be positive")
    return parsed


def parse_condition_kinds(value: str) -> list[str]:
    allowed = {"baseline", "erase_raw", "erase_height", "erase_orthogonal", "erase_gaussian"}
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    if "baseline" not in parsed:
        raise ValueError("baseline condition is required for paired interpretation")
    return parsed


def make_condition_plan(
    *,
    condition_kinds: list[str],
    control_scales: list[float],
    raw_scale: float,
    height_scale: float,
) -> list[ControlCondition]:
    plan = [ControlCondition("baseline", None, 0.0)]
    if "erase_raw" in condition_kinds:
        plan.append(ControlCondition(f"erase_raw_{scale_label(raw_scale)}", "raw", raw_scale))
    if "erase_height" in condition_kinds:
        plan.append(ControlCondition(f"erase_height_{scale_label(height_scale)}", "height", height_scale))
    for kind, vector_kind in (("erase_orthogonal", "orthogonal"), ("erase_gaussian", "gaussian")):
        if kind not in condition_kinds:
            continue
        for scale in control_scales:
            plan.append(ControlCondition(f"{kind}_{scale_label(scale)}", vector_kind, scale))
    return plan


def _positive_scores(model: Any, x: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    logreg = model[-1]
    positive_index = int(np.where(logreg.classes_ == 1)[0][0])
    return [float(score) for score in model.predict_proba(x[indices])[:, positive_index]]


def projection_stats_for_indices(x: np.ndarray, unit: np.ndarray, indices: list[int]) -> dict[str, float]:
    if not indices:
        raise ValueError("cannot compute projection stats on empty index set")
    projections = np.asarray(x[indices].astype(np.float64) @ np.asarray(unit, dtype=np.float64))
    std = float(projections.std(ddof=0))
    if std == 0.0:
        raise ValueError("projection standard deviation is zero")
    return {
        "projection_mean": float(projections.mean()),
        "projection_std": std,
        "projection_min": float(projections.min()),
        "projection_max": float(projections.max()),
        "n": len(indices),
        "scope": "train_split",
    }


def train_height_control_direction(
    *,
    activation_path: Path,
    sidecar_path: Path,
    splits_path: Path,
    source_file: str,
    split_family: str,
    positive_height_min: int,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
) -> dict[str, Any]:
    """Train a binary height/difficulty direction in raw activation space."""
    dataset = load_probe_dataset(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        drop_parse_failed=True,
    )
    x = dataset["x"]
    sidecar = dataset["sidecar"]
    labels = [1 if int(row.get("height", 0)) >= positive_height_min else 0 for row in sidecar]
    split_assignments = read_split_assignments(splits_path)
    splits = split_indices_from_assignments(
        sidecar,
        assignments=split_assignments,
        source_file=source_file,
        split_field=f"{split_family}_split",
    )
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    split_counts = {split: _class_counts(labels, indices) for split, indices in splits.items()}
    if not _has_two_classes(labels, train_indices):
        raise ValueError(f"height control training split has one class: {split_counts['train']}")
    if not _has_two_classes(labels, val_indices):
        raise ValueError(f"height control validation split has one class: {split_counts['val']}")
    if not _has_two_classes(labels, test_indices):
        raise ValueError(f"height control test split has one class: {split_counts['test']}")

    best: dict[str, Any] | None = None
    for c_value in c_values:
        model = _make_logistic_pipeline(x, c_value=c_value, max_iter=max_iter, solver=solver)
        model.fit(x[train_indices], [labels[idx] for idx in train_indices])
        val_scores = _positive_scores(model, x, val_indices)
        val_auc = _safe_auc([labels[idx] for idx in val_indices], val_scores)
        rank_auc = val_auc if val_auc is not None else -math.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {"model": model, "c": float(c_value), "val_auc": val_auc, "rank_auc": rank_auc}
    assert best is not None

    model = best["model"]
    scaler = model[0]
    logreg = model[-1]
    if not np.array_equal(logreg.classes_, np.array([0, 1])):
        raise ValueError(f"expected logistic classes [0, 1], got {logreg.classes_.tolist()}")
    coef_std = logreg.coef_[0].astype(np.float64)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)
    raw_coef = coef_std / scaler_scale
    raw_norm = float(np.linalg.norm(raw_coef))
    if raw_norm == 0.0:
        raise ValueError("height control coefficient has zero norm")
    unit_direction = raw_coef / raw_norm
    train_projection = x[train_indices].astype(np.float64) @ unit_direction
    projection_std = float(train_projection.std(ddof=0))
    if projection_std == 0.0:
        raise ValueError("height control train projection has zero standard deviation")
    test_scores = _positive_scores(model, x, test_indices)
    test_auc = _safe_auc([labels[idx] for idx in test_indices], test_scores)

    return {
        "unit_direction": unit_direction.astype(np.float32),
        "raw_coef": raw_coef.astype(np.float32),
        "coef_std": coef_std.astype(np.float32),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "label": f"height_ge_{positive_height_min}",
        "positive_height_min": int(positive_height_min),
        "best_c": best["c"],
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "val_auc": best["val_auc"],
        "test_auc": test_auc,
        "split_counts": split_counts,
        "input_rows": dataset["input_rows"],
        "kept_rows": dataset["kept_rows"],
        "d_model": dataset["d_model"],
        "raw_coef_norm": raw_norm,
        "train_projection_mean": float(train_projection.mean()),
        "train_projection_std": projection_std,
        "train_projection_min": float(train_projection.min()),
        "train_projection_max": float(train_projection.max()),
    }


def make_control_matching_hook(
    *,
    vector: np.ndarray,
    projection_mean: float,
    projection_std: float,
    scale: float,
) -> tuple[Any, dict[str, Any]]:
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    state: dict[str, Any] = {
        "calls": 0,
        "prompt_calls": 0,
        "decode_calls": 0,
        "positions": 0,
        "sum_delta_sd": 0.0,
        "sum_delta_sd2": 0.0,
        "sum_abs_delta_sd": 0.0,
        "within_call_var_sd2_sum": 0.0,
        "within_call_var_sd2_weighted_sum": 0.0,
        "within_call_count": 0,
        "max_abs_delta_sd": 0.0,
        "scale": float(scale),
    }

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        key = (str(act.device), act.dtype)
        unit = cache.get(key)
        if unit is None:
            unit = torch.as_tensor(vector, device=act.device, dtype=torch.float32)
            cache[key] = unit
        projection = act.float() @ unit
        delta = projection - float(projection_mean)
        delta_sd = (delta / float(projection_std)).detach().float().reshape(-1)
        n = int(delta_sd.numel())
        if n:
            sum_delta = float(delta_sd.sum().item())
            sum_delta2 = float((delta_sd * delta_sd).sum().item())
            sum_abs = float(delta_sd.abs().sum().item())
            mean = sum_delta / n
            var = max(0.0, sum_delta2 / n - mean * mean)
            state["calls"] += 1
            state["positions"] += n
            state["sum_delta_sd"] += sum_delta
            state["sum_delta_sd2"] += sum_delta2
            state["sum_abs_delta_sd"] += sum_abs
            state["within_call_var_sd2_sum"] += var
            state["within_call_var_sd2_weighted_sum"] += var * n
            state["within_call_count"] += 1
            state["max_abs_delta_sd"] = max(state["max_abs_delta_sd"], float(delta_sd.abs().max().item()))
            if n > 1:
                state["prompt_calls"] += 1
            else:
                state["decode_calls"] += 1
        act -= (float(scale) * delta.unsqueeze(-1) * unit).to(act.dtype)
        return act

    return hook_fn, state


def summarize_hook_state(state: dict[str, Any]) -> dict[str, Any]:
    positions = int(state["positions"])
    calls = int(state["calls"])
    if positions:
        mean = state["sum_delta_sd"] / positions
        variance = max(0.0, state["sum_delta_sd2"] / positions - mean * mean)
    else:
        mean = None
        variance = None
    return {
        "calls": calls,
        "prompt_calls": int(state["prompt_calls"]),
        "decode_calls": int(state["decode_calls"]),
        "positions": positions,
        "mean_delta_sd": mean,
        "std_delta_sd": math.sqrt(variance) if variance is not None else None,
        "mean_abs_delta_sd": (state["sum_abs_delta_sd"] / positions if positions else None),
        "mean_within_call_var_sd2": (
            state["within_call_var_sd2_sum"] / state["within_call_count"]
            if state["within_call_count"]
            else None
        ),
        "position_weighted_within_call_var_sd2": (
            state["within_call_var_sd2_weighted_sum"] / positions if positions else None
        ),
        "max_abs_delta_sd": float(state["max_abs_delta_sd"]),
        "scale": float(state["scale"]),
        "sum_delta_sd": float(state["sum_delta_sd"]),
        "sum_delta_sd2": float(state["sum_delta_sd2"]),
        "sum_abs_delta_sd": float(state["sum_abs_delta_sd"]),
        "within_call_var_sd2_weighted_sum": float(state["within_call_var_sd2_weighted_sum"]),
        "within_call_var_sd2_sum": float(state["within_call_var_sd2_sum"]),
        "within_call_count": int(state["within_call_count"]),
    }


def summarize_hook_states(states: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {f"L{layer}": summarize_hook_state(state) for layer, state in sorted(states.items())}


def summarize_projection_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {
        "generations": 0,
        "calls": 0,
        "prompt_calls": 0,
        "decode_calls": 0,
        "positions": 0,
        "sum_delta_sd": 0.0,
        "sum_delta_sd2": 0.0,
        "sum_abs_delta_sd": 0.0,
        "within_call_var_sd2_weighted_sum": 0.0,
        "within_call_var_sd2_sum": 0.0,
        "within_call_count": 0,
        "max_abs_delta_sd": 0.0,
    }))
    for row in rows:
        condition = row["condition"]
        for layer, summary in row.get("hook_summary", {}).items():
            agg = aggregates[condition][layer]
            agg["generations"] += 1
            for key in (
                "calls",
                "prompt_calls",
                "decode_calls",
                "positions",
                "within_call_count",
            ):
                agg[key] += int(summary.get(key, 0) or 0)
            for key in (
                "sum_delta_sd",
                "sum_delta_sd2",
                "sum_abs_delta_sd",
                "within_call_var_sd2_weighted_sum",
                "within_call_var_sd2_sum",
            ):
                agg[key] += float(summary.get(key, 0.0) or 0.0)
            agg["max_abs_delta_sd"] = max(agg["max_abs_delta_sd"], float(summary.get("max_abs_delta_sd", 0.0) or 0.0))

    out: dict[str, Any] = {}
    for condition, by_layer in sorted(aggregates.items()):
        out[condition] = {}
        for layer, agg in sorted(by_layer.items()):
            positions = agg["positions"]
            if positions:
                mean = agg["sum_delta_sd"] / positions
                variance = max(0.0, agg["sum_delta_sd2"] / positions - mean * mean)
            else:
                mean = None
                variance = None
            out[condition][layer] = {
                "generations": agg["generations"],
                "calls": agg["calls"],
                "prompt_calls": agg["prompt_calls"],
                "decode_calls": agg["decode_calls"],
                "positions": positions,
                "mean_delta_sd": mean,
                "std_delta_sd": math.sqrt(variance) if variance is not None else None,
                "mean_abs_delta_sd": agg["sum_abs_delta_sd"] / positions if positions else None,
                "position_weighted_within_call_var_sd2": (
                    agg["within_call_var_sd2_weighted_sum"] / positions if positions else None
                ),
                "mean_within_call_var_sd2": (
                    agg["within_call_var_sd2_sum"] / agg["within_call_count"]
                    if agg["within_call_count"]
                    else None
                ),
                "max_abs_delta_sd": agg["max_abs_delta_sd"],
            }
    return out


def build_layer_directions(
    *,
    layers: list[int],
    activation_dir: Path,
    model_key: str,
    task: str,
    splits_path: Path,
    source_file: str,
    split_family: str,
    probe_seed: int,
    orthogonal_seed: int,
    gaussian_seed: int,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    height_control_min: int,
) -> dict[int, dict[str, Any]]:
    split_assignments = read_split_assignments(splits_path)
    by_layer: dict[int, dict[str, Any]] = {}
    for layer in layers:
        prefix = activation_dir / f"{model_key}_{task}_L{layer}"
        activation_path = prefix.with_suffix(".safetensors")
        sidecar_path = prefix.with_suffix(".example_ids.jsonl")
        direction = train_raw_probe_direction(
            activation_path=activation_path,
            sidecar_path=sidecar_path,
            splits_path=splits_path,
            source_file=source_file,
            split_family=split_family,
            seed=probe_seed,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
        )
        height_direction = train_height_control_direction(
            activation_path=activation_path,
            sidecar_path=sidecar_path,
            splits_path=splits_path,
            source_file=source_file,
            split_family=split_family,
            positive_height_min=height_control_min,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
        )
        orthogonal = make_orthogonal_unit_direction(direction["unit_direction"], seed=orthogonal_seed + layer)
        gaussian = make_gaussian_unit_direction(direction["unit_direction"], seed=gaussian_seed + layer)
        dataset = load_probe_dataset(
            activation_path=activation_path,
            sidecar_path=sidecar_path,
            drop_parse_failed=True,
        )
        splits = split_indices_from_assignments(
            dataset["sidecar"],
            assignments=split_assignments,
            source_file=source_file,
            split_field=f"{split_family}_split",
        )
        train_indices = splits["train"]
        vectors = {
            "raw": direction["unit_direction"],
            "height": height_direction["unit_direction"],
            "orthogonal": orthogonal,
            "gaussian": gaussian,
        }
        stats = {
            "raw": {
                "projection_mean": float(direction["train_projection_mean"]),
                "projection_std": float(direction["train_projection_std"]),
                "projection_min": float(direction["train_projection_min"]),
                "projection_max": float(direction["train_projection_max"]),
                "scope": "train_split",
            },
            "height": {
                "projection_mean": float(height_direction["train_projection_mean"]),
                "projection_std": float(height_direction["train_projection_std"]),
                "projection_min": float(height_direction["train_projection_min"]),
                "projection_max": float(height_direction["train_projection_max"]),
                "scope": "train_split",
            },
            "orthogonal": projection_stats_for_indices(dataset["x"], orthogonal, train_indices),
            "gaussian": projection_stats_for_indices(dataset["x"], gaussian, train_indices),
        }
        raw_unit = np.asarray(vectors["raw"], dtype=np.float64)
        by_layer[layer] = {
            "direction": direction,
            "height_direction": height_direction,
            "vectors": vectors,
            "stats": stats,
            "direction_cosines_vs_raw": {
                kind: float(np.asarray(vector, dtype=np.float64) @ raw_unit)
                for kind, vector in vectors.items()
                if kind != "raw"
            },
        }
        print(
            f"L{layer}: raw_auc={direction['test_auc']:.4f} raw_std={direction['train_projection_std']:.4f} "
            f"height_auc={height_direction['test_auc']:.4f} height_std={height_direction['train_projection_std']:.4f} "
            f"cos(height,raw)={by_layer[layer]['direction_cosines_vs_raw']['height']:.4f}",
            flush=True,
        )
    return by_layer


def save_layer_direction_artifact(path: Path, by_layer: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for layer, entry in by_layer.items():
        for kind, vector in entry["vectors"].items():
            arrays[f"L{layer}_{kind}_unit"] = np.asarray(vector, dtype=np.float32)
        for kind, stats in entry["stats"].items():
            arrays[f"L{layer}_{kind}_projection_mean"] = np.array(stats["projection_mean"], dtype=np.float32)
            arrays[f"L{layer}_{kind}_projection_std"] = np.array(stats["projection_std"], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="15,30,40,45,53")
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=4)
    parser.add_argument("--samples-per-row", type=int, default=4)
    parser.add_argument("--row-shard", default=None, help="Optional 'i/n' slice of selected rows.")
    parser.add_argument("--selection-seed", type=int, default=20260427)
    parser.add_argument("--probe-seed", type=int, default=20260472)
    parser.add_argument("--orthogonal-seed", type=int, default=20260545)
    parser.add_argument("--gaussian-seed", type=int, default=20260604)
    parser.add_argument("--height-control-min", type=int, default=4)
    parser.add_argument("--conditions", default="baseline,erase_raw,erase_height,erase_orthogonal,erase_gaussian")
    parser.add_argument("--control-scales", default="0.25,0.5,1.0")
    parser.add_argument("--raw-scale", type=float, default=1.0)
    parser.add_argument("--height-scale", type=float, default=1.0)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/erasure_control_matching_27b_property.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/erasure/erasure_control_matching_27b_property_directions.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/erasure_control_matching_27b_property.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    if args.samples_per_row < 1:
        raise ValueError("--samples-per-row must be >= 1")
    if args.samples_per_row > 1 and not args.do_sample:
        raise ValueError("--samples-per-row > 1 requires --do-sample")
    if args.raw_scale <= 0 or args.height_scale <= 0:
        raise ValueError("--raw-scale/--height-scale must be positive")

    layers = parse_int_list(args.layers)
    condition_plan = make_condition_plan(
        condition_kinds=parse_condition_kinds(args.conditions),
        control_scales=parse_float_list(args.control_scales),
        raw_scale=args.raw_scale,
        height_scale=args.height_scale,
    )
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)

    print("Erasure control-matching follow-up", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layers={layers}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"samples_per_row={args.samples_per_row}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)

    selected_rows, selection_summary = select_balanced_stage1_rows(
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=parse_int_list(args.heights),
        per_height_label=args.per_height_label,
        seed=args.selection_seed,
        target_split="test",
    )
    if args.row_shard:
        shard_index, shard_count = (int(part) for part in args.row_shard.split("/"))
        if not (0 <= shard_index < shard_count):
            raise ValueError(f"invalid --row-shard {args.row_shard!r}")
        selected_rows = selected_rows[shard_index::shard_count]
        selection_summary["row_shard"] = args.row_shard
        selection_summary["shard_rows"] = len(selected_rows)
    total_generations = len(selected_rows) * len(condition_plan) * args.samples_per_row
    print(
        f"selected_rows={len(selected_rows)} total_generations={total_generations} "
        f"selection={selection_summary}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selection": selection_summary,
                    "conditions": [condition.__dict__ for condition in condition_plan],
                    "layers": layers,
                    "total_generations": total_generations,
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            ),
            flush=True,
        )
        return 0

    by_layer = build_layer_directions(
        layers=layers,
        activation_dir=args.activation_dir,
        model_key=args.model_key,
        task=args.task,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        probe_seed=args.probe_seed,
        orthogonal_seed=args.orthogonal_seed,
        gaussian_seed=args.gaussian_seed,
        c_values=tuple(float(part) for part in args.c_values.split(",")),
        max_iter=args.max_iter,
        solver=args.solver,
        height_control_min=args.height_control_min,
    )
    save_layer_direction_artifact(args.direction_output, by_layer)

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(
        f"scorer_preflight: strong={scorer_preflight['is_correct_strong']} "
        f"parse_failed={scorer_preflight['parse_failed']}",
        flush=True,
    )

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    hook_names = validate_hooks(model, layers)
    hook_name_by_layer = dict(zip(layers, hook_names))
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    print(f"using_hooks={hook_name_by_layer}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            prompt_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if len(token_ids) > args.n_ctx:
                raise ValueError(
                    f"row {stage1_row['row_index']} prompt exceeds n_ctx={args.n_ctx}: {len(token_ids)}"
                )
            print(
                f"row {row_idx}/{len(selected_rows)} source_row={stage1_row['row_index']} "
                f"h={stage1_row.get('height')} original_strong={stage1_row.get('is_correct_strong')} "
                f"prompt_tokens={len(token_ids)}",
                flush=True,
            )
            for condition_index, condition in enumerate(condition_plan):
                for sample_index in range(args.samples_per_row):
                    if args.do_sample:
                        torch.manual_seed(
                            args.selection_seed
                            + int(stage1_row["row_index"]) * 10007
                            + condition_index * 101
                            + sample_index
                        )
                    hook_states: dict[int, dict[str, Any]] = {}
                    fwd_hooks = []
                    if condition.vector_kind is not None:
                        for layer in layers:
                            entry = by_layer[layer]
                            stats = entry["stats"][condition.vector_kind]
                            hook_fn, hook_state = make_control_matching_hook(
                                vector=entry["vectors"][condition.vector_kind],
                                projection_mean=stats["projection_mean"],
                                projection_std=stats["projection_std"],
                                scale=condition.scale,
                            )
                            hook_states[layer] = hook_state
                            fwd_hooks.append((hook_name_by_layer[layer], hook_fn))
                    with model.hooks(fwd_hooks=fwd_hooks):
                        new_ids, reply = generate_one(
                            model=model,
                            token_ids=token_ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype,
                        )
                    score = score_reply(stage1_row, reply)
                    output_row = {
                        "schema_version": 1,
                        "source_file": source_file,
                        "source_row_index": int(stage1_row["row_index"]),
                        "example_id": stage1_row.get("example_id"),
                        "task": stage1_row.get("task"),
                        "height": stage1_row.get("height"),
                        "model": args.model,
                        "original_is_correct_strong": bool(stage1_row.get("is_correct_strong")),
                        "original_is_correct_weak": bool(stage1_row.get("is_correct_weak")),
                        "original_parse_failed": bool(stage1_row.get("parse_failed")),
                        "condition": condition.label,
                        "condition_vector_kind": condition.vector_kind,
                        "condition_scale": condition.scale,
                        "sample_index": sample_index,
                        "method": "erasure_control_matching",
                        "target_variable": "free_form_correctness",
                        "representation_type": "raw_direction_and_height_control",
                        "erasure_kind": condition.vector_kind,
                        "erasure_layers": layers,
                        "prompt_token_count": len(token_ids),
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        "hook_summary": summarize_hook_states(hook_states),
                        **score,
                    }
                    rows.append(output_row)
                    fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                    print(
                        f"  {condition.label}[{sample_index}]: strong={output_row['is_correct_strong']} "
                        f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                        f"new_tokens={len(new_ids)}",
                        flush=True,
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    erasure_summary = summarize_erasure_rows(rows)
    projection_telemetry = summarize_projection_telemetry(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_erasure_control_matching.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "free_form_correctness",
        "split": args.split_family,
        "site_or_layer": ",".join(f"L{layer}" for layer in layers),
        "method": "erasure_control_matching",
        "representation_type": "raw_direction_and_height_control",
        "layers": layers,
        "hook_names": hook_name_by_layer,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "direction_output": str(args.direction_output),
        "out_jsonl": str(args.out_jsonl),
        "probe_directions": {
            f"L{layer}": serializable_direction_summary(entry["direction"])
            for layer, entry in by_layer.items()
        },
        "height_control_directions": {
            f"L{layer}": serializable_direction_summary(entry["height_direction"])
            for layer, entry in by_layer.items()
        },
        "projection_stats": {
            f"L{layer}": entry["stats"] for layer, entry in by_layer.items()
        },
        "direction_cosines_vs_raw": {
            f"L{layer}": entry["direction_cosines_vs_raw"] for layer, entry in by_layer.items()
        },
        "selection": selection_summary,
        "generation": {
            "conditions": [condition.__dict__ for condition in condition_plan],
            "samples_per_row": args.samples_per_row,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "summary": erasure_summary,
        "projection_telemetry": projection_telemetry,
        "baseline_metrics": erasure_summary["by_condition"].get("baseline"),
        "intervention_metrics": {
            condition: metrics
            for condition, metrics in erasure_summary["by_condition"].items()
            if condition != "baseline"
        },
        "paired_flips": erasure_summary["paired_vs_baseline"],
        "n": len(rows),
        "controls": [
            "regenerated_baseline",
            "raw_direction",
            "height_or_difficulty_between_run_direction",
            "orthogonal_direction_dose_response",
            "matched_gaussian_noise_dose_response",
        ],
        "pre_registered_decision_rule": (
            "If the raw direction's within-run positional projection variance is far below "
            "orthogonal/Gaussian controls, the constant-offset account is live and the "
            "between-run height-control outcome governs wording. If variance is comparable, "
            "or if the low-variance height control is behaviorally destructive while raw erasure "
            "remains null, the existing potent-machinery interpretation strengthens."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
