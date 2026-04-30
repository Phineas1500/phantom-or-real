"""Helpers for Stage 2 raw-direction steering pilots."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .stage2_probes import (
    DEFAULT_C_VALUES,
    _class_counts,
    _has_two_classes,
    _make_logistic_pipeline,
    _safe_auc,
    load_probe_dataset,
    read_split_assignments,
    split_indices_from_assignments,
)


@dataclass(frozen=True)
class SteeringCondition:
    label: str
    direction_kind: str | None
    strength_sd: float


def parse_float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("expected at least one float")
    return parsed


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def parse_condition_kinds(value: str) -> list[str]:
    allowed = {"baseline", "raw", "orthogonal"}
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition kind")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    return parsed


def strength_label(strength: float) -> str:
    sign = "pos" if strength >= 0 else "neg"
    magnitude = f"{abs(strength):g}".replace(".", "p")
    return f"{sign}{magnitude}sd"


def make_condition_plan(
    *,
    condition_kinds: Iterable[str],
    strengths: Iterable[float],
) -> list[SteeringCondition]:
    kinds = list(condition_kinds)
    strength_values = list(strengths)
    plan: list[SteeringCondition] = []
    if "baseline" in kinds:
        plan.append(SteeringCondition("baseline", None, 0.0))
    for kind in ("raw", "orthogonal"):
        if kind not in kinds:
            continue
        for strength in strength_values:
            if strength == 0:
                continue
            plan.append(
                SteeringCondition(
                    label=f"{kind}_{strength_label(strength)}",
                    direction_kind=kind,
                    strength_sd=float(strength),
                )
            )
    if not plan:
        raise ValueError("condition plan is empty")
    return plan


def _positive_scores(model: Any, x: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    logreg = model[-1]
    positive_index = int(np.where(logreg.classes_ == 1)[0][0])
    return [float(score) for score in model.predict_proba(x[indices])[:, positive_index]]


def _train_feature_densities(x: Any, train_indices: list[int], feature_ids: np.ndarray) -> np.ndarray:
    from scipy import sparse

    if feature_ids.size == 0:
        return np.asarray([], dtype=np.float64)
    if not train_indices:
        raise ValueError("train_indices is empty")
    train_x = x[train_indices]
    if sparse.issparse(train_x):
        columns = train_x[:, feature_ids].tocsc(copy=False)
        nonzero = np.diff(columns.indptr)
    else:
        columns = np.asarray(train_x)[:, feature_ids]
        nonzero = np.count_nonzero(columns, axis=0)
    return nonzero.astype(np.float64) / float(len(train_indices))


def train_sparse_probe_bundle_direction(
    *,
    x: Any,
    labels: list[int],
    splits: dict[str, list[int]],
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "liblinear",
    top_positive: int = 25,
    top_negative: int = 25,
    min_density: float = 0.02,
    max_density: float = 0.50,
) -> dict[str, Any]:
    """Fit a sparse logistic probe and select top coefficients for steering.

    The returned feature weights live in standardized probe space. This is the
    most direct representation of what the sparse probe used to make its
    prediction; callers may optionally use ``input_weight`` if they want weights
    in raw feature-activation units instead.
    """

    from scipy import sparse

    if top_positive < 0 or top_negative < 0:
        raise ValueError("top_positive/top_negative must be non-negative")
    if top_positive + top_negative <= 0:
        raise ValueError("at least one selected feature is required")
    if not 0.0 <= min_density <= max_density <= 1.0:
        raise ValueError("expected 0 <= min_density <= max_density <= 1")

    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    split_counts = {split: _class_counts(labels, indices) for split, indices in splits.items()}
    if not _has_two_classes(labels, train_indices):
        raise ValueError(f"training split has one class: {split_counts['train']}")
    if not _has_two_classes(labels, val_indices):
        raise ValueError(f"validation split has one class: {split_counts['val']}")
    if not _has_two_classes(labels, test_indices):
        raise ValueError(f"test split has one class: {split_counts['test']}")

    if sparse.issparse(x):
        active_feature_ids = np.unique(x[train_indices].nonzero()[1]).astype(np.int64, copy=False)
    else:
        active_feature_ids = np.arange(x.shape[1], dtype=np.int64)
    if active_feature_ids.size == 0:
        raise ValueError("train split has no active features")
    x_fit = x[:, active_feature_ids]

    best: dict[str, Any] | None = None
    for c_value in c_values:
        model = _make_logistic_pipeline(x_fit, c_value=c_value, max_iter=max_iter, solver=solver)
        model.fit(x_fit[train_indices], [labels[idx] for idx in train_indices])
        val_scores = _positive_scores(model, x_fit, val_indices)
        val_auc = _safe_auc([labels[idx] for idx in val_indices], val_scores)
        rank_auc = val_auc if val_auc is not None else -math.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "best_c": float(c_value),
                "val_auc": val_auc,
                "rank_auc": rank_auc,
            }
    if best is None:
        raise ValueError("no C values provided")

    model = best["model"]
    scaler = model[0]
    logreg = model[-1]
    coef = np.asarray(logreg.coef_[0], dtype=np.float64)
    scaler_scale = np.asarray(getattr(scaler, "scale_", np.ones_like(coef)), dtype=np.float64)
    densities = _train_feature_densities(x, train_indices, active_feature_ids)
    candidates: list[dict[str, Any]] = []
    for fit_index, feature in enumerate(active_feature_ids):
        weight = float(coef[fit_index])
        if weight == 0.0:
            continue
        density = float(densities[fit_index])
        if density < min_density or density > max_density:
            continue
        scale = float(scaler_scale[fit_index])
        candidates.append(
            {
                "fit_feature_index": int(fit_index),
                "feature": int(feature),
                "standardized_coef": weight,
                "abs_standardized_coef": abs(weight),
                "input_weight": float(weight / scale) if scale != 0.0 else None,
                "scaler_scale": scale,
                "train_density": density,
                "association": "correct" if weight > 0 else "incorrect",
            }
        )
    positive = sorted(
        (row for row in candidates if row["standardized_coef"] > 0),
        key=lambda row: (-row["standardized_coef"], row["feature"]),
    )[:top_positive]
    negative = sorted(
        (row for row in candidates if row["standardized_coef"] < 0),
        key=lambda row: (row["standardized_coef"], row["feature"]),
    )[:top_negative]
    selected = []
    for rank, row in enumerate(positive, start=1):
        selected.append({**row, "rank_within_sign": rank})
    for rank, row in enumerate(negative, start=1):
        selected.append({**row, "rank_within_sign": rank})
    if not selected:
        raise ValueError("density filters removed all nonzero probe coefficients")

    test_scores = _positive_scores(model, x_fit, test_indices)
    test_auc = _safe_auc([labels[idx] for idx in test_indices], test_scores)
    return {
        "selected_features": selected,
        "selected_feature_ids": [int(row["feature"]) for row in selected],
        "selected_positive_n": len(positive),
        "selected_negative_n": len(negative),
        "candidate_feature_n": len(candidates),
        "fit_active_feature_n": int(active_feature_ids.size),
        "fit_active_feature_source": "train_nonzero",
        "best_c": best["best_c"],
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "val_auc": best["val_auc"],
        "test_auc": test_auc,
        "split_counts": split_counts,
        "density_filter": {
            "min_density": min_density,
            "max_density": max_density,
        },
    }


def build_weighted_decoder_bundle(
    decoder_rows: dict[int, np.ndarray],
    selected_features: list[dict[str, Any]],
    *,
    weight_key: str = "standardized_coef",
) -> dict[str, Any]:
    """Combine selected decoder rows into one unit-norm steering direction."""

    if not selected_features:
        raise ValueError("selected_features is empty")
    raw: np.ndarray | None = None
    components = []
    for row in selected_features:
        feature = int(row["feature"])
        if feature not in decoder_rows:
            raise KeyError(f"missing decoder row for feature {feature}")
        weight_value = row.get(weight_key)
        if weight_value is None:
            raise ValueError(f"feature {feature} has no usable {weight_key}")
        weight = float(weight_value)
        vector = np.asarray(decoder_rows[feature], dtype=np.float64)
        raw = weight * vector if raw is None else raw + weight * vector
        components.append(
            {
                "feature": feature,
                "weight": weight,
                "decoder_row_norm": float(np.linalg.norm(vector)),
            }
        )
    assert raw is not None
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        raise ValueError("weighted decoder bundle has zero norm")
    unit = (raw / norm).astype(np.float32)
    return {
        "unit_direction": unit,
        "raw_direction": raw.astype(np.float32),
        "raw_norm": norm,
        "weight_key": weight_key,
        "components": components,
    }


def train_raw_probe_direction(
    *,
    activation_path: Path,
    sidecar_path: Path,
    splits_path: Path,
    source_file: str,
    split_family: str,
    seed: int,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "lbfgs",
) -> dict[str, Any]:
    """Refit the raw residual probe and recover its unit direction in input space.

    The Stage 2 raw probes use a StandardScaler followed by logistic regression.
    The steering direction has to undo that scaling: if the fitted logit is
    ``coef_std @ ((x - mean) / scale)``, the equivalent raw-space coefficient is
    ``coef_std / scale``.
    """
    dataset = load_probe_dataset(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        drop_parse_failed=True,
    )
    x = dataset["x"]
    labels = dataset["labels"]
    sidecar = dataset["sidecar"]
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
        raise ValueError(f"training split has one class: {split_counts['train']}")
    if not _has_two_classes(labels, val_indices):
        raise ValueError(f"validation split has one class: {split_counts['val']}")
    if not _has_two_classes(labels, test_indices):
        raise ValueError(f"test split has one class: {split_counts['test']}")

    best: dict[str, Any] | None = None
    for c_value in c_values:
        model = _make_logistic_pipeline(x, c_value=c_value, max_iter=max_iter, solver=solver)
        model.fit(x[train_indices], [labels[idx] for idx in train_indices])
        val_scores = _positive_scores(model, x, val_indices)
        val_auc = _safe_auc([labels[idx] for idx in val_indices], val_scores)
        rank_auc = val_auc if val_auc is not None else -math.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "c": float(c_value),
                "val_auc": val_auc,
                "rank_auc": rank_auc,
            }
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
        raise ValueError("raw probe coefficient has zero norm")
    unit_direction = raw_coef / raw_norm
    train_projection = x[train_indices].astype(np.float64) @ unit_direction
    projection_std = float(train_projection.std(ddof=0))
    if projection_std == 0.0:
        raise ValueError("train projection has zero standard deviation")
    test_scores = _positive_scores(model, x, test_indices)
    test_auc = _safe_auc([labels[idx] for idx in test_indices], test_scores)

    return {
        "unit_direction": unit_direction.astype(np.float32),
        "raw_coef": raw_coef.astype(np.float32),
        "coef_std": coef_std.astype(np.float32),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
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


def make_orthogonal_unit_direction(
    direction: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    unit = np.asarray(direction, dtype=np.float64)
    unit_norm = np.linalg.norm(unit)
    if unit_norm == 0.0:
        raise ValueError("direction has zero norm")
    unit = unit / unit_norm
    rng = np.random.default_rng(seed)
    for _ in range(100):
        candidate = rng.normal(size=unit.shape)
        candidate = candidate - float(candidate @ unit) * unit
        norm = np.linalg.norm(candidate)
        if norm > 1e-12:
            return (candidate / norm).astype(np.float32)
    raise ValueError("failed to sample nonzero orthogonal direction")


def select_balanced_stage1_rows(
    *,
    jsonl_path: Path,
    splits_path: Path,
    source_file: str,
    split_family: str,
    heights: list[int],
    per_height_label: int,
    seed: int,
    drop_parse_failed: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a balanced deterministic subset from the requested split."""
    assignments = read_split_assignments(splits_path)
    target_split = "test"
    height_set = set(heights)
    groups: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    with jsonl_path.open() as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            height = row.get("height")
            if height not in height_set:
                continue
            if drop_parse_failed and row.get("parse_failed"):
                continue
            assignment = assignments.get((source_file, row_index))
            if assignment is None or assignment.get(f"{split_family}_split") != target_split:
                continue
            labeled = dict(row)
            labeled["row_index"] = row_index
            groups[(int(height), bool(row["is_correct_strong"]))].append(labeled)

    counts = {
        f"h{height}_{'correct' if label else 'incorrect'}": len(groups[(height, label)])
        for height in heights
        for label in (False, True)
    }
    selected: list[dict[str, Any]] = []
    missing = {}
    for height in heights:
        for label in (False, True):
            key = (height, label)
            rows = list(groups[key])
            if len(rows) < per_height_label:
                missing[f"h{height}_{'correct' if label else 'incorrect'}"] = len(rows)
                continue
            rng = random.Random(seed + height * 1009 + int(label) * 9176)
            rng.shuffle(rows)
            selected.extend(rows[:per_height_label])
    if missing:
        raise ValueError(
            f"not enough rows for balanced steering subset; requested {per_height_label}, counts={missing}"
        )
    selected.sort(key=lambda row: (int(row["height"]), bool(row["is_correct_strong"]), int(row["row_index"])))
    summary = {
        "source_file": source_file,
        "split_family": split_family,
        "split": target_split,
        "heights": heights,
        "per_height_label": per_height_label,
        "drop_parse_failed": drop_parse_failed,
        "seed": seed,
        "available_counts": counts,
        "selected_rows": len(selected),
        "selected_counts": {
            f"h{height}_{'correct' if label else 'incorrect'}": sum(
                1
                for row in selected
                if row.get("height") == height and bool(row.get("is_correct_strong")) is label
            )
            for height in heights
            for label in (False, True)
        },
    }
    return selected, summary


def score_reply(stage1_row: dict[str, Any], reply: str) -> dict[str, Any]:
    from .bd_path import ensure_on_path
    from .gemma3_parse import parse_hypotheses
    from .inference import classify_failure

    ensure_on_path()
    from evaluate import (  # noqa: WPS433
        compute_quality,
        compute_strong_accuracy,
        compute_weak_accuracy,
        parse_ground_truth,
    )

    pred_hyps = parse_hypotheses(reply or "")
    gt_hyps = parse_ground_truth(stage1_row["ground_truth"])
    raw = stage1_row["ontology_raw"]
    strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
    weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, raw["observations"], raw["theories"])
    quality = compute_quality(pred_hyps, gt_hyps, raw["observations"], raw["theories"])
    if strong_acc == 1:
        weak_acc = 1
        quality = 1.0
    failure_mode = classify_failure(reply or "", pred_hyps)
    return {
        "parsed_hypotheses": pred_hyps,
        "is_correct_strong": bool(strong_acc),
        "is_correct_weak": bool(weak_acc),
        "quality_score": float(quality),
        "failure_mode": failure_mode,
        "parse_failed": failure_mode is not None,
    }


def summarize_steering_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    summary: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        n = len(condition_rows)
        strong = sum(bool(row["is_correct_strong"]) for row in condition_rows)
        weak = sum(bool(row["is_correct_weak"]) for row in condition_rows)
        parse_failed = sum(bool(row["parse_failed"]) for row in condition_rows)
        output_tokens = [int(row.get("generated_token_count", 0)) for row in condition_rows]
        output_chars = [len(row.get("model_output", "")) for row in condition_rows]
        by_original = {}
        for original in (False, True):
            subset = [row for row in condition_rows if bool(row.get("original_is_correct_strong")) is original]
            if subset:
                by_original[str(original).lower()] = {
                    "n": len(subset),
                    "strong_accuracy": sum(bool(row["is_correct_strong"]) for row in subset) / len(subset),
                    "parse_fail_rate": sum(bool(row["parse_failed"]) for row in subset) / len(subset),
                }
        by_height = {}
        for height in sorted({row.get("height") for row in condition_rows}):
            subset = [row for row in condition_rows if row.get("height") == height]
            by_height[f"h{height}"] = {
                "n": len(subset),
                "strong_accuracy": sum(bool(row["is_correct_strong"]) for row in subset) / len(subset),
                "parse_fail_rate": sum(bool(row["parse_failed"]) for row in subset) / len(subset),
            }
        summary[condition] = {
            "n": n,
            "strong_accuracy": strong / n if n else None,
            "weak_accuracy": weak / n if n else None,
            "parse_fail_rate": parse_failed / n if n else None,
            "mean_quality": sum(float(row["quality_score"]) for row in condition_rows) / n if n else None,
            "mean_generated_tokens": sum(output_tokens) / n if n else None,
            "mean_output_chars": sum(output_chars) / n if n else None,
            "by_original_is_correct_strong": by_original,
            "by_height": by_height,
        }

    baselines = {
        int(row["source_row_index"]): row
        for row in rows
        if row.get("condition") == "baseline"
    }
    flips = {}
    for condition, condition_rows in sorted(by_condition.items()):
        if condition == "baseline":
            continue
        paired = [
            (baselines[int(row["source_row_index"])], row)
            for row in condition_rows
            if int(row["source_row_index"]) in baselines
        ]
        if not paired:
            continue
        false_to_true = sum(
            (not base["is_correct_strong"]) and steered["is_correct_strong"]
            for base, steered in paired
        )
        true_to_false = sum(
            base["is_correct_strong"] and (not steered["is_correct_strong"])
            for base, steered in paired
        )
        changed = sum(base["is_correct_strong"] != steered["is_correct_strong"] for base, steered in paired)
        flips[condition] = {
            "paired_n": len(paired),
            "false_to_true": int(false_to_true),
            "true_to_false": int(true_to_false),
            "changed": int(changed),
            "net_accuracy_delta": (
                sum(bool(steered["is_correct_strong"]) for _, steered in paired)
                - sum(bool(base["is_correct_strong"]) for base, _ in paired)
            )
            / len(paired),
        }
    return {
        "by_condition": summary,
        "flips_vs_baseline": flips,
    }
