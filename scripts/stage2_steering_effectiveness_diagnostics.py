#!/usr/bin/env python3
"""Build artifact-level steering-effectiveness diagnostics for next-paper Track 2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_causal_abstraction import make_experiment_report, now_utc  # noqa: E402
from src.stage2_probes import read_json, write_json  # noqa: E402


DEFAULT_LAP_SCORE = "gold_vs_output_first_diff_logit_margin"


DEFAULT_ROWS = (
    {
        "model": "gemma3_27b",
        "model_display": "Gemma 3 27B",
        "task": "infer_property",
        "split": "s1",
        "site_or_layer": "L45",
        "raw_report": "docs/raw_probe_27b_s1.json",
        "b0_report": "docs/stage2_b0_summary_27b_s1.json",
        "b0_key": "gemma3-27b__infer_property",
        "metadata_residualization": "docs/raw_probe_metadata_residualization_27b_l45.json",
        "steering_report": "docs/raw_steering_27b_l45_property_decode_sweep.json",
        "related_error_steering_report": "docs/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep.json",
        "positive_control_report": "docs/positive_control_format_gemma3_27b_l45.json",
        "lap_report": "docs/lap_gemma3_27b_infer_property_s1.json",
    },
    {
        "model": "gemma3_27b",
        "model_display": "Gemma 3 27B",
        "task": "infer_subtype",
        "split": "s1",
        "site_or_layer": "L45",
        "raw_report": "docs/raw_probe_27b_s1.json",
        "b0_report": "docs/stage2_b0_summary_27b_s1.json",
        "b0_key": "gemma3-27b__infer_subtype",
        "metadata_residualization": "docs/raw_probe_metadata_residualization_27b_l45.json",
        "steering_report": None,
        "positive_control_report": "docs/positive_control_format_gemma3_27b_l45.json",
        "lap_report": "docs/lap_gemma3_27b_infer_subtype_s1.json",
    },
    {
        "model": "qwen35_27b",
        "model_display": "Qwen3.5 27B",
        "task": "infer_property",
        "split": "s1",
        "site_or_layer": "L45",
        "raw_report": "docs/qwen_scope_raw_probe_27b_l45_s1.json",
        "b0_report": "docs/qwen35_27b_b0_summary.json",
        "b0_key": "Qwen/Qwen3.5-27B__infer_property",
        "metadata_residualization": "docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json",
        "steering_report": "docs/qwen35_raw_steering_27b_l45_property_pilot.json",
        "lap_report": "docs/lap_qwen35_27b_infer_property_s1.json",
    },
    {
        "model": "qwen35_27b",
        "model_display": "Qwen3.5 27B",
        "task": "infer_property",
        "split": "s1",
        "site_or_layer": "L53",
        "raw_report": "docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s1.json",
        "b0_report": "docs/qwen35_27b_b0_summary.json",
        "b0_key": "Qwen/Qwen3.5-27B__infer_property",
        "metadata_residualization": "docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json",
        "steering_report": "docs/qwen35_raw_steering_27b_l53_property_pilot.json",
        "lap_report": "docs/lap_qwen35_27b_infer_property_s1.json",
    },
    {
        "model": "qwen35_27b",
        "model_display": "Qwen3.5 27B",
        "task": "infer_subtype",
        "split": "s1",
        "site_or_layer": "L45",
        "raw_report": "docs/qwen_scope_raw_probe_27b_l45_s1.json",
        "b0_report": "docs/qwen35_27b_b0_summary.json",
        "b0_key": "Qwen/Qwen3.5-27B__infer_subtype",
        "metadata_residualization": "docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json",
        "steering_report": None,
        "lap_report": "docs/lap_qwen35_27b_infer_subtype_s1.json",
    },
    {
        "model": "qwen35_27b",
        "model_display": "Qwen3.5 27B",
        "task": "infer_subtype",
        "split": "s1",
        "site_or_layer": "L53",
        "raw_report": "docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s1.json",
        "b0_report": "docs/qwen35_27b_b0_summary.json",
        "b0_key": "Qwen/Qwen3.5-27B__infer_subtype",
        "metadata_residualization": "docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json",
        "steering_report": None,
        "lap_report": "docs/lap_qwen35_27b_infer_subtype_s1.json",
    },
)


def nested(payload: dict[str, Any] | None, keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def read_json_or_none(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return read_json(candidate)


def layer_key(site_or_layer: str) -> str:
    if site_or_layer.startswith("L"):
        return site_or_layer
    return f"L{site_or_layer}"


def raw_probe_metrics(raw_report: dict[str, Any], task: str, site_or_layer: str) -> dict[str, Any]:
    site = layer_key(site_or_layer)
    task_results = nested(raw_report, ["results", task], {})
    layer_result = nested(raw_report, ["results", task, site], {})
    best = nested(raw_report, ["best_by_task", task], {})
    best_layer = best.get("layer")
    best_auc = best.get("test_auc")
    scanned_layers = []
    for key, value in sorted(task_results.items()):
        if isinstance(value, dict):
            scanned_layers.append({"site_or_layer": key, "test_auc": value.get("test_auc")})
    return {
        "site_or_layer": site,
        "raw_auc": layer_result.get("test_auc"),
        "raw_auc_ci": layer_result.get("test_auc_ci"),
        "val_auc": layer_result.get("val_auc"),
        "n_test": nested(layer_result, ["split_counts", "test", "n"]),
        "test_positive_n": nested(layer_result, ["split_counts", "test", "positive_n"]),
        "test_negative_n": nested(layer_result, ["split_counts", "test", "negative_n"]),
        "best_scanned_layer": best_layer,
        "best_scanned_raw_auc": best_auc,
        "scanned_layers": scanned_layers,
    }


def best_b0_metrics(b0_report: dict[str, Any] | None, b0_key: str, split: str) -> dict[str, Any]:
    if b0_report is None:
        return {}
    best = nested(b0_report, ["best_pre_output_baseline", b0_key, split], {})
    all_sets = nested(b0_report, ["results", b0_key, split], {})
    return {
        "best_feature_set": best.get("feature_set"),
        "best_test_auc": best.get("test_auc"),
        "best_val_auc": best.get("val_auc"),
        "all_feature_sets": all_sets,
    }


def best_by_auc(items: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    best_name = None
    best_value: dict[str, Any] = {}
    best_auc = None
    for name, value in items.items():
        if not isinstance(value, dict):
            continue
        auc = value.get("test_auc")
        if auc is None:
            continue
        if best_auc is None or auc > best_auc:
            best_name = name
            best_value = value
            best_auc = auc
    return best_name, best_value


def metadata_adjustment_metrics(
    residualization_report: dict[str, Any] | None,
    *,
    split: str,
    task: str,
    requested_site: str,
) -> dict[str, Any]:
    if residualization_report is None:
        return {"status": "missing"}
    available_layer = residualization_report.get("layer")
    available_site = f"L{available_layer}" if available_layer is not None else None
    result = nested(residualization_report, ["results", split, task], {})
    if not result:
        return {"status": "missing_task"}

    best_metadata_name, best_metadata = best_by_auc(result.get("metadata_only", {}))
    if best_metadata_name is None:
        return {"status": "missing_metadata_only", "available_site_or_layer": available_site}
    plus = nested(result, ["metadata_plus_raw_score", best_metadata_name], {})
    residualized = nested(result, ["raw_residualized", best_metadata_name], {})
    raw = result.get("raw", {})
    return {
        "status": "available",
        "available_site_or_layer": available_site,
        "exact_site_match": available_site == requested_site,
        "reference_metadata_set": best_metadata_name,
        "metadata_only_auc": best_metadata.get("test_auc"),
        "raw_refit_auc": raw.get("refit_test_auc") or raw.get("reported_test_auc"),
        "metadata_plus_raw_auc": plus.get("test_auc"),
        "metadata_plus_raw_delta_vs_metadata": plus.get("test_auc_delta_vs_metadata"),
        "raw_residualized_auc": residualized.get("test_auc"),
        "raw_residualized_delta_vs_metadata": residualized.get("test_auc_delta_vs_metadata"),
        "raw_residualized_drop_vs_raw": residualized.get("test_auc_drop_vs_raw"),
        "raw_score_train_r2_from_metadata": residualized.get("train_r2_raw_score"),
    }


def max_flip_metric(flips: dict[str, Any], prefix: str, key: str) -> int | float | None:
    values = [
        value.get(key)
        for label, value in flips.items()
        if label.startswith(prefix) and isinstance(value, dict) and value.get(key) is not None
    ]
    if not values:
        return None
    return max(values)


def max_abs_delta(flips: dict[str, Any], prefix: str) -> float | None:
    values = [
        abs(value.get("net_accuracy_delta", 0.0))
        for label, value in flips.items()
        if label.startswith(prefix) and isinstance(value, dict)
    ]
    if not values:
        return None
    return float(max(values))


def condition_labels(flips: dict[str, Any], prefix: str) -> list[str]:
    return sorted(label for label in flips if label.startswith(prefix))


def direction_group_summary(flips: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "conditions": condition_labels(flips, prefix),
        "max_false_to_true": max_flip_metric(flips, prefix, "false_to_true"),
        "max_true_to_false": max_flip_metric(flips, prefix, "true_to_false"),
        "max_changed": max_flip_metric(flips, prefix, "changed"),
        "max_abs_accuracy_delta": max_abs_delta(flips, prefix),
    }


def _max_existing(values: list[int | float | None]) -> int | float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def matched_noise_from_steering(flips: dict[str, Any]) -> dict[str, Any]:
    raw = direction_group_summary(flips, "raw_")
    orthogonal = direction_group_summary(flips, "orthogonal_")
    gaussian = direction_group_summary(flips, "gaussian_")
    control_false_to_true = _max_existing(
        [orthogonal.get("max_false_to_true"), gaussian.get("max_false_to_true")]
    )
    control_changed = _max_existing([orthogonal.get("max_changed"), gaussian.get("max_changed")])
    raw_repairs = raw.get("max_false_to_true")
    has_gaussian = bool(gaussian["conditions"])
    return {
        "status": "available_no_sigma_estimate" if has_gaussian else "missing_gaussian_control",
        "has_matched_gaussian_control": has_gaussian,
        "raw_max_false_to_true": raw_repairs,
        "raw_max_changed": raw.get("max_changed"),
        "orthogonal_max_false_to_true": orthogonal.get("max_false_to_true"),
        "orthogonal_max_changed": orthogonal.get("max_changed"),
        "gaussian_max_false_to_true": gaussian.get("max_false_to_true"),
        "gaussian_max_changed": gaussian.get("max_changed"),
        "control_max_false_to_true": control_false_to_true,
        "control_max_changed": control_changed,
        "repair_exceeds_controls_by_count": (
            raw_repairs is not None
            and control_false_to_true is not None
            and raw_repairs > control_false_to_true
        ),
        "sigma_test_status": "not_estimated_from_single_matched_control_family",
    }


def steering_metrics(steering_report: dict[str, Any] | None, *, report_path: str | None = None) -> dict[str, Any]:
    if steering_report is None:
        return {"status": "missing"}
    summary = steering_report.get("summary", {})
    flips = summary.get("flips_vs_baseline", {})
    baseline = nested(summary, ["by_condition", "baseline"], {})
    by_condition = summary.get("by_condition", {})
    raw = direction_group_summary(flips, "raw_")
    orthogonal = direction_group_summary(flips, "orthogonal_")
    gaussian = direction_group_summary(flips, "gaussian_")
    parse_rates = [
        value.get("parse_fail_rate")
        for value in by_condition.values()
        if isinstance(value, dict) and value.get("parse_fail_rate") is not None
    ]
    return {
        "status": "available",
        "json_report_path": report_path,
        "report_path": steering_report.get("out_jsonl"),
        "slurm_job_id": steering_report.get("slurm_job_id"),
        "elapsed_seconds": steering_report.get("elapsed_seconds"),
        "selection": steering_report.get("selection"),
        "baseline_n": baseline.get("n"),
        "baseline_strong_accuracy": baseline.get("strong_accuracy"),
        "baseline_weak_accuracy": baseline.get("weak_accuracy"),
        "baseline_parse_fail_rate": baseline.get("parse_fail_rate"),
        "raw_conditions": raw["conditions"],
        "orthogonal_conditions": orthogonal["conditions"],
        "gaussian_conditions": gaussian["conditions"],
        "has_regenerated_baseline": baseline.get("n") is not None,
        "has_matched_gaussian_control": bool(gaussian["conditions"]),
        "raw_max_false_to_true": raw["max_false_to_true"],
        "raw_max_true_to_false": raw["max_true_to_false"],
        "raw_max_changed": raw["max_changed"],
        "raw_max_abs_accuracy_delta": raw["max_abs_accuracy_delta"],
        "orthogonal_max_false_to_true": orthogonal["max_false_to_true"],
        "orthogonal_max_true_to_false": orthogonal["max_true_to_false"],
        "orthogonal_max_changed": orthogonal["max_changed"],
        "orthogonal_max_abs_accuracy_delta": orthogonal["max_abs_accuracy_delta"],
        "gaussian_max_false_to_true": gaussian["max_false_to_true"],
        "gaussian_max_true_to_false": gaussian["max_true_to_false"],
        "gaussian_max_changed": gaussian["max_changed"],
        "gaussian_max_abs_accuracy_delta": gaussian["max_abs_accuracy_delta"],
        "paired_n_max": max_flip_metric(flips, "", "paired_n"),
        "parse_fail_rate_min": min(parse_rates) if parse_rates else None,
        "parse_fail_rate_max": max(parse_rates) if parse_rates else None,
        "matched_noise_summary": matched_noise_from_steering(flips),
        "all_flips_vs_baseline": flips,
    }


def positive_control_metrics(path: str | None) -> dict[str, Any]:
    if not path:
        return {"status": "not_configured"}
    candidate = Path(path)
    if not candidate.exists():
        return {"status": "missing", "report_path": str(candidate)}
    report = read_json(candidate)
    summary = nested(report, ["summary", "matched_noise_summary"], {})
    passed = summary.get("passed_positive_control_gate")
    return {
        "status": "passed" if passed is True else "failed_or_unclear",
        "report_path": str(candidate),
        "target_variable": report.get("target_variable"),
        "task": report.get("task"),
        "passed_positive_control_gate": passed,
        "best_effect_over_control_sigma": nested(summary, ["best_toward_upper", "effect_over_control_sigma"]),
        "control_abs_delta_std": summary.get("control_abs_uppercase_delta_std"),
    }



def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_deltas = [x - x_mean for x in xs]
    y_deltas = [y - y_mean for y in ys]
    x_var = sum(delta * delta for delta in x_deltas)
    y_var = sum(delta * delta for delta in y_deltas)
    if x_var == 0.0 or y_var == 0.0:
        return None
    return float(sum(x * y for x, y in zip(x_deltas, y_deltas)) / math.sqrt(x_var * y_var))


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        j = idx + 1
        while j < len(indexed) and indexed[j][1] == indexed[idx][1]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for original_idx, _ in indexed[idx:j]:
            ranks[original_idx] = avg_rank
        idx = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rank_values(xs), _rank_values(ys))


def _correlate_projection_effects(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    projection_z = [pair["projection_z"] for pair in pairs]
    abs_projection_z = [abs(pair["projection_z"]) for pair in pairs]
    accuracy_delta = [pair["accuracy_delta"] for pair in pairs]
    repaired = [pair["repaired"] for pair in pairs]
    degraded = [pair["degraded"] for pair in pairs]
    changed = [pair["changed"] for pair in pairs]
    baseline_false_n = sum(1 for pair in pairs if pair["baseline_correct"] == 0)
    baseline_true_n = sum(1 for pair in pairs if pair["baseline_correct"] == 1)
    return {
        "paired_n": len(pairs),
        "baseline_false_n": baseline_false_n,
        "baseline_true_n": baseline_true_n,
        "repair_n": int(sum(repaired)),
        "degradation_n": int(sum(degraded)),
        "changed_n": int(sum(changed)),
        "mean_accuracy_delta": float(sum(accuracy_delta) / len(accuracy_delta)) if accuracy_delta else None,
        "pearson_projection_z_vs_accuracy_delta": _pearson(projection_z, accuracy_delta),
        "spearman_projection_z_vs_accuracy_delta": _spearman(projection_z, accuracy_delta),
        "pearson_projection_z_vs_repair": _pearson(projection_z, repaired),
        "spearman_projection_z_vs_repair": _spearman(projection_z, repaired),
        "pearson_abs_projection_z_vs_changed": _pearson(abs_projection_z, changed),
        "spearman_abs_projection_z_vs_changed": _spearman(abs_projection_z, changed),
    }


def _condition_kind(row: dict[str, Any]) -> str:
    kind = row.get("direction_kind")
    if kind:
        return str(kind)
    label = str(row.get("condition") or "")
    if "_" in label:
        return label.split("_", 1)[0]
    return label or "unknown"


def probe_confidence_correlation_status(steering: dict[str, Any]) -> dict[str, Any]:
    jsonl_path = steering.get("report_path")
    if steering.get("status") != "available" or not jsonl_path:
        return {"status": "missing_steering_rows"}
    candidate = Path(jsonl_path)
    if not candidate.exists():
        return {"status": "missing_steering_rows", "report_path": jsonl_path}
    rows = []
    with candidate.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    score_keys = ("direction_projection_z", "probe_score", "probe_confidence", "probe_margin", "direction_projection")
    score_key = next((key for key in score_keys if any(key in row for row in rows)), None)
    if score_key is None:
        return {
            "status": "needs_probe_score_sidecar",
            "report_path": jsonl_path,
            "required_row_fields": ["example_id", "condition", "is_correct_strong", "direction_projection_z"],
            "note": "Current steering JSONL stores paired outcomes but not per-row probe confidence/projection.",
        }

    baseline_by_row: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("condition") != "baseline":
            continue
        row_index = row.get("source_row_index")
        if row_index is not None:
            baseline_by_row[int(row_index)] = row

    pairs_by_kind: dict[str, list[dict[str, Any]]] = {}
    missing_baseline = 0
    missing_score = 0
    for row in rows:
        if row.get("condition") == "baseline":
            continue
        row_index = row.get("source_row_index")
        if row_index is None or int(row_index) not in baseline_by_row:
            missing_baseline += 1
            continue
        projection_z = _finite_float(row.get("direction_projection_z"))
        if projection_z is None:
            projection_z = _finite_float(row.get(score_key))
        if projection_z is None:
            missing_score += 1
            continue
        baseline = baseline_by_row[int(row_index)]
        baseline_correct = int(bool(baseline.get("is_correct_strong")))
        intervention_correct = int(bool(row.get("is_correct_strong")))
        model_output = row.get("model_output")
        baseline_output = baseline.get("model_output")
        pair = {
            "projection_z": projection_z,
            "baseline_correct": baseline_correct,
            "intervention_correct": intervention_correct,
            "accuracy_delta": intervention_correct - baseline_correct,
            "repaired": int(baseline_correct == 0 and intervention_correct == 1),
            "degraded": int(baseline_correct == 1 and intervention_correct == 0),
            "changed": int(model_output != baseline_output),
        }
        pairs_by_kind.setdefault(_condition_kind(row), []).append(pair)

    by_direction_kind = {
        kind: _correlate_projection_effects(pairs)
        for kind, pairs in sorted(pairs_by_kind.items())
        if pairs
    }
    paired_n = sum(metrics["paired_n"] for metrics in by_direction_kind.values())
    if not by_direction_kind:
        return {
            "status": "score_field_available_no_paired_rows",
            "report_path": jsonl_path,
            "score_key": score_key,
            "missing_baseline_rows": missing_baseline,
            "missing_score_rows": missing_score,
        }
    return {
        "status": "available",
        "report_path": jsonl_path,
        "score_key": score_key,
        "paired_n": paired_n,
        "missing_baseline_rows": missing_baseline,
        "missing_score_rows": missing_score,
        "by_direction_kind": by_direction_kind,
        "interpretation_note": (
            "Correlations are descriptive diagnostics over regenerated paired rows; they are not causal evidence. "
            "Null values indicate insufficient variance in the score or outcome."
        ),
    }

def short_lap_score_name(score_name: str | None) -> str:
    labels = {
        "gold_vs_output_first_diff_logit_margin": "first-diff margin",
        "gold_vs_output_first_diff_logprob_margin": "first-diff logprob",
        "gold_minus_output_mean_logprob": "gold-output mean",
        "gold_mean_logprob": "gold mean",
        "gold_first_token_logprob": "gold first-token",
        "output_mean_logprob": "output mean",
    }
    return labels.get(score_name or "", score_name or "pending")


def lap_profile_metrics(path: str | None, requested_site: str) -> dict[str, Any]:
    if not path:
        return {
            "status": "pending_model_forward",
            "artifact_proxy": "best scanned raw-probe layer only",
        }
    candidate = Path(path)
    if not candidate.exists():
        return {
            "status": "pending_model_forward",
            "report_path": str(candidate),
            "artifact_proxy": "best scanned raw-probe layer only",
        }
    report = read_json(candidate)
    layer_scores = nested(report, ["results", requested_site, "scores", "test"], {})
    requested_auc_by_score = {
        score_name: metrics.get("auc")
        for score_name, metrics in layer_scores.items()
        if isinstance(metrics, dict) and "auc" in metrics
    }
    peaks = report.get("peak_by_score", {})
    selected_peak = peaks.get(DEFAULT_LAP_SCORE, {}) if isinstance(peaks, dict) else {}
    return {
        "status": "available",
        "report_path": str(candidate),
        "target_variable": report.get("target_variable"),
        "method": report.get("method"),
        "score_names": report.get("score_names", []),
        "selected_score": DEFAULT_LAP_SCORE,
        "selected_score_label": short_lap_score_name(DEFAULT_LAP_SCORE),
        "requested_site_or_layer": requested_site,
        "requested_layer_auc_by_score": requested_auc_by_score,
        "requested_layer_selected_auc": requested_auc_by_score.get(DEFAULT_LAP_SCORE),
        "peak_by_score": peaks,
        "selected_peak_layer": selected_peak.get("layer"),
        "selected_peak_auc": selected_peak.get("auc"),
        "kept_rows": report.get("kept_rows"),
        "notes": report.get("notes", []),
    }


def predicted_regime(
    *,
    raw_auc: float | None,
    raw_minus_b0_auc: float | None,
    lap_selected_peak_auc: float | None,
    steering: dict[str, Any],
) -> str:
    if raw_auc is None:
        return "missing_raw_probe"
    if raw_auc < 0.8:
        return "weak_or_moderate_linear_readout"
    low_lap = lap_selected_peak_auc is not None and lap_selected_peak_auc < 0.55
    if steering.get("status") != "available":
        if low_lap and raw_minus_b0_auc is not None and raw_minus_b0_auc >= 0.1:
            return "strong_activation_over_metadata_low_lap_accessibility_steerability_untested"
        if raw_minus_b0_auc is not None and raw_minus_b0_auc >= 0.1:
            return "strong_activation_over_metadata_steerability_untested"
        return "linear_readout_steerability_untested"
    if steering.get("raw_max_false_to_true") == 0:
        if steering.get("has_matched_gaussian_control"):
            return "linearly_readable_matched_control_raw_steering_null"
        if low_lap:
            return "linearly_readable_low_lap_accessibility_historical_raw_steering_null"
        if raw_minus_b0_auc is not None and raw_minus_b0_auc >= 0.1:
            return "linearly_readable_activation_over_metadata_historical_raw_steering_null"
        return "linearly_readable_historical_raw_steering_null"
    return "historical_raw_steering_has_some_repair"


def build_row(config: dict[str, Any]) -> dict[str, Any]:
    raw_report = read_json(Path(config["raw_report"]))
    b0_report = read_json_or_none(config.get("b0_report"))
    residualization_report = read_json_or_none(config.get("metadata_residualization"))
    steering_report = read_json_or_none(config.get("steering_report"))
    related_error_steering_report = read_json_or_none(config.get("related_error_steering_report"))

    raw = raw_probe_metrics(raw_report, config["task"], config["site_or_layer"])
    b0 = best_b0_metrics(b0_report, config["b0_key"], config["split"])
    metadata_adjustment = metadata_adjustment_metrics(
        residualization_report,
        split=config["split"],
        task=config["task"],
        requested_site=raw["site_or_layer"],
    )
    steering = steering_metrics(steering_report, report_path=config.get("steering_report"))
    related_error_steering = steering_metrics(
        related_error_steering_report,
        report_path=config.get("related_error_steering_report"),
    )
    lap_profile = lap_profile_metrics(config.get("lap_report"), raw["site_or_layer"])
    positive_control = positive_control_metrics(config.get("positive_control_report"))
    confidence_correlation = probe_confidence_correlation_status(steering)

    raw_auc = raw.get("raw_auc")
    best_b0_auc = b0.get("best_test_auc")
    raw_minus_b0_auc = raw_auc - best_b0_auc if raw_auc is not None and best_b0_auc is not None else None
    regime = predicted_regime(
        raw_auc=raw_auc,
        raw_minus_b0_auc=raw_minus_b0_auc,
        lap_selected_peak_auc=lap_profile.get("selected_peak_auc"),
        steering=steering,
    )

    controls = []
    if steering.get("orthogonal_conditions"):
        controls.append("orthogonal_direction")
    if steering.get("has_matched_gaussian_control"):
        controls.append("matched_gaussian_noise")
    if steering.get("status") == "available" and positive_control.get("passed_positive_control_gate"):
        controls.append("positive_control")
    result_type = "predictive_and_causal" if steering.get("status") == "available" else "predictive"
    if steering.get("status") == "available" and steering.get("has_matched_gaussian_control"):
        claim = (
            "Artifact-level diagnostic: raw correctness is linearly readable, but the regenerated raw-direction "
            "decode-step intervention did not produce false-to-true repair above orthogonal or Gaussian controls."
        )
    elif steering.get("status") == "available":
        claim = (
            "Artifact-level diagnostic: raw correctness is linearly readable and historical raw-direction steering did "
            "not produce false-to-true repair; matched-noise controls remain required for this row."
        )
    else:
        claim = "Artifact-level diagnostic: raw correctness is linearly readable; steerability is not yet tested for this row."
    notes = [
        "Predicted regime is a planning label, not a manuscript claim.",
    ]
    if lap_profile.get("status") == "available":
        notes.append("LAP/logit-lens accessibility report is available for this row.")
    else:
        notes.append("LAP/logit-lens accessibility is pending GPU/model-forward implementation.")
    if metadata_adjustment.get("status") == "available" and not metadata_adjustment.get("exact_site_match"):
        notes.append(
            "Metadata-residualization proxy is from "
            f"{metadata_adjustment.get('available_site_or_layer')}, not requested site {raw['site_or_layer']}."
        )
    if steering.get("status") == "available" and not steering.get("has_matched_gaussian_control"):
        notes.append("Historical steering lacks matched Gaussian/noise controls.")
    if positive_control.get("status") == "passed":
        notes.append("Gemma L45 format/casing positive-control gate passed for this hook family.")
    if confidence_correlation.get("status") == "needs_probe_score_sidecar":
        notes.append("Probe-confidence vs steering-effect correlation needs row-level probe scores or projections.")

    report = make_experiment_report(
        model=config["model"],
        task=config["task"],
        target_variable="free_form_correctness",
        split=config["split"],
        site_or_layer=raw["site_or_layer"],
        method="steering_effectiveness_artifact_preflight",
        representation_type="raw_direction",
        result_type=result_type,
        controls=controls,
        n=raw.get("n_test"),
        baseline_metrics={
            "raw_probe": raw,
            "best_metadata_baseline": b0,
            "raw_minus_best_metadata_auc": raw_minus_b0_auc,
            "metadata_adjustment": metadata_adjustment,
            "lap_profile": {
                **lap_profile,
                "artifact_proxy_best_scanned_layer": raw.get("best_scanned_layer"),
                "artifact_proxy_best_scanned_raw_auc": raw.get("best_scanned_raw_auc"),
            },
            "positive_control_gate": positive_control,
            "probe_confidence_vs_steering_effect": confidence_correlation,
        },
        intervention_metrics={
            "raw_steering": steering,
            "related_error_subspace_steering": related_error_steering,
        },
        paired_flips=steering.get("all_flips_vs_baseline", {}),
        parse_fail_rate=steering.get("baseline_parse_fail_rate"),
        matched_noise_summary=steering.get(
            "matched_noise_summary",
            {"status": "missing", "required_for_new_interventions": True},
        ),
        causal_abstraction_claim=claim,
        notes=notes,
    )
    return {
        "row_id": f"{config['model']}__{config['task']}__{raw['site_or_layer']}__{config['split']}",
        "model_display": config["model_display"],
        "predicted_steerability_regime": regime,
        "diagnostics": {
            "raw_auc": raw_auc,
            "best_metadata_baseline_auc": best_b0_auc,
            "best_metadata_feature_set": b0.get("best_feature_set"),
            "raw_minus_best_metadata_auc": raw_minus_b0_auc,
            "metadata_adjustment": metadata_adjustment,
            "lap_profile": lap_profile,
            "raw_steering": steering,
            "related_error_subspace_steering": related_error_steering,
            "positive_control_gate": positive_control,
            "probe_confidence_vs_steering_effect": confidence_correlation,
        },
        "experiment_report": report,
    }

def qwen_metadata_note(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qwen = [
        row
        for row in rows
        if row["experiment_report"]["model"] == "qwen35_27b"
        and row["experiment_report"]["site_or_layer"] == "L53"
    ]
    gemma_subtype = next(
        (
            row
            for row in rows
            if row["experiment_report"]["model"] == "gemma3_27b"
            and row["experiment_report"]["task"] == "infer_subtype"
        ),
        None,
    )
    return {
        "claim": "Qwen strengthens the activation-over-metadata result because Qwen B0 baselines are much weaker than raw readouts.",
        "qwen_l53": [
            {
                "task": row["experiment_report"]["task"],
                "raw_auc": row["diagnostics"]["raw_auc"],
                "best_b0_auc": row["diagnostics"]["best_metadata_baseline_auc"],
                "raw_minus_b0_auc": row["diagnostics"]["raw_minus_best_metadata_auc"],
            }
            for row in qwen
        ],
        "gemma_subtype_context": {
            "raw_auc": nested(gemma_subtype, ["diagnostics", "raw_auc"]),
            "best_b0_auc": nested(gemma_subtype, ["diagnostics", "best_metadata_baseline_auc"]),
            "raw_minus_b0_auc": nested(gemma_subtype, ["diagnostics", "raw_minus_best_metadata_auc"]),
            "note": "Gemma subtype remains predictive, but its best metadata baseline is much closer to raw than Qwen's.",
        },
    }


def build_payload() -> dict[str, Any]:
    rows = [build_row(dict(config)) for config in DEFAULT_ROWS]
    return {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "report_kind": "steering_effectiveness_artifact_preflight",
        "target_track": "Track 2: Steering-Effectiveness Diagnostics",
        "scope": {
            "status": "artifact_preflight",
            "included": [
                "raw-probe AUCs",
                "metadata baselines",
                "metadata-adjusted proxy diagnostics where available",
                "historical raw-direction steering summaries where available",
                "matched Gaussian controls for refreshed Gemma raw/error steering reruns",
                "positive-control gate status for Gemma L45 hook family",
            ],
            "pending": [
                "entropy/branching/KL steerability predictors",
                "probe-confidence vs steering-effect correlations for Qwen/historical rows that still lack projection-enabled sidecars",
                "matched Gaussian/noise controls for Qwen steering pilots and future intervention families",
                "known positive-control steering task for Qwen/future non-Gemma hook families",
            ],
        },
        "diagnostics": rows,
        "qwen_metadata_comparison": qwen_metadata_note(rows),
        "next_actions": [
            "Gemma property projection-enabled reruns are complete; add projection-enabled matched-control reruns only for rows that still need causal interpretation.",
            "Add entropy/branching/KL-style steerability predictors where the required logits are available.",
            "Refresh Qwen property steering with regenerated baseline, orthogonal, Gaussian, and Qwen positive-control gates if Qwen causal interpretation is needed.",
            "Use this diagnostic table to choose the next intervention family: optimized vectors, DAS/distributed interchange, or decode-time correction.",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Steering-Effectiveness Diagnostics",
        "",
        "This is the Track 2 artifact preflight. It summarizes existing probe, metadata, steering, and LAP/logit-lens artifacts where available.",
        "",
        "LAP/logit-lens artifacts are complete for the target rows. Entropy/branching/KL predictors are still pending; confidence/effect correlations are available for the regenerated Gemma property row and require sidecar fields for historical or untested rows.",
        "",
        "Gemma L45 raw/error steering now has regenerated baselines and matched Gaussian controls; Qwen property steering remains a historical pilot without Gaussian controls.",
        "",
        "## Diagnostic Table",
        "",
        "| Model | Task | Site | Raw AUC | Best B0 | Raw-B0 | Metadata-adjusted proxy | LAP profile | Prior steering | Planning regime |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["diagnostics"]:
        report = row["experiment_report"]
        diag = row["diagnostics"]
        meta = diag["metadata_adjustment"]
        lap = diag.get("lap_profile", {})
        steering = diag["raw_steering"]
        if meta.get("status") == "available":
            exact = "exact" if meta.get("exact_site_match") else f"proxy {meta.get('available_site_or_layer')}"
            meta_text = (
                f"{exact}; +raw delta={fmt(meta.get('metadata_plus_raw_delta_vs_metadata'))}; "
                f"resid AUC={fmt(meta.get('raw_residualized_auc'))}"
            )
        else:
            meta_text = meta.get("status", "missing")
        if lap.get("status") == "available":
            lap_text = (
                f"{lap.get('selected_peak_layer')} peak {fmt(lap.get('selected_peak_auc'))}; "
                f"row {fmt(lap.get('requested_layer_selected_auc'))} "
                f"({lap.get('selected_score_label', short_lap_score_name(lap.get('selected_score')))})"
            )
        else:
            lap_text = "pending"
        if steering.get("status") == "available":
            control_text = "; Gaussian ctrl" if steering.get("has_matched_gaussian_control") else ""
            steer_text = (
                f"n={fmt(steering.get('baseline_n'), 0)}, "
                f"raw F->T max={fmt(steering.get('raw_max_false_to_true'), 0)}, "
                f"raw changed max={fmt(steering.get('raw_max_changed'), 0)}"
                f"{control_text}"
            )
        else:
            steer_text = "untested"
        lines.append(
            "| "
            f"{row['model_display']} | "
            f"`{report['task']}` | "
            f"`{report['site_or_layer']}` | "
            f"{fmt(diag['raw_auc'])} | "
            f"{fmt(diag['best_metadata_baseline_auc'])} (`{diag['best_metadata_feature_set']}`) | "
            f"{fmt(diag['raw_minus_best_metadata_auc'])} | "
            f"{meta_text} | "
            f"{lap_text} | "
            f"{steer_text} | "
            f"`{row['predicted_steerability_regime']}` |"
        )

    qwen_note = payload["qwen_metadata_comparison"]
    lines.extend(
        [
            "",
            "## Qwen Metadata Takeaway",
            "",
            qwen_note["claim"],
            "",
            "| Task | Qwen L53 raw AUC | Qwen best B0 AUC | Raw-B0 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in qwen_note["qwen_l53"]:
        lines.append(
            f"| `{item['task']}` | {fmt(item['raw_auc'])} | {fmt(item['best_b0_auc'])} | {fmt(item['raw_minus_b0_auc'])} |"
        )
    context = qwen_note["gemma_subtype_context"]
    lines.extend(
        [
            "",
            "Gemma subtype context: "
            f"raw AUC={fmt(context['raw_auc'])}, best B0={fmt(context['best_b0_auc'])}, "
            f"raw-B0={fmt(context['raw_minus_b0_auc'])}. "
            f"{context['note']}",
            "",
            "## Control Status",
            "",
            "| Row | Gaussian control | Positive control | Confidence/effect correlation |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload["diagnostics"]:
        report = row["experiment_report"]
        diag = row["diagnostics"]
        steering = diag["raw_steering"]
        positive = diag["positive_control_gate"]
        corr = diag["probe_confidence_vs_steering_effect"]
        if steering.get("has_matched_gaussian_control"):
            gaussian = "yes"
        elif steering.get("status") == "available":
            gaussian = "no"
        else:
            gaussian = "untested"
        positive_text = positive.get("status", "not_configured")
        lines.append(
            f"| {row['model_display']} `{report['task']}` `{report['site_or_layer']}` | "
            f"{gaussian} | {positive_text} | {corr.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Gemma property L45 now has a matched-control raw-direction null; rows with no steering report remain predictive-only.",
            "- Qwen property steering pilots still need regenerated matched Gaussian/noise controls before causal interpretation.",
            "- Qwen L53 has the strongest activation-over-metadata margin, but its metadata-residualization proxy is currently L45.",
            "- Probe-confidence vs steering-effect correlation is now available for Gemma property L45; Qwen historical pilots and untested rows still need projection-enabled reruns if we want comparable causal diagnostics.",
            "- Do not claim `causally distributed` from this table; DAS/distributed interventions with passing controls are required.",
            "",
            "## Next Jobs",
            "",
        ]
    )
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=Path("docs/steering_effectiveness_diagnostics.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/steering_effectiveness_diagnostics.md"))
    args = parser.parse_args()

    payload = build_payload()
    write_json(args.json_output, payload)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(payload))
    print(args.json_output)
    print(args.markdown_output)


if __name__ == "__main__":
    main()
