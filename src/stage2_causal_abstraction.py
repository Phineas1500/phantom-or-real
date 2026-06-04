"""Shared causal-abstraction schema for next-paper Stage 2 experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .stage2_probes import write_json


CAUSAL_VARIABLES = (
    "target_concept",
    "candidate_hypothesis",
    "relation_or_property",
    "proof_depth",
    "selected_hypothesis",
    "gold_vs_foil_margin",
    "free_form_correctness",
    "recognition_correctness",
    "commitment_state",
)

AUXILIARY_TARGET_VARIABLES = (
    "positive_control_behavior",
)


REPRESENTATION_TYPES = (
    "raw_direction",
    "sparse_feature",
    "reconstruction",
    "error_subspace",
    "das_subspace",
    "patched_residual_state",
    "decode_time_correction_state",
)

RESULT_TYPES = ("predictive", "causal", "predictive_and_causal")

CONTROL_TYPES = (
    "regenerated_baseline",
    "orthogonal_direction",
    "matched_gaussian_noise",
    "positive_control",
    "exact_patch_validation",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def schema_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "causal_variables": list(CAUSAL_VARIABLES),
        "auxiliary_target_variables": list(AUXILIARY_TARGET_VARIABLES),
        "representation_types": list(REPRESENTATION_TYPES),
        "result_types": list(RESULT_TYPES),
        "control_types": list(CONTROL_TYPES),
        "required_report_fields": [
            "model",
            "task",
            "target_variable",
            "split",
            "site_or_layer",
            "method",
            "representation_type",
            "result_type",
            "controls",
            "n",
            "baseline_metrics",
            "intervention_metrics",
            "paired_flips",
            "parse_fail_rate",
            "matched_noise_summary",
            "causal_abstraction_claim",
        ],
        "operational_thresholds": {
            "repair_over_noise_sigma": 2.0,
            "min_paired_repairs": 3,
            "dictionary_raw_auc_gap_practically_closed": 0.03,
            "dictionary_error_auc_reduction_signal": 0.05,
            "status": "planning_heuristics_not_manuscript_claims",
        },
    }


def make_experiment_report(
    *,
    model: str,
    task: str,
    target_variable: str,
    split: str,
    site_or_layer: str,
    method: str,
    representation_type: str,
    result_type: str,
    controls: list[str],
    n: int | None = None,
    baseline_metrics: dict[str, Any] | None = None,
    intervention_metrics: dict[str, Any] | None = None,
    paired_flips: dict[str, Any] | None = None,
    parse_fail_rate: float | None = None,
    matched_noise_summary: dict[str, Any] | None = None,
    causal_abstraction_claim: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    validate_choice("target_variable", target_variable, CAUSAL_VARIABLES + AUXILIARY_TARGET_VARIABLES)
    validate_choice("representation_type", representation_type, REPRESENTATION_TYPES)
    validate_choice("result_type", result_type, RESULT_TYPES)
    for control in controls:
        validate_choice("control", control, CONTROL_TYPES)
    return {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "model": model,
        "task": task,
        "target_variable": target_variable,
        "split": split,
        "site_or_layer": site_or_layer,
        "method": method,
        "representation_type": representation_type,
        "result_type": result_type,
        "controls": controls,
        "n": n,
        "baseline_metrics": baseline_metrics or {},
        "intervention_metrics": intervention_metrics or {},
        "paired_flips": paired_flips or {},
        "parse_fail_rate": parse_fail_rate,
        "matched_noise_summary": matched_noise_summary or {},
        "causal_abstraction_claim": causal_abstraction_claim,
        "notes": notes or [],
    }


def validate_choice(field: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field}={value!r} is not one of {sorted(allowed)}")


def write_schema(path: Path) -> None:
    write_json(path, schema_payload())
