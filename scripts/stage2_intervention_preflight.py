#!/usr/bin/env python3
"""Build the Part 1 intervention-preflight checklist/report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_causal_abstraction import now_utc  # noqa: E402
from src.stage2_probes import write_json  # noqa: E402


RAW_STEERING_REPORT = Path("docs/raw_steering_27b_l45_property_decode_sweep.json")
ERROR_STEERING_REPORT = Path("docs/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep.json")
ANSWER_PROPERTY_REPORT = Path("docs/answer_property_steering_27b_l45_polarity_smoke.json")
DASHBOARD = Path("docs/next_paper_causal_abstraction_dashboard.md")
SCHEMA = Path("docs/next_paper_causal_abstraction_schema.json")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def nested(payload: dict[str, Any] | None, keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def condition_labels(report: dict[str, Any] | None) -> list[str]:
    labels = []
    for row in nested(report, ["generation", "conditions"], []):
        if isinstance(row, dict) and row.get("label") is not None:
            labels.append(str(row["label"]))
    return labels


def has_condition_prefix(report: dict[str, Any] | None, prefix: str) -> bool:
    return any(label.startswith(prefix) for label in condition_labels(report))


def report_brief(path: Path) -> dict[str, Any]:
    report = read_json(path)
    if report is None:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "model": report.get("model"),
        "task": report.get("task"),
        "layer": report.get("layer"),
        "selection": report.get("selection", {}),
        "conditions": condition_labels(report),
        "baseline_n": nested(report, ["summary", "by_condition", "baseline", "n"]),
        "baseline_strong_accuracy": nested(report, ["summary", "by_condition", "baseline", "strong_accuracy"]),
        "baseline_parse_fail_rate": nested(report, ["summary", "by_condition", "baseline", "parse_fail_rate"]),
    }


def build_payload() -> dict[str, Any]:
    raw_report = read_json(RAW_STEERING_REPORT)
    error_report = read_json(ERROR_STEERING_REPORT)
    answer_report = read_json(ANSWER_PROPERTY_REPORT)
    historical_has_baseline = any(
        has_condition_prefix(report, "baseline") for report in (raw_report, error_report, answer_report)
    )
    historical_has_orthogonal = any(
        has_condition_prefix(report, "orthogonal") for report in (raw_report, error_report, answer_report)
    )

    gates = [
        {
            "gate": "current_evidence_freeze",
            "status": "branch_pushed_pending_merge_or_pr",
            "required_before_new_interpretation": True,
            "evidence": [
                str(DASHBOARD),
                "docs/steering_effectiveness_diagnostics.md",
                "docs/stage2_name_scramble_error_steering_plan.md",
            ],
            "next_action": "Merge or open a PR for codex/name-scramble-error-steering before treating the current evidence as frozen on main.",
        },
        {
            "gate": "regenerated_balanced_baseline",
            "status": "historical_small_n_available_refresh_required",
            "required_before_new_interpretation": True,
            "evidence": [
                report_brief(RAW_STEERING_REPORT),
                report_brief(ERROR_STEERING_REPORT),
            ],
            "passes_for_future_jobs": False,
            "next_action": "Regenerate balanced h3/h4 Gemma 3 27B infer_property baselines in the same job as each new intervention family; use paired regenerated rows, not historical labels, for flips.",
        },
        {
            "gate": "positive_control_steering",
            "status": "historical_answer_property_candidate_failed_new_control_required",
            "required_before_new_interpretation": True,
            "failed_candidate_script": "scripts/stage2_steer_answer_property_direction.py",
            "failed_candidate_job": "scripts/stage2_steer_answer_property_27b_L45_property_smoke.sbatch",
            "evidence": {
                "artifact": report_brief(ANSWER_PROPERTY_REPORT),
                "interpretation_doc": "docs/stage2_27b_answer_property_steering_results.md",
                "summary": "Existing 27B answer-property steering had a perfect offline polarity probe but no target-directed free-form answer movement, so it is not an accepted positive-control gate.",
                "replacement_spec": "docs/positive_control_steering_spec.md",
            },
            "passes_for_future_jobs": False,
            "next_action": "Implement the verbosity/output-format positive-control gate in docs/positive_control_steering_spec.md, using Gemma 3 27B L45 with baseline, orthogonal, and matched-Gaussian controls.",
        },
        {
            "gate": "orthogonal_direction_control",
            "status": "implemented_and_historical_available" if historical_has_orthogonal else "pending",
            "required_before_new_interpretation": True,
            "evidence": [str(RAW_STEERING_REPORT), str(ERROR_STEERING_REPORT), str(ANSWER_PROPERTY_REPORT)],
            "passes_for_future_jobs": historical_has_orthogonal,
            "next_action": "Keep orthogonal controls in every optimized, DAS, decode-time, and patch-validation intervention report where the method permits a direction control.",
        },
        {
            "gate": "matched_gaussian_noise_control",
            "status": "raw_direction_scaffolded_pending_gpu_rerun",
            "required_before_new_interpretation": True,
            "evidence": [
                "src/stage2_steering.py supports `gaussian` condition for raw-direction steering.",
                "scripts/stage2_steer_raw_direction.py saves `gaussian_direction` and can run `gaussian_*` conditions.",
                "scripts/stage2_steer_raw_27b_L45_property_decode_sweep.sbatch includes gaussian controls for future reruns.",
                "scripts/stage2_steer_error_27b_L45_property_decode_sweep.sbatch includes gaussian controls for future reruns.",
            ],
            "passes_for_future_jobs": False,
            "next_action": "Rerun the relevant Gemma 27B steering job with `gaussian` conditions before claiming repair exceeds matched noise.",
        },
        {
            "gate": "paired_flip_and_parse_reporting",
            "status": "schema_available_needs_enforcement_per_method",
            "required_before_new_interpretation": True,
            "evidence": [str(SCHEMA), "src/stage2_causal_abstraction.py"],
            "passes_for_future_jobs": False,
            "next_action": "Every new JSON report must include paired false-to-true/true-to-false flips, parse-fail rate, matched-noise summary, and a causal_abstraction_claim.",
        },
        {
            "gate": "exact_patch_validation",
            "status": "pending_for_attribution_rankings",
            "required_before_new_interpretation": False,
            "evidence": [],
            "passes_for_future_jobs": False,
            "next_action": "When AtP*/attribution rankings are added, validate top-ranked sites with exact patching before treating localization as causal evidence.",
        },
    ]

    return {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "report_kind": "part1_intervention_preflight",
        "status": "started",
        "objective": "Freeze current Gemma/Qwen evidence and build the preflight gate for interpretable causal-intervention tests.",
        "primary_model": "gemma3_27b",
        "primary_task": "infer_property",
        "primary_split": "s1_test_h3_h4_balanced",
        "gates": gates,
        "next_jobs": [
            {
                "priority": 1,
                "purpose": "Verify A40 availability before long jobs.",
                "command": "srun -A gpu --constraint=J --gres=gpu:1 --time=00:03:00 --ntasks=1 --cpus-per-task=1 --mem=12G --immediate=60 bash -lc 'hostname; nvidia-smi -L'",
            },
            {
                "priority": 2,
                "purpose": "Implement the true positive-control steering gate; the historical answer-property smoke did not pass.",
                "command": "see docs/positive_control_steering_spec.md",
            },
            {
                "priority": 3,
                "purpose": "Refresh raw-direction correctness steering with regenerated baseline, orthogonal, and matched-Gaussian controls.",
                "command": "sbatch scripts/stage2_steer_raw_27b_L45_property_decode_sweep.sbatch",
            },
            {
                "priority": 4,
                "purpose": "Refresh reconstruction-error steering with regenerated baseline, orthogonal, and matched-Gaussian controls if the raw preflight passes.",
                "command": "sbatch scripts/stage2_steer_error_27b_L45_property_decode_sweep.sbatch",
            },
        ],
        "interpretation_rule": "Do not interpret optimized vectors, DAS, decode-time correction, or AtP* nulls unless regenerated baselines, positive controls, orthogonal controls, and matched-noise controls are present for the relevant method.",
    }


def fmt_status(status: str) -> str:
    return status.replace("_", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Intervention Preflight",
        "",
        "Purpose: Part 1 gate for the next causal-intervention experiments.",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Objective",
        "",
        payload["objective"],
        "",
        "## Gates",
        "",
        "| Gate | Status | Required | Passes For Future Jobs | Next Action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gate in payload["gates"]:
        required = "yes" if gate.get("required_before_new_interpretation") else "method-specific"
        passes = "yes" if gate.get("passes_for_future_jobs") else "no"
        lines.append(
            f"| `{gate['gate']}` | {fmt_status(gate['status'])} | {required} | {passes} | {gate['next_action']} |"
        )

    lines.extend(
        [
            "",
            "## Next Jobs",
            "",
            "| Priority | Purpose | Command |",
            "| --- | --- | --- |",
        ]
    )
    for job in payload["next_jobs"]:
        lines.append(f"| {job['priority']} | {job['purpose']} | `{job['command']}` |")

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            payload["interpretation_rule"],
            "",
            "## Notes",
            "",
            "- Historical steering reports remain useful context but do not pass the full preflight because matched Gaussian/noise controls and a declared positive-control gate were not yet in place.",
            "- The answer-property steering artifact is a failed positive-control candidate, not a gate: it did not produce target-directed free-form answer movement.",
            "- Future Qwen comparisons should still label local dictionaries as local stand-ins, not first-party Qwen Scope artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=Path("docs/intervention_preflight.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/intervention_preflight.md"))
    args = parser.parse_args()

    payload = build_payload()
    write_json(args.json, payload)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(payload))
    print(args.json)
    print(args.markdown)


if __name__ == "__main__":
    main()
