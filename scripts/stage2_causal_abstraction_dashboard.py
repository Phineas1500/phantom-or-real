#!/usr/bin/env python3
"""Build the next-paper causal-abstraction dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_causal_abstraction import (  # noqa: E402
    CAUSAL_VARIABLES,
    CONTROL_TYPES,
    REPRESENTATION_TYPES,
    write_schema,
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def nested(payload: dict[str, Any] | None, path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_existing_evidence() -> list[dict[str, Any]]:
    namescramble = read_json(Path("docs/namescramble_27b_l45_raw_probe_s1.json"))
    error_steer = read_json(Path("docs/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep.json"))
    qwen_forced_choice = read_json(Path("docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json"))
    qwen_raw_l53 = read_json(Path("docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s1.json"))

    qwen_l53_property = nested(qwen_raw_l53, ["results", "infer_property", "L53", "test_auc"])
    qwen_l53_subtype = nested(qwen_raw_l53, ["results", "infer_subtype", "L53", "test_auc"])
    qwen_fc_n = nested(qwen_forced_choice, ["summary", "overall", "n"])
    qwen_fc_acc = nested(qwen_forced_choice, ["summary", "overall", "mcq_choice_accuracy"])
    qwen_fc_correct = round(qwen_fc_acc * qwen_fc_n) if qwen_fc_acc is not None and qwen_fc_n is not None else None

    return [
        {
            "claim": "Gemma raw correctness readout survives full name-scrambling with measurable loss.",
            "variable": "free_form_correctness",
            "representation": "raw_direction",
            "status": "completed",
            "evidence": (
                "fixed original L45 probes on scrambled activations: "
                f"property natural={fmt(nested(namescramble, ['results', 'infer_property', 'conditions', 'natural', 'scrambled', 'auc']))}, "
                f"property nonce={fmt(nested(namescramble, ['results', 'infer_property', 'conditions', 'nonce', 'scrambled', 'auc']))}, "
                f"subtype natural={fmt(nested(namescramble, ['results', 'infer_subtype', 'conditions', 'natural', 'scrambled', 'auc']))}, "
                f"subtype nonce={fmt(nested(namescramble, ['results', 'infer_subtype', 'conditions', 'nonce', 'scrambled', 'auc']))}"
            ),
        },
        {
            "claim": "Gemma reconstruction-error direction is predictive but not a decode-step repair handle.",
            "variable": "free_form_correctness",
            "representation": "error_subspace",
            "status": "completed",
            "evidence": (
                f"error-direction test AUC={fmt(nested(error_steer, ['probe_direction', 'test_auc']))}; "
                "paired strong flips=0 at all tested error-direction strengths"
            ),
        },
        {
            "claim": "Qwen strengthens activation-over-metadata readout and exposes a raw-vs-LAP gap.",
            "variable": "free_form_correctness",
            "representation": "raw_direction",
            "status": "completed",
            "evidence": (
                f"Qwen L53 S1 raw AUC property={fmt(qwen_l53_property)}, subtype={fmt(qwen_l53_subtype)}; "
                "metadata baselines are much weaker than raw readouts, while LAP margins are near chance"
            ),
        },
        {
            "claim": "Recognition-vs-generation holds across models but is not a matched replication.",
            "variable": "recognition_correctness",
            "representation": "patched_residual_state",
            "status": "completed",
            "evidence": (
                "Gemma: 14/16 property h3/h4 hard-foil rows; "
                f"Qwen: {fmt(qwen_fc_correct)}/{fmt(qwen_fc_n)} subtype h4 hard-foil rows"
            ),
        },
    ]


def planned_tracks() -> list[dict[str, str]]:
    return [
        {
            "track": "Shared causal-abstraction model",
            "next_step": "Use the shared variables and report schema for every new experiment.",
            "success": "Every report names target variable, representation, and predictive/causal status.",
        },
        {
            "track": "Steering-effectiveness diagnostics",
            "next_step": "Use completed LAP/logit-lens and metadata-adjusted summaries to choose the next intervention family.",
            "success": "Dashboard records which correctness directions are linearly readable but not directly logit-accessible.",
        },
        {
            "track": "Stronger interventions",
            "next_step": "Add optimized vectors, DAS/distributed interchange, decode-time correction, and AtP* ranking with exact patch validation.",
            "success": "Repairs exceed matched noise by 2 sigma and at least 3 paired false-to-true examples, or nulls have passing positive controls.",
        },
        {
            "track": "Dictionary/dark-matter tests",
            "next_step": "Audit Gemma e2e/KL/Matryoshka/BatchTopK availability, then train local Gemma dictionaries if needed.",
            "success": "Determine whether predictive signal moves from error into reconstruction in the right basis/objective.",
        },
        {
            "track": "Commitment and recognition trajectory",
            "next_step": "Probe and patch selected hypothesis, margin, and commitment variables across decode positions.",
            "success": "Identify whether Gemma reverse disruption localizes to a commitment transition and whether Qwen lacks the same transition.",
        },
        {
            "track": "Target and OOD extensions",
            "next_step": "Run weak/quality-score, name-scramble, and height-extrapolation probe variants.",
            "success": "Separate correctness, parsimony, depth/difficulty, and name-familiarity components.",
        },
    ]


def track_reports() -> list[dict[str, str]]:
    steering = read_json(Path("docs/steering_effectiveness_diagnostics.json"))
    steering_status = "pending"
    if steering is not None:
        steering_status = nested(steering, ["scope", "status"], "available")

    intervention = read_json(Path("docs/intervention_preflight.json"))
    intervention_status = "pending"
    if intervention is not None:
        intervention_status = intervention.get("status", "available")

    return [
        {
            "track": "Steering-effectiveness diagnostics",
            "status": steering_status,
            "artifact": "docs/steering_effectiveness_diagnostics.md",
            "note": "Artifact preflight over existing probe, metadata, and historical steering reports.",
        },
        {
            "track": "Intervention preflight",
            "status": intervention_status,
            "artifact": "docs/intervention_preflight.md",
            "note": "Part 1 gate for regenerated baselines, positive controls, matched noise, paired flips, and parse-failure reporting.",
        },
    ]


def render_dashboard() -> str:
    evidence = build_existing_evidence()
    tracks = planned_tracks()
    reports = track_reports()
    lines = [
        "# Next-Paper Causal-Abstraction Dashboard",
        "",
        "Purpose: track the program testing whether InAbHyD correctness is linearly readable, sparsely lossy, and possibly causally distributed.",
        "",
        "## Shared Causal Variables",
        "",
        ", ".join(f"`{name}`" for name in CAUSAL_VARIABLES),
        "",
        "## Representation Types",
        "",
        ", ".join(f"`{name}`" for name in REPRESENTATION_TYPES),
        "",
        "## Required Controls",
        "",
        ", ".join(f"`{name}`" for name in CONTROL_TYPES),
        "",
        "## Existing Evidence",
        "",
        "| Claim | Variable | Representation | Status | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in evidence:
        lines.append(
            f"| {row['claim']} | `{row['variable']}` | `{row['representation']}` | {row['status']} | {row['evidence']} |"
        )

    lines.extend(
        [
            "",
            "## Track Reports",
            "",
            "| Track | Status | Artifact | Note |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in reports:
        lines.append(f"| {row['track']} | {row['status']} | {row['artifact']} | {row['note']} |")

    lines.extend(
        [
            "",
            "## Planned Tracks",
            "",
            "| Track | Next step | Success condition |",
            "| --- | --- | --- |",
        ]
    )
    for row in tracks:
        lines.append(f"| {row['track']} | {row['next_step']} | {row['success']} |")

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `causally distributed` is a hypothesis, not a settled result.",
            "- If DAS/distributed interventions fail with passing positive controls, use `causally inaccessible under tested methods`.",
            "- Treat `0.03` raw-AUC gap and `0.05` error-AUC reduction as planning heuristics until explicitly approved for manuscript use.",
            "- Keep Qwen local MLP/transcoder/crosscoder dictionaries labeled as local stand-ins, not first-party Qwen Scope artifacts.",
            "- Record model, task, height, row-selection rule, and foil type for every recognition-vs-generation result.",
            "- Verify new 2026 citations from primary sources before adding them to manuscript text.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, default=Path("docs/next_paper_causal_abstraction_dashboard.md"))
    parser.add_argument("--schema", type=Path, default=Path("docs/next_paper_causal_abstraction_schema.json"))
    args = parser.parse_args()

    write_schema(args.schema)
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.dashboard.write_text(render_dashboard())
    print(args.schema)
    print(args.dashboard)


if __name__ == "__main__":
    main()
