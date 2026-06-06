#!/usr/bin/env python3
"""Build a commitment/recognition trajectory preflight from existing artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "id": "gemma_hardfoil_forced_choice",
        "category": "recognition",
        "path": "docs/answer_property_margins_27b_l45_polarity_hardfoil.json",
        "jsonl": "results/stage2/steering/answer_property_margins_27b_l45_polarity_hardfoil.jsonl",
    },
    {
        "id": "qwen_h4_hardfoil_forced_choice",
        "category": "recognition",
        "path": "docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json",
        "jsonl": "results/stage2/qwen_causal/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.jsonl",
    },
    {
        "id": "gemma_forward_patching",
        "category": "patching",
        "path": "docs/clean_to_corrupt_patching_27b_property_margin_pilot.json",
        "jsonl": "results/stage2/patching/clean_to_corrupt_27b_property_margin_pilot.jsonl",
    },
    {
        "id": "gemma_reverse_patching",
        "category": "patching",
        "path": "docs/corrupt_to_clean_patching_27b_property_margin_pilot.json",
        "jsonl": "results/stage2/patching/corrupt_to_clean_27b_property_margin_pilot.jsonl",
    },
    {
        "id": "qwen_forward_patching",
        "category": "patching",
        "path": "docs/qwen35_27b_infer_subtype_clean_to_corrupt_patching_margin_pilot.json",
        "jsonl": "results/stage2/patching/qwen35_27b_infer_subtype_clean_to_corrupt_margin_pilot.jsonl",
    },
    {
        "id": "qwen_reverse_patching",
        "category": "patching",
        "path": "docs/qwen35_27b_infer_subtype_corrupt_to_clean_patching_margin_pilot.json",
        "jsonl": "results/stage2/patching/qwen35_27b_infer_subtype_corrupt_to_clean_margin_pilot.jsonl",
    },
    {
        "id": "gemma_l45_das_forward",
        "category": "das",
        "path": "docs/das_subspace_27b_l45_property_clean_to_corrupt_pilot.json",
        "jsonl": "results/stage2/das/das_subspace_27b_l45_property_clean_to_corrupt_pilot.jsonl",
    },
    {
        "id": "gemma_l45_das_reverse",
        "category": "das",
        "path": "docs/das_subspace_27b_l45_property_corrupt_to_clean_pilot.json",
        "jsonl": "results/stage2/das/das_subspace_27b_l45_property_corrupt_to_clean_pilot.jsonl",
    },
    {
        "id": "gemma_l50_das_reverse",
        "category": "das",
        "path": "docs/das_subspace_27b_l50_last_prompt_property_corrupt_to_clean_pilot.json",
        "jsonl": "results/stage2/das/das_subspace_27b_l50_last_prompt_property_corrupt_to_clean_pilot.jsonl",
    },
    {
        "id": "gemma_l50_atp_reverse",
        "category": "atp",
        "path": "docs/atp_rank_validate_27b_l50_last_prompt_property_corrupt_to_clean_pilot.json",
        "jsonl": "results/stage2/atp/atp_rank_validate_27b_l50_last_prompt_property_corrupt_to_clean_pilot.jsonl",
    },
    {
        "id": "gemma_decode_trace",
        "category": "decode_trace",
        "path": "docs/decode_projection_trace_27b_l45_l53_property_pilot.json",
        "jsonl": "results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_pilot.jsonl",
    },
    {
        "id": "gemma_decode_correction",
        "category": "decode_correction",
        "path": "docs/decode_time_correction_27b_l45_property_pilot.json",
        "jsonl": "results/stage2/decode_time/decode_time_correction_27b_l45_property_pilot.jsonl",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def jsonl_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def artifact_inventory(repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in ARTIFACTS:
        path = repo_root / spec["path"]
        jsonl = repo_root / spec["jsonl"]
        data = read_json(path) if path.exists() else {}
        out[spec["id"]] = {
            **spec,
            "exists": path.exists(),
            "jsonl_exists": jsonl.exists(),
            "jsonl_rows": jsonl_count(jsonl),
            "model_key": data.get("model_key"),
            "task": data.get("task"),
            "target_variable": data.get("target_variable"),
            "representation_type": data.get("representation_type"),
            "n": data.get("n"),
            "layers": data.get("layers") or ([data["layer"]] if "layer" in data else None),
            "selection": data.get("selection"),
        }
    return out


def recognition_summary(repo_root: Path) -> list[dict[str, Any]]:
    gemma = read_json(repo_root / "docs/answer_property_margins_27b_l45_polarity_hardfoil.json")
    qwen = read_json(repo_root / "docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json")
    gemma_base = gemma["summary"]["by_condition"]["baseline"]
    qwen_overall = qwen["summary"]["overall"]
    return [
        {
            "model": "gemma3_27b",
            "task": "infer_property",
            "height": "h3/h4",
            "row_selection": "S1 test, parsed, free-form strong-incorrect, balanced by height and answer polarity",
            "foil_type": "stage1_model_output_hard_foil",
            "n": gemma_base["n"],
            "recognition_accuracy": gemma_base["mcq_choice_accuracy"],
            "parse_fail_rate": gemma_base["mcq_parse_fail_rate"],
            "mean_original_margin": gemma_base["mean_original_margin"],
            "mean_mcq_margin": gemma_base["mean_mcq_margin"],
            "claim": "recognition often intact despite free-form wrong output; not a matched comparison to Qwen",
        },
        {
            "model": "qwen35_27b",
            "task": "infer_subtype",
            "height": "h4",
            "row_selection": "S1 test, parsed, free-form strong-incorrect, first 64 by seed",
            "foil_type": "stage1_model_output_hard_foil",
            "n": qwen_overall["n"],
            "recognition_accuracy": qwen_overall["mcq_choice_accuracy"],
            "parse_fail_rate": qwen_overall["mcq_parse_fail_rate"],
            "mean_original_margin": qwen_overall["mean_original_margin"],
            "mean_mcq_margin": qwen_overall["mean_mcq_margin"],
            "claim": "same recognition-vs-generation theme, but task/height/row selection differ from Gemma",
        },
    ]


def patch_row(report: dict[str, Any], *, model: str, direction: str, mode: str, layer: int) -> dict[str, Any]:
    key = f"{mode}_L{layer}_last_prompt"
    summary = report["summary"].get(key, {})
    return {
        "model": model,
        "task": report.get("task"),
        "direction": direction,
        "layer": layer,
        "landmark": "last_prompt",
        "mode": mode,
        "n": summary.get("n"),
        "mean_margin_delta": summary.get("mean_margin_delta"),
        "mean_recovery_fraction": summary.get("mean_recovery_fraction"),
        "mean_breakage_fraction": summary.get("mean_breakage_fraction"),
        "positive_recovery_count": summary.get("positive_recovery_count"),
        "positive_breakage_count": summary.get("positive_breakage_count"),
        "above_0p25_recovery_count": summary.get("above_0p25_recovery_count"),
        "above_0p25_breakage_count": summary.get("above_0p25_breakage_count"),
    }


def patching_summary(repo_root: Path) -> list[dict[str, Any]]:
    gemma_forward = read_json(repo_root / "docs/clean_to_corrupt_patching_27b_property_margin_pilot.json")
    gemma_reverse = read_json(repo_root / "docs/corrupt_to_clean_patching_27b_property_margin_pilot.json")
    qwen_forward = read_json(repo_root / "docs/qwen35_27b_infer_subtype_clean_to_corrupt_patching_margin_pilot.json")
    qwen_reverse = read_json(repo_root / "docs/qwen35_27b_infer_subtype_corrupt_to_clean_patching_margin_pilot.json")

    rows = []
    for layer in (35, 40, 45, 50):
        rows.append(patch_row(gemma_forward, model="gemma3_27b", direction="h1_to_h4", mode="clean", layer=layer))
        rows.append(patch_row(gemma_forward, model="gemma3_27b", direction="h1_to_h4", mode="noise", layer=layer))
        rows.append(patch_row(gemma_reverse, model="gemma3_27b", direction="h4_to_h1", mode="corrupt", layer=layer))
        rows.append(patch_row(gemma_reverse, model="gemma3_27b", direction="h4_to_h1", mode="noise", layer=layer))
    for layer in (35, 40, 45):
        rows.append(patch_row(qwen_forward, model="qwen35_27b", direction="h1_to_h4", mode="clean", layer=layer))
        rows.append(patch_row(qwen_forward, model="qwen35_27b", direction="h1_to_h4", mode="noise", layer=layer))
        rows.append(patch_row(qwen_reverse, model="qwen35_27b", direction="h4_to_h1", mode="corrupt", layer=layer))
        rows.append(patch_row(qwen_reverse, model="qwen35_27b", direction="h4_to_h1", mode="noise", layer=layer))
    return rows


def das_atp_summary(repo_root: Path) -> list[dict[str, Any]]:
    specs = [
        ("gemma_l45_das_forward", "docs/das_subspace_27b_l45_property_clean_to_corrupt_pilot.json", "das_subspace_L45_last_prompt_r4", "matched_gaussian_L45_last_prompt_r4"),
        ("gemma_l45_das_reverse", "docs/das_subspace_27b_l45_property_corrupt_to_clean_pilot.json", "das_subspace_L45_last_prompt_r4", "matched_gaussian_L45_last_prompt_r4"),
        ("gemma_l50_das_reverse", "docs/das_subspace_27b_l50_last_prompt_property_corrupt_to_clean_pilot.json", "das_subspace_L50_last_prompt_r4", "matched_gaussian_L50_last_prompt_r4"),
    ]
    rows = []
    for artifact_id, path, key, noise_key in specs:
        report = read_json(repo_root / path)
        metric = report["intervention_metrics"][key]
        noise = report["matched_noise_summary"][noise_key]
        rows.append(
            {
                "artifact": artifact_id,
                "method": "DAS-style low-rank interchange",
                "n": metric.get("n"),
                "mean_margin_delta": metric.get("mean_margin_delta"),
                "mean_breakage_fraction": metric.get("mean_breakage_fraction"),
                "mean_recovery_fraction": metric.get("mean_recovery_fraction"),
                "noise_mean_margin_delta": noise.get("mean_margin_delta"),
                "mean_delta_vs_noise_sigma": metric.get("mean_delta_vs_noise_sigma"),
                "false_to_true": metric.get("false_to_true_repair_count"),
                "true_to_false": metric.get("true_to_false_disruption_count"),
            }
        )
    atp = read_json(repo_root / "docs/atp_rank_validate_27b_l50_last_prompt_property_corrupt_to_clean_pilot.json")
    exact = atp["intervention_metrics"]["exact_validation"]["source_L50_last_prompt"]
    noise = atp["matched_noise_summary"]["matched_gaussian_L50_last_prompt"]
    rows.append(
        {
            "artifact": "gemma_l50_atp_reverse",
            "method": "AtP-style ranking plus exact patch validation",
            "n": exact.get("n"),
            "mean_margin_delta": exact.get("mean_margin_delta"),
            "mean_breakage_fraction": exact.get("mean_breakage_fraction"),
            "mean_recovery_fraction": exact.get("mean_recovery_fraction"),
            "noise_mean_margin_delta": noise.get("mean_margin_delta"),
            "mean_delta_vs_noise_sigma": exact.get("mean_delta_vs_noise_sigma"),
            "false_to_true": exact.get("false_to_true_repair_count"),
            "true_to_false": exact.get("true_to_false_disruption_count"),
        }
    )
    return rows


def decode_summary(repo_root: Path) -> list[dict[str, Any]]:
    trace = read_json(repo_root / "docs/decode_projection_trace_27b_l45_l53_property_pilot.json")
    correction = read_json(repo_root / "docs/decode_time_correction_27b_l45_property_pilot.json")
    rows = []
    for layer_key, layer_summary in trace["summary"].items():
        rows.append(
            {
                "artifact": "gemma_decode_trace",
                "model": "gemma3_27b",
                "task": "infer_property",
                "layer": layer_key,
                "n": trace["n"],
                "baseline_accuracy": layer_summary.get("baseline_strong_accuracy"),
                "decode_z_mean": layer_summary["all_decode_z"]["mean"],
                "decode_z_fraction_below_0": layer_summary["all_decode_z"]["fraction_below_0"],
                "prefill_z_mean": layer_summary.get("all_prefill_z", {}).get("mean"),
                "claim": "prompt-trained z threshold does not separate regenerated-correct and regenerated-wrong decode trajectories",
            }
        )
    for condition, flips in correction["paired_flips"].items():
        rows.append(
            {
                "artifact": "gemma_decode_time_correction",
                "model": "gemma3_27b",
                "task": "infer_property",
                "layer": "L45",
                "condition": condition,
                "n": flips["paired_n"],
                "false_to_true": flips["false_to_true"],
                "true_to_false": flips["true_to_false"],
                "changed": flips["changed"],
                "claim": "conditional decode-time injection fired but did not repair correctness",
            }
        )
    return rows


def build_report(repo_root: Path) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track": "commitment_recognition_trajectory_preflight",
        "causal_variables": [
            "selected_hypothesis",
            "gold_vs_foil_margin",
            "recognition_correctness",
            "commitment_state",
            "free_form_correctness",
        ],
        "artifact_inventory": artifact_inventory(repo_root),
        "recognition_summary": recognition_summary(repo_root),
        "patching_summary": patching_summary(repo_root),
        "das_atp_summary": das_atp_summary(repo_root),
        "decode_summary": decode_summary(repo_root),
        "gaps": [
            "Existing Gemma and Qwen recognition runs support the same recognition-vs-generation theme but are not matched by model, task, height, or foil distribution.",
            "Existing decode traces only cover Gemma property, 8 rows, L45/L53, and a prompt-trained correctness projection; there is no Qwen decode trace yet.",
            "Existing patching is margin-first, not free-form generation repair; discrete false-to-true generation effects remain untested for localized commitment sites.",
            "Gemma forward/reverse patching uses strict natural h1/h4 pairs sharing full gold hypotheses, not exact same-ontology cross-height pairs.",
            "DAS-style and AtP-style runs localize weak margin effects but do not isolate a successful low-rank causal repair handle.",
            "No single canonical row set currently ties together free-form wrong rows, forced-choice recognition, decode trajectories, and patching pairs.",
        ],
        "recommended_next_steps": [
            {
                "priority": 1,
                "step": "Build a canonical commitment row-set manifest.",
                "details": "Record model, task, height, source row, foil type, free-form correctness, recognition result, and patch-pair membership for Gemma and Qwen.",
            },
            {
                "priority": 2,
                "step": "Extend decode trajectory measurement to gold-vs-foil and selected-hypothesis margins.",
                "details": "Use the canonical manifest; run Gemma property first, then Qwen subtype if the measurement is informative.",
            },
            {
                "priority": 3,
                "step": "Only then consider new patching/generation jobs.",
                "details": "A new GPU job should target a specific commitment transition, not repeat broad patching scans.",
            },
        ],
    }
    return report


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Commitment/Recognition Trajectory Preflight",
        "",
        f"Generated: `{report['created_at_utc']}`",
        "",
        "Purpose: prepare the commitment/recognition track before launching new GPU jobs. This preflight inventories existing recognition, patching, DAS/AtP, and decode-time artifacts and identifies the missing row/trajectory data needed for the next experiment.",
        "",
        "## Current Interpretation",
        "",
        "Existing evidence supports a recognition-vs-generation theme and a Gemma-specific forward-null/reverse-disruption asymmetry. It does not yet localize a clean `commitment_state` transition or a causal repair handle. The next work should unify row selection and trajectory variables before more intervention jobs.",
        "",
        "## Recognition Evidence",
        "",
    ]
    recognition_rows = []
    for row in report["recognition_summary"]:
        recognition_rows.append(
            [
                row["model"],
                row["task"],
                row["height"],
                str(row["n"]),
                row["foil_type"],
                pct(row["recognition_accuracy"]),
                fmt(row["mean_original_margin"]),
                fmt(row["mean_mcq_margin"]),
            ]
        )
    lines.append(
        md_table(
            ["model", "task", "height", "n", "foil", "recognition acc.", "orig margin", "MCQ margin"],
            recognition_rows,
        )
    )
    lines.extend(
        [
            "",
            "Note: Gemma `14/16` and Qwen `43/64` support the same theme but are not matched replications.",
            "",
            "## Natural Patching Snapshot",
            "",
        ]
    )
    patch_rows = []
    for row in report["patching_summary"]:
        if row["landmark"] != "last_prompt":
            continue
        if row["model"] == "gemma3_27b" and row["layer"] not in (35, 40, 45, 50):
            continue
        if row["model"] == "qwen35_27b" and row["layer"] not in (35, 40, 45):
            continue
        patch_rows.append(
            [
                row["model"],
                row["task"],
                row["direction"],
                f"L{row['layer']}",
                row["mode"],
                str(row.get("n")),
                fmt(row.get("mean_recovery_fraction")),
                fmt(row.get("mean_breakage_fraction")),
                fmt(row.get("mean_margin_delta")),
            ]
        )
    lines.append(
        md_table(
            ["model", "task", "direction", "layer", "mode", "n", "recovery", "breakage", "margin delta"],
            patch_rows,
        )
    )
    lines.extend(["", "## DAS/AtP Snapshot", ""])
    das_rows = []
    for row in report["das_atp_summary"]:
        das_rows.append(
            [
                row["artifact"],
                row["method"],
                str(row.get("n")),
                fmt(row.get("mean_margin_delta")),
                fmt(row.get("mean_breakage_fraction")),
                fmt(row.get("mean_delta_vs_noise_sigma")),
                str(row.get("false_to_true")),
                str(row.get("true_to_false")),
            ]
        )
    lines.append(
        md_table(
            ["artifact", "method", "n", "margin delta", "breakage", "vs noise sigma", "F->T", "T->F"],
            das_rows,
        )
    )
    lines.extend(["", "## Decode Trajectory Snapshot", ""])
    decode_rows = []
    for row in report["decode_summary"]:
        if row["artifact"] == "gemma_decode_trace":
            decode_rows.append(
                [
                    row["artifact"],
                    row["layer"],
                    str(row["n"]),
                    pct(row["baseline_accuracy"]),
                    fmt(row["decode_z_mean"]),
                    pct(row["decode_z_fraction_below_0"]),
                    fmt(row.get("prefill_z_mean")),
                    row["claim"],
                ]
            )
        else:
            decode_rows.append(
                [
                    f"{row['artifact']}:{row.get('condition', 'NA')}",
                    row["layer"],
                    str(row["n"]),
                    "NA",
                    "NA",
                    "NA",
                    f"F->T {row['false_to_true']}, T->F {row['true_to_false']}, changed {row['changed']}",
                    row["claim"],
                ]
            )
    lines.append(
        md_table(
            ["artifact", "layer", "n", "baseline acc.", "decode z mean", "z<0", "prefill/flip summary", "claim"],
            decode_rows,
        )
    )
    lines.extend(["", "## Missing Pieces", ""])
    for gap in report["gaps"]:
        lines.append(f"- {gap}")
    lines.extend(["", "## Recommended Next Steps", ""])
    for step in report["recommended_next_steps"]:
        lines.append(f"{step['priority']}. {step['step']} {step['details']}")
    lines.extend(
        [
            "",
            "## Artifact Inventory",
            "",
        ]
    )
    inventory_rows = []
    for artifact_id, artifact in sorted(report["artifact_inventory"].items()):
        inventory_rows.append(
            [
                artifact_id,
                artifact["category"],
                artifact.get("model_key") or "NA",
                artifact.get("task") or "NA",
                str(artifact.get("n")),
                str(artifact.get("jsonl_rows")),
                artifact["path"],
            ]
        )
    lines.append(
        md_table(
            ["artifact", "category", "model", "task", "report n", "jsonl rows", "path"],
            inventory_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Causal-Abstraction Claim",
            "",
            "This preflight is diagnostic only. It organizes existing evidence for `selected_hypothesis`, `gold_vs_foil_margin`, `recognition_correctness`, and `commitment_state`; it does not add a new causal intervention result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=Path("docs/commitment_trajectory_preflight.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/commitment_trajectory_preflight.md"))
    args = parser.parse_args()

    report = build_report(args.repo_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(report))
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
