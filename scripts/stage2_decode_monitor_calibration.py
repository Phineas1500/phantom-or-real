#!/usr/bin/env python3
"""Calibrate prompt-margin monitors for decode-time intervention gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def sanitize(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def checkpoint_rows(rows: list[dict[str, Any]], checkpoint: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("checkpoint")) == str(checkpoint)]


def margin_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    return float(value)


def trigger_metric(row: dict[str, Any], *, metric: str, operator: str, threshold: float) -> bool:
    value = margin_value(row, metric)
    if value is None:
        return False
    if operator == "lt":
        return value < threshold
    if operator == "le":
        return value <= threshold
    if operator == "gt":
        return value > threshold
    if operator == "ge":
        return value >= threshold
    raise ValueError(f"unknown operator {operator!r}")


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_candidate(
    *,
    rows: list[dict[str, Any]],
    label: str,
    metric: str,
    operator: str,
    threshold: float,
) -> dict[str, Any]:
    triggered = [row for row in rows if trigger_metric(row, metric=metric, operator=operator, threshold=threshold)]
    wrong = [row for row in rows if not bool(row.get("generated_is_correct_strong"))]
    correct = [row for row in rows if bool(row.get("generated_is_correct_strong"))]
    tp = sum(not bool(row.get("generated_is_correct_strong")) for row in triggered)
    fp = sum(bool(row.get("generated_is_correct_strong")) for row in triggered)
    fn = len(wrong) - tp
    tn = len(correct) - fp
    precision = tp / len(triggered) if triggered else None
    recall = tp / len(wrong) if wrong else None
    specificity = tn / len(correct) if correct else None
    return {
        "label": label,
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "n": len(rows),
        "triggered": len(triggered),
        "trigger_rate": len(triggered) / len(rows) if rows else None,
        "wrong_total": len(wrong),
        "correct_total": len(correct),
        "wrong_triggered": tp,
        "correct_triggered": fp,
        "wrong_missed": fn,
        "correct_not_triggered": tn,
        "precision_for_wrong": precision,
        "recall_for_wrong": recall,
        "specificity_for_correct": specificity,
        "parse_failed_triggered": sum(bool(row.get("parse_failed")) for row in triggered),
        "triggered_source_row_indices": [int(row["source_row_index"]) for row in triggered],
    }


def build_candidates(
    rows: list[dict[str, Any]],
    *,
    gold_thresholds: list[float],
    selected_thresholds: list[float],
) -> list[dict[str, Any]]:
    candidates = []
    for threshold in gold_thresholds:
        candidates.append(
            evaluate_candidate(
                rows=rows,
                label=f"gold_vs_foil_lt_{sanitize(threshold)}",
                metric="gold_vs_foil_logprob_margin",
                operator="lt",
                threshold=threshold,
            )
        )
    for threshold in selected_thresholds:
        candidates.append(
            evaluate_candidate(
                rows=rows,
                label=f"selected_vs_gold_ge_{sanitize(threshold)}",
                metric="selected_vs_gold_logprob_margin",
                operator="ge",
                threshold=threshold,
            )
        )
    return candidates


def choose_recommended_gate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    zero_fp = [row for row in candidates if row["correct_triggered"] == 0 and row["wrong_triggered"] > 0]
    gold_zero_fp = [row for row in zero_fp if row["metric"] == "gold_vs_foil_logprob_margin"]
    pool = gold_zero_fp or zero_fp
    if not pool:
        return max(candidates, key=lambda row: (row["precision_for_wrong"] or -1.0, row["wrong_triggered"]))
    return max(pool, key=lambda row: (row["wrong_triggered"], -(abs(float(row["threshold"])))) )


def summarize_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [float(row["gold_vs_foil_logprob_margin"]) for row in rows if row.get("gold_vs_foil_logprob_margin") is not None]
    selected = [float(row["selected_vs_gold_logprob_margin"]) for row in rows if row.get("selected_vs_gold_logprob_margin") is not None]
    by_correct = {}
    for is_correct in (False, True):
        subset = [row for row in rows if bool(row.get("generated_is_correct_strong")) is is_correct]
        by_correct[str(is_correct).lower()] = {
            "n": len(subset),
            "gold_vs_foil_logprob_margin": summarize_values(
                [float(row["gold_vs_foil_logprob_margin"]) for row in subset if row.get("gold_vs_foil_logprob_margin") is not None]
            ),
            "selected_vs_gold_logprob_margin": summarize_values(
                [float(row["selected_vs_gold_logprob_margin"]) for row in subset if row.get("selected_vs_gold_logprob_margin") is not None]
            ),
        }
    return {
        "n": len(rows),
        "generated_strong_accuracy": sum(bool(row.get("generated_is_correct_strong")) for row in rows) / len(rows) if rows else None,
        "parse_fail_rate": sum(bool(row.get("parse_failed")) for row in rows) / len(rows) if rows else None,
        "gold_vs_foil_logprob_margin": summarize_values(gold),
        "selected_vs_gold_logprob_margin": summarize_values(selected),
        "by_generated_is_correct_strong": by_correct,
    }


def raw_z_gate_summary(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    by_layer: dict[str, dict[str, Any]] = {}
    for layer in sorted({layer for row in rows for layer in (row.get("projection_traces") or {})}):
        layer_rows = []
        for row in rows:
            traces = (row.get("projection_traces") or {}).get(layer) or []
            prefill = [float(item["projection_z"]) for item in traces if item.get("phase") == "prefill"]
            decode = [float(item["projection_z"]) for item in traces if item.get("phase") == "decode"]
            layer_rows.append(
                {
                    "source_row_index": int(row["source_row_index"]),
                    "is_correct_strong": bool(row.get("is_correct_strong")),
                    "parse_failed": bool(row.get("parse_failed")),
                    "prefill_z": prefill[0] if prefill else None,
                    "decode_z_values": decode,
                }
            )
        threshold_rows = {}
        for threshold in thresholds:
            triggered_any = [row for row in layer_rows if any(value < threshold for value in row["decode_z_values"])]
            triggered_fraction = []
            for row in layer_rows:
                vals = row["decode_z_values"]
                if vals:
                    triggered_fraction.append(sum(value < threshold for value in vals) / len(vals))
            threshold_rows[f"zlt_{sanitize(threshold)}"] = {
                "threshold": threshold,
                "rows_triggered_any_decode": len(triggered_any),
                "wrong_triggered_any_decode": sum(not row["is_correct_strong"] for row in triggered_any),
                "correct_triggered_any_decode": sum(row["is_correct_strong"] for row in triggered_any),
                "mean_decode_trigger_fraction": float(np.mean(triggered_fraction)) if triggered_fraction else None,
            }
        by_layer[layer] = {
            "n": len(layer_rows),
            "prefill_z": summarize_values([row["prefill_z"] for row in layer_rows if row["prefill_z"] is not None]),
            "decode_z": summarize_values([value for row in layer_rows for value in row["decode_z_values"]]),
            "thresholds": threshold_rows,
        }
    return by_layer


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["recommended_gate"]
    lines = [
        "# Decode Monitor Calibration",
        "",
        f"Generated: `{report['created_at_utc']}`",
        "",
        "Purpose: calibrate a conservative prompt-margin gate for the next decode-time intervention, using the completed Gemma prefix-conditioned trajectory and Qwen comparison as predictive evidence.",
        "",
        "## Recommendation",
        "",
        (
            f"Use `{gate['metric']} {gate['operator']} {gate['threshold']}` as the first prompt-margin gated intervention trigger. "
            f"On the Gemma manifest checkpoint-0 rows it triggers `{gate['wrong_triggered']}/{gate['wrong_total']}` regenerated-wrong rows and "
            f"`{gate['correct_triggered']}/{gate['correct_total']}` regenerated-correct rows."
        ),
        "",
        "This is a planning gate, not a manuscript-level statistical claim. It is calibrated on only 14 Gemma recognition-gap rows and should be interpreted through matched raw, orthogonal, and Gaussian intervention outcomes.",
        "",
        "## Prompt-Margin Candidates",
        "",
        "| candidate | triggered | wrong triggered | correct triggered | precision | recall | specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["gemma_prompt_checkpoint0_candidates"]:
        lines.append(
            f"| `{row['label']}` | {row['triggered']}/{row['n']} | {row['wrong_triggered']}/{row['wrong_total']} | "
            f"{row['correct_triggered']}/{row['correct_total']} | {fmt(row['precision_for_wrong'])} | "
            f"{fmt(row['recall_for_wrong'])} | {fmt(row['specificity_for_correct'])} |"
        )
    lines += [
        "",
        "## Historical Raw-Z Gate Context",
        "",
        "The earlier raw-projection decode gate is retained as a negative calibration example: `z < 0` fired on nearly every decode trajectory and did not separate regenerated-correct from regenerated-wrong outputs.",
        "",
        "| layer | z gate | rows triggered | wrong triggered | correct triggered | mean decode trigger fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for layer, layer_summary in report["historical_raw_z_gate_summary"].items():
        for label, row in layer_summary["thresholds"].items():
            lines.append(
                f"| {layer} | `{label}` | {row['rows_triggered_any_decode']} | {row['wrong_triggered_any_decode']} | "
                f"{row['correct_triggered_any_decode']} | {fmt(row['mean_decode_trigger_fraction'])} |"
            )
    qwen = report.get("qwen_prompt_checkpoint0_summary") or {}
    lines += [
        "",
        "## Qwen Context",
        "",
        (
            f"Qwen is used as cross-model predictive support only: its h4 subtype subset has `selected>=gold` on "
            f"`{qwen.get('selected_vs_gold_nonnegative')}/{qwen.get('selected_available')}` rows and `gold>=foil` on "
            f"`{qwen.get('gold_vs_foil_nonnegative')}/{qwen.get('n')}` rows at checkpoint 0. It is not a matched intervention rowset."
        ),
        "",
        "## Next Run",
        "",
        "Run `scripts/stage2_prompt_margin_gated_decode_correction_27b_L45_property_manifest.sbatch` with the recommended threshold. Interpret a repair claim only if raw false-to-true repairs exceed matched Gaussian noise and orthogonal controls under the existing preflight criteria.",
        "",
        "## Causal-Abstraction Claim",
        "",
        report["causal_abstraction_claim"],
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-prefix-report", type=Path, default=Path("docs/prefix_conditioned_margin_trajectory_gemma_manifest.json"))
    parser.add_argument("--gemma-prefix-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--qwen-prefix-report", type=Path, default=Path("docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.json"))
    parser.add_argument("--projection-jsonl", type=Path, default=Path("results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap.jsonl"))
    parser.add_argument("--correction-report", type=Path, default=Path("docs/decode_time_correction_27b_l45_property_pilot.json"))
    parser.add_argument("--gold-thresholds", default="0,-1,-2,-5,-10,-15,-20")
    parser.add_argument("--selected-thresholds", default="0,1,2,5,10,15,20")
    parser.add_argument("--checkpoint", default="0")
    parser.add_argument("--output", type=Path, default=Path("docs/decode_monitor_calibration_gemma_qwen_prefix.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/decode_monitor_calibration_gemma_qwen_prefix.md"))
    return parser


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    args = build_arg_parser().parse_args()
    gemma_report = read_json(args.gemma_prefix_report)
    qwen_report = read_json(args.qwen_prefix_report) if args.qwen_prefix_report.exists() else None
    prefix_rows = checkpoint_rows(read_jsonl(args.gemma_prefix_jsonl), args.checkpoint)
    if not prefix_rows:
        raise ValueError(f"no prefix rows found for checkpoint {args.checkpoint!r}")
    candidates = build_candidates(
        prefix_rows,
        gold_thresholds=parse_float_list(args.gold_thresholds),
        selected_thresholds=parse_float_list(args.selected_thresholds),
    )
    recommended = choose_recommended_gate(candidates)
    projection_rows = read_jsonl(args.projection_jsonl) if args.projection_jsonl.exists() else []
    correction_report = read_json(args.correction_report) if args.correction_report.exists() else None
    qwen_c0 = None
    if qwen_report is not None:
        qwen_c0 = qwen_report["summary"]["by_checkpoint"]["0"]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/stage2_decode_monitor_calibration.py",
        "method": "decode_monitor_prompt_margin_calibration",
        "model": "google/gemma-3-27b-it",
        "model_key": "gemma3_27b",
        "task": "infer_property",
        "target_variable": "commitment_state",
        "representation_type": "decode_time_correction_state",
        "site_or_layer": "prompt_margin_gate_for_L45_decode_correction",
        "split": "S1",
        "controls": ["regenerated_baseline", "orthogonal_direction", "matched_gaussian_noise", "positive_control"],
        "n": len(prefix_rows),
        "baseline_metrics": gemma_report.get("baseline_metrics"),
        "intervention_metrics": {},
        "paired_flips": {},
        "parse_fail_rate": {"gemma_prefix_rows": sum(bool(row.get("parse_failed")) for row in prefix_rows) / len(prefix_rows)},
        "matched_noise_summary": {},
        "inputs": {
            "gemma_prefix_report": str(args.gemma_prefix_report),
            "gemma_prefix_jsonl": str(args.gemma_prefix_jsonl),
            "qwen_prefix_report": str(args.qwen_prefix_report),
            "projection_jsonl": str(args.projection_jsonl),
            "correction_report": str(args.correction_report),
        },
        "gemma_prompt_checkpoint0_summary": summarize_prompt_rows(prefix_rows),
        "gemma_prompt_checkpoint0_candidates": candidates,
        "recommended_gate": recommended,
        "historical_raw_z_gate_summary": raw_z_gate_summary(projection_rows, thresholds=[0.0, -1.0, -2.0]) if projection_rows else {},
        "historical_decode_time_correction_summary": {
            "slurm_job_id": correction_report.get("slurm_job_id") if correction_report else None,
            "paired_flips": correction_report.get("paired_flips") if correction_report else None,
            "parse_fail_rate": correction_report.get("parse_fail_rate") if correction_report else None,
            "generation": correction_report.get("generation") if correction_report else None,
        },
        "qwen_prompt_checkpoint0_summary": {
            "n": qwen_c0.get("n"),
            "selected_available": qwen_c0.get("selected_available"),
            "selected_vs_gold_nonnegative": qwen_c0.get("selected_vs_gold_nonnegative"),
            "gold_vs_foil_nonnegative": qwen_c0.get("gold_vs_foil_nonnegative"),
            "selected_vs_gold_mean": qwen_c0["selected_vs_gold_logprob_margin"]["mean"],
            "gold_vs_foil_mean": qwen_c0["gold_vs_foil_logprob_margin"]["mean"],
        } if qwen_c0 else None,
        "next_run": {
            "script": "scripts/stage2_prompt_margin_gated_decode_correction.py",
            "sbatch": "scripts/stage2_prompt_margin_gated_decode_correction_27b_L45_property_manifest.sbatch",
            "prompt_gold_vs_foil_threshold": recommended["threshold"],
            "conditions": "baseline,prompt_margin_raw,prompt_margin_orthogonal,prompt_margin_gaussian",
            "strengths": "1",
            "rowset": str(args.gemma_prefix_jsonl),
        },
        "causal_abstraction_claim": (
            "Predictive calibration only. It selects a prompt-margin gate over `gold_vs_foil_margin` "
            "for a future decode-time intervention on `commitment_state`; it does not itself intervene. "
            "A causal claim requires false-to-true repairs above orthogonal and matched-Gaussian controls."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_output.write_text(render_markdown(report) + "\n")
    print(f"recommended_gate={recommended['label']} wrong_triggered={recommended['wrong_triggered']} correct_triggered={recommended['correct_triggered']}")
    print(f"wrote {args.output}")
    print(f"wrote {args.md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
