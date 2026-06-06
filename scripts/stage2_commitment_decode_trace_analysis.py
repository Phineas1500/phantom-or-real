#!/usr/bin/env python3
"""Analyze manifest-selected Gemma decode traces for commitment signals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


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


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "fraction_below_0": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "fraction_below_0": float(np.mean(arr < 0.0)),
    }


def trace_values(row: dict[str, Any], layer: str, phase: str) -> list[float]:
    trace = row.get("projection_traces", {}).get(layer, [])
    return [float(point["projection_z"]) for point in trace if point.get("phase") == phase]


def row_trace_summary(row: dict[str, Any], layer: str) -> dict[str, Any]:
    prefill = trace_values(row, layer, "prefill")
    decode = trace_values(row, layer, "decode")
    return {
        "prefill_z_mean": float(np.mean(prefill)) if prefill else None,
        "first_decode_z": decode[0] if decode else None,
        "last_decode_z": decode[-1] if decode else None,
        "decode_z_mean": float(np.mean(decode)) if decode else None,
        "decode_z_min": float(np.min(decode)) if decode else None,
        "decode_z_max": float(np.max(decode)) if decode else None,
        "decode_z_fraction_below_0": float(np.mean(np.asarray(decode) < 0.0)) if decode else None,
    }


def by_bool(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "false": [row for row in rows if bool(row.get(key)) is False],
        "true": [row for row in rows if bool(row.get(key)) is True],
    }


def safe_margin(row: dict[str, Any], key: str) -> float | None:
    value = row.get("prompt_margin_scores", {}).get(key)
    return float(value) if value is not None else None


def margin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "gold_vs_foil_logprob_margin",
        "gold_vs_foil_mean_logprob_margin",
        "selected_vs_gold_logprob_margin",
        "selected_vs_foil_logprob_margin",
        "gold_vs_selected_logprob_margin",
    ]
    out = {}
    for key in keys:
        out[key] = summarize_values([safe_margin(row, key) for row in rows if safe_margin(row, key) is not None])
    out["counts"] = {
        "gold_vs_foil_below_0": sum(
            1 for row in rows if (safe_margin(row, "gold_vs_foil_logprob_margin") is not None and safe_margin(row, "gold_vs_foil_logprob_margin") < 0)
        ),
        "selected_vs_gold_ge_0": sum(
            1 for row in rows if (safe_margin(row, "selected_vs_gold_logprob_margin") is not None and safe_margin(row, "selected_vs_gold_logprob_margin") >= 0)
        ),
        "selected_margin_available": sum(1 for row in rows if safe_margin(row, "selected_vs_gold_logprob_margin") is not None),
    }
    return out


def layer_summary(rows: list[dict[str, Any]], layers: list[str]) -> dict[str, Any]:
    out = {}
    for layer in layers:
        all_decode = []
        first_decode = []
        last_decode = []
        prefill = []
        for row in rows:
            decode = trace_values(row, layer, "decode")
            pre = trace_values(row, layer, "prefill")
            all_decode.extend(decode)
            first_decode.extend(decode[:1])
            if decode:
                last_decode.append(decode[-1])
            prefill.extend(pre)
        out[layer] = {
            "prefill_z": summarize_values(prefill),
            "decode_z": summarize_values(all_decode),
            "first_decode_z": summarize_values(first_decode),
            "last_decode_z": summarize_values(last_decode),
            "by_generated_is_correct_strong": {},
        }
        for label, subset in by_bool(rows, "is_correct_strong").items():
            subset_decode = [value for row in subset for value in trace_values(row, layer, "decode")]
            subset_first = []
            subset_last = []
            for row in subset:
                decode = trace_values(row, layer, "decode")
                subset_first.extend(decode[:1])
                if decode:
                    subset_last.append(decode[-1])
            out[layer]["by_generated_is_correct_strong"][label] = {
                "rows": len(subset),
                "decode_z": summarize_values(subset_decode),
                "first_decode_z": summarize_values(subset_first),
                "last_decode_z": summarize_values(subset_last),
            }
    return out


def build_row_summaries(rows: list[dict[str, Any]], layers: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        recognition = row.get("recognition") or {}
        margin = row.get("prompt_margin_scores") or {}
        row_summary = {
            "source_row_index": int(row["source_row_index"]),
            "example_id": row.get("example_id"),
            "height": row.get("height"),
            "generated_is_correct_strong": bool(row.get("is_correct_strong")),
            "generated_is_correct_weak": bool(row.get("is_correct_weak")),
            "parse_failed": bool(row.get("parse_failed")),
            "quality_score": row.get("quality_score"),
            "original_margin_gold_minus_foil": recognition.get("original_margin_gold_minus_foil"),
            "mcq_margin_gold_minus_foil": recognition.get("mcq_margin_gold_minus_foil"),
            "prompt_gold_vs_foil_logprob_margin": margin.get("gold_vs_foil_logprob_margin"),
            "prompt_selected_vs_gold_logprob_margin": margin.get("selected_vs_gold_logprob_margin"),
            "prompt_selected_vs_foil_logprob_margin": margin.get("selected_vs_foil_logprob_margin"),
            "selected_hypothesis": margin.get("selected_hypothesis"),
            "selected_hypothesis_count": margin.get("selected_hypothesis_count"),
            "layer_trace": {layer: row_trace_summary(row, layer) for layer in layers},
        }
        out.append(row_summary)
    return sorted(out, key=lambda row: (int(row["height"]), int(row["source_row_index"])))


def build_report(report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    layers = [f"L{layer}" for layer in report.get("layers", [])]
    if not layers and rows:
        layers = sorted(rows[0].get("projection_traces", {}).keys())
    generated_counts = Counter(bool(row.get("is_correct_strong")) for row in rows)
    parse_fail_count = sum(bool(row.get("parse_failed")) for row in rows)
    height_counts: dict[str, dict[str, Any]] = {}
    for height in sorted({row.get("height") for row in rows}):
        subset = [row for row in rows if row.get("height") == height]
        height_counts[f"h{height}"] = {
            "n": len(subset),
            "strong_accuracy": sum(bool(row.get("is_correct_strong")) for row in subset) / len(subset),
            "parse_fail_rate": sum(bool(row.get("parse_failed")) for row in subset) / len(subset),
        }

    row_summaries = build_row_summaries(rows, layers)
    margins = margin_summary(rows)
    layers_out = layer_summary(rows, layers)

    claim = (
        "The manifest-selected Gemma rows preserve the recognition-vs-generation gap: "
        "forced-choice recognition was correct by construction, but regenerated free-form "
        "strong correctness remained low. Prompt-trained correctness projections stay "
        "negative through decoding and do not distinguish regenerated-correct rows. "
        "Prompt gold-vs-foil scoring remains wrong on almost all rows, while the selected "
        "hypothesis is at least as prompt-likely as gold on every parsed row; this points "
        "toward testing prefix-conditioned margins rather than reusing the prompt-trained z gate."
    )
    next_job = {
        "name": "prefix_conditioned_margin_trajectory_gemma_manifest",
        "model": report.get("model"),
        "model_key": report.get("model_key"),
        "task": report.get("task"),
        "input_rows": "same 14 rows from docs/commitment_rowset_manifest.json with recognition_generation_gap",
        "candidate_continuations": ["gold_hypothesis", "hard_foil_hypothesis", "selected_hypothesis"],
        "prefix_checkpoints": [0, 1, 4, 8, 16, 32, 64, "final"],
        "metrics": [
            "gold_vs_foil_logprob_margin_after_prefix",
            "selected_vs_gold_logprob_margin_after_prefix",
            "selected_vs_foil_logprob_margin_after_prefix",
            "first_checkpoint_where_selected_exceeds_gold",
            "first_checkpoint_where_gold_exceeds_foil",
        ],
        "interpretation": (
            "If selected-vs-gold becomes positive early and stays positive on wrong rows, "
            "that is a stronger commitment-state signature than prompt-trained raw-probe z. "
            "If margins only flip late, commitment may be output-trajectory-local rather than pre-generation."
        ),
    }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": report.get("script"),
        "source_report_json": "docs/decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap.json",
        "source_trace_jsonl": "results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap.jsonl",
        "model": report.get("model"),
        "model_key": report.get("model_key"),
        "task": report.get("task"),
        "target_variable": "commitment_state",
        "representation_type": "raw_direction_and_prompt_margin",
        "method": "row_level_commitment_decode_trace_analysis",
        "n": len(rows),
        "baseline_metrics": {
            "strong_accuracy": generated_counts[True] / len(rows) if rows else None,
            "weak_accuracy": sum(bool(row.get("is_correct_weak")) for row in rows) / len(rows) if rows else None,
            "parse_fail_rate": parse_fail_count / len(rows) if rows else None,
            "generated_strong_correct": generated_counts[True],
            "generated_strong_incorrect": generated_counts[False],
            "parse_fail_count": parse_fail_count,
            "by_height": height_counts,
        },
        "margin_summary": margins,
        "layer_summary": layers_out,
        "row_summaries": row_summaries,
        "next_job_recommendation": next_job,
        "causal_abstraction_claim": claim,
    }


def render_md(analysis: dict[str, Any]) -> str:
    baseline = analysis["baseline_metrics"]
    margins = analysis["margin_summary"]
    lines = [
        "# Gemma Manifest Decode Commitment Analysis",
        "",
        f"Generated: `{analysis['generated_at']}`",
        "",
        "Purpose: analyze the completed manifest-selected Gemma decode trace run at row level before launching another GPU job.",
        "",
        "## Summary",
        "",
        f"- Rows: `{analysis['n']}` free-form-wrong, hard-foil-recognition-correct Gemma property rows.",
        f"- Regenerated strong accuracy: `{baseline['generated_strong_correct']}/{analysis['n']}` ({pct(baseline['strong_accuracy'])}); parse failures: `{baseline['parse_fail_count']}/{analysis['n']}`.",
        f"- Prompt gold-vs-foil margin stayed below zero on `{margins['counts']['gold_vs_foil_below_0']}/{analysis['n']}` rows.",
        f"- Selected-vs-gold prompt margin was nonnegative on `{margins['counts']['selected_vs_gold_ge_0']}/{margins['counts']['selected_margin_available']}` parsed rows.",
        "- L45/L53 prompt-trained correctness z stayed strongly negative through decoding for both regenerated-correct and regenerated-wrong rows.",
        "",
        "Interpretation: this is stronger evidence that the prompt-trained correctness direction is not a decode-time commitment monitor. The useful next measurement is prefix-conditioned margin scoring: ask when the generated/selected hypothesis becomes more likely than gold as the output prefix accumulates.",
        "",
        "## Aggregate Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| strong accuracy | {pct(baseline['strong_accuracy'])} |",
        f"| weak accuracy | {pct(baseline['weak_accuracy'])} |",
        f"| parse fail rate | {pct(baseline['parse_fail_rate'])} |",
        f"| gold-vs-foil margin mean | {fmt(margins['gold_vs_foil_logprob_margin']['mean'])} |",
        f"| gold-vs-foil below 0 | {margins['counts']['gold_vs_foil_below_0']}/{analysis['n']} |",
        f"| selected-vs-gold margin mean | {fmt(margins['selected_vs_gold_logprob_margin']['mean'])} |",
        f"| selected-vs-gold nonnegative | {margins['counts']['selected_vs_gold_ge_0']}/{margins['counts']['selected_margin_available']} |",
        "",
        "## Decode Projection Summary",
        "",
        "| layer | decode z mean | z<0 | wrong-row mean | correct-row mean | first decode mean | last decode mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for layer, summary in analysis["layer_summary"].items():
        wrong = summary["by_generated_is_correct_strong"]["false"]["decode_z"]["mean"]
        correct = summary["by_generated_is_correct_strong"]["true"]["decode_z"]["mean"]
        lines.append(
            f"| {layer} | {fmt(summary['decode_z']['mean'])} | {pct(summary['decode_z']['fraction_below_0'])} | {fmt(wrong)} | {fmt(correct)} | {fmt(summary['first_decode_z']['mean'])} | {fmt(summary['last_decode_z']['mean'])} |"
        )

    lines += [
        "",
        "## Row-Level Pattern",
        "",
        "| row | example | h | strong | parse fail | gold-vs-foil | selected-vs-gold | L45 mean z | L53 mean z |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in analysis["row_summaries"]:
        l45 = row["layer_trace"].get("L45", {})
        l53 = row["layer_trace"].get("L53", {})
        lines.append(
            f"| {row['source_row_index']} | {row['example_id']} | {row['height']} | {row['generated_is_correct_strong']} | {row['parse_failed']} | {fmt(row['prompt_gold_vs_foil_logprob_margin'])} | {fmt(row['prompt_selected_vs_gold_logprob_margin'])} | {fmt(l45.get('decode_z_mean'))} | {fmt(l53.get('decode_z_mean'))} |"
        )

    next_job = analysis["next_job_recommendation"]
    lines += [
        "",
        "## Next Job Recommendation",
        "",
        f"Run `{next_job['name']}` before a Qwen decode trace. Use the same 14 Gemma manifest rows and score `gold_hypothesis`, `hard_foil_hypothesis`, and `selected_hypothesis` after generated-prefix checkpoints `{next_job['prefix_checkpoints']}`.",
        "",
        "Primary readout: the first prefix checkpoint where selected-vs-gold becomes positive, and whether it stays positive on regenerated-wrong rows. This directly tests `commitment_state`; the current prompt-trained z readout does not.",
        "",
        "## Causal-Abstraction Claim",
        "",
        analysis["causal_abstraction_claim"],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap.json"))
    parser.add_argument("--jsonl", type=Path, default=Path("results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap.jsonl"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/commitment_decode_trace_analysis_gemma_manifest.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/commitment_decode_trace_analysis_gemma_manifest.md"))
    args = parser.parse_args()

    report = read_json(args.report)
    rows = read_jsonl(args.jsonl)
    analysis = build_report(report, rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_md(analysis) + "\n")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Rows: {analysis['n']}")
    print(f"Strong accuracy: {analysis['baseline_metrics']['strong_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
