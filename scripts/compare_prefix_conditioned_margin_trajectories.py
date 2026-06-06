#!/usr/bin/env python3
"""Compare Gemma and Qwen prefix-conditioned margin trajectory reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def row_metadata(jsonl_path: Path) -> dict[str, Any]:
    if not jsonl_path.exists():
        return {"jsonl": str(jsonl_path), "exists": False}
    rows = read_jsonl(jsonl_path)
    by_source = {}
    for row in rows:
        by_source.setdefault(int(row["source_row_index"]), row)
    first_rows = list(by_source.values())
    return {
        "jsonl": str(jsonl_path),
        "exists": True,
        "n_rows": len(first_rows),
        "n_trajectory_rows": len(rows),
        "tasks": dict(Counter(row.get("task") for row in first_rows)),
        "heights": dict(Counter(str(row.get("height")) for row in first_rows)),
        "models": dict(Counter(row.get("model_key") for row in first_rows)),
        "parse_failures": dict(Counter(str(row.get("parse_failed", row.get("original_parse_failed"))) for row in first_rows)),
    }


def checkpoint_row(report: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    return report["summary"]["by_checkpoint"][checkpoint]


def report_summary(
    *,
    name: str,
    report: dict[str, Any],
    jsonl_path: Path,
    selected_definition: str,
    comparison_caveat: str,
) -> dict[str, Any]:
    c0 = checkpoint_row(report, "0")
    final = checkpoint_row(report, "final")
    return {
        "name": name,
        "model_key": report.get("model_key"),
        "task": report.get("task"),
        "n_rows": report.get("n_rows"),
        "n_trajectory_rows": report.get("n"),
        "row_metadata": row_metadata(jsonl_path),
        "baseline_metrics": report.get("baseline_metrics", {}),
        "selected_definition": selected_definition,
        "comparison_caveat": comparison_caveat,
        "checkpoint_0": {
            "selected_available": c0.get("selected_available"),
            "selected_vs_gold_mean": c0["selected_vs_gold_logprob_margin"]["mean"],
            "selected_vs_gold_nonnegative": c0.get("selected_vs_gold_nonnegative"),
            "gold_vs_foil_mean": c0["gold_vs_foil_logprob_margin"]["mean"],
            "gold_vs_foil_nonnegative": c0.get("gold_vs_foil_nonnegative"),
            "n": c0.get("n"),
        },
        "final": {
            "selected_available": final.get("selected_available"),
            "selected_vs_gold_mean": final["selected_vs_gold_logprob_margin"]["mean"],
            "selected_vs_gold_nonnegative": final.get("selected_vs_gold_nonnegative"),
            "gold_vs_foil_mean": final["gold_vs_foil_logprob_margin"]["mean"],
            "gold_vs_foil_nonnegative": final.get("gold_vs_foil_nonnegative"),
            "n": final.get("n"),
        },
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    gemma, qwen = comparison["models"]
    lines = [
        "# Gemma/Qwen Prefix-Conditioned Margin Comparison",
        "",
        f"Generated: `{comparison['created_at_utc']}`",
        "",
        "Purpose: compare the completed prefix-conditioned margin diagnostics for Gemma and Qwen on recognition-gap rows.",
        "",
        "## Bottom Line",
        "",
        comparison["bottom_line"],
        "",
        "## Prompt-Only Checkpoint",
        "",
        "| model | task / heights | selected definition | selected>=gold | selected-vs-gold mean | gold>=foil | gold-vs-foil mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in (gemma, qwen):
        heights = ",".join(sorted(item["row_metadata"].get("heights", {}).keys()))
        c0 = item["checkpoint_0"]
        lines.append(
            f"| {item['name']} | {item['task']} h={heights} | {item['selected_definition']} | "
            f"{c0['selected_vs_gold_nonnegative']}/{c0['selected_available']} | "
            f"{fmt(c0['selected_vs_gold_mean'])} | "
            f"{c0['gold_vs_foil_nonnegative']}/{c0['n']} | "
            f"{fmt(c0['gold_vs_foil_mean'])} |"
        )

    lines += [
        "",
        "## Final Prefix Checkpoint",
        "",
        "| model | selected>=gold | selected-vs-gold mean | gold>=foil | gold-vs-foil mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in (gemma, qwen):
        final = item["final"]
        lines.append(
            f"| {item['name']} | "
            f"{final['selected_vs_gold_nonnegative']}/{final['selected_available']} | "
            f"{fmt(final['selected_vs_gold_mean'])} | "
            f"{final['gold_vs_foil_nonnegative']}/{final['n']} | "
            f"{fmt(final['gold_vs_foil_mean'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Both models show the key prompt-only signature: the free-form selected or hard-foil hypothesis is already preferred to gold before any generated prefix is added.",
        "- This makes the prefix result a stronger recognition-vs-generation diagnostic than a clean commitment-transition localization. The wrong-hypothesis preference is present at checkpoint 0.",
        "- Gemma and Qwen are not matched replications. Gemma is property h3/h4 with regenerated decode traces and one parse failure; Qwen is subtype h4 with original Stage 1 output prefixes and selected defined as the hard foil.",
        "- Later-prefix behavior should be interpreted cautiously. In Qwen, selected is identical to the hard foil, so selected-vs-gold is the negative of gold-vs-foil by construction.",
        "",
        "## Recommendation",
        "",
        comparison["recommendation"],
        "",
        "## Causal-Abstraction Claim",
        "",
        comparison["causal_abstraction_claim"],
        "",
        "## Inputs",
        "",
        f"- Gemma report: `{comparison['inputs']['gemma_report']}`",
        f"- Qwen report: `{comparison['inputs']['qwen_report']}`",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-report", type=Path, default=Path("docs/prefix_conditioned_margin_trajectory_gemma_manifest.json"))
    parser.add_argument("--qwen-report", type=Path, default=Path("docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.json"))
    parser.add_argument(
        "--gemma-jsonl",
        type=Path,
        default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"),
    )
    parser.add_argument(
        "--qwen-jsonl",
        type=Path,
        default=Path("results/stage2/decode_time/qwen_prefix_conditioned_margin_trajectory_h4_subset.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/prefix_conditioned_margin_trajectory_comparison_gemma_qwen.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/prefix_conditioned_margin_trajectory_comparison_gemma_qwen.md"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    gemma_report = read_json(args.gemma_report)
    qwen_report = read_json(args.qwen_report)
    comparison = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/compare_prefix_conditioned_margin_trajectories.py",
        "inputs": {
            "gemma_report": str(args.gemma_report),
            "qwen_report": str(args.qwen_report),
            "gemma_jsonl": str(args.gemma_jsonl),
            "qwen_jsonl": str(args.qwen_jsonl),
        },
        "models": [
            report_summary(
                name="Gemma 3 27B",
                report=gemma_report,
                jsonl_path=args.gemma_jsonl,
                selected_definition="regenerated selected hypothesis",
                comparison_caveat="property h3/h4 regenerated decode trace; one parse-failed row lacks selected hypothesis",
            ),
            report_summary(
                name="Qwen3.5 27B",
                report=qwen_report,
                jsonl_path=args.qwen_jsonl,
                selected_definition="Stage 1 hard foil",
                comparison_caveat="subtype h4 original Stage 1 output prefixes; selected equals hard foil by construction",
            ),
        ],
        "bottom_line": (
            "The prefix-conditioned diagnostics support a cross-model predictive pattern: on recognition-gap rows, "
            "the wrong/free-form hypothesis is already more likely than gold at the prompt-only checkpoint. "
            "This is evidence for a recognition-vs-generation deployment gap, not evidence that we have found "
            "a causal commitment-transition handle."
        ),
        "recommendation": (
            "Close the prefix-conditioned trajectory measurement as predictive evidence for now. Do not run another "
            "broad trajectory batch unless it is tied to a specific intervention. If we continue this track causally, "
            "the next experiment should first train or calibrate a decode-trajectory monitor on selected-vs-gold or "
            "gold-vs-foil margin state, then test a gated intervention against matched-noise and positive controls."
        ),
        "causal_abstraction_claim": (
            "Predictive only. The reports test `selected_hypothesis`, `gold_vs_foil_margin`, and `commitment_state` "
            "as readouts under prefix-conditioned contexts; they do not intervene on the model state."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    args.md_output.write_text(render_markdown(comparison) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
