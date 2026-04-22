"""Pre-handoff integrity + sanity checks per BEHAVIORAL_DATA_PLAN.md §"Sanity Checks"."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import read_jsonl  # noqa: E402


REQUIRED_FIELDS = {
    "example_id",
    "task",
    "height",
    "model",
    "prompt_text",
    "system_prompt",
    "ground_truth",
    "model_output",
    "is_correct_strong",
    "is_correct_weak",
    "quality_score",
    "parse_failed",
    "failure_mode",
    "error_type",
    "structural",
    "ontology_raw",
    "ontology_fol_string",
    "ontology_fol_structured",
}


def check_file(path: Path) -> dict:
    rows = read_jsonl(path)
    n = len(rows)
    issues = []

    # Integrity
    for i, r in enumerate(rows):
        missing = REQUIRED_FIELDS - set(r.keys())
        if missing:
            issues.append(f"row {i}: missing fields {missing}")
            break
        if not isinstance(r["is_correct_strong"], bool):
            issues.append(f"row {i}: is_correct_strong not bool")
            break
        if not isinstance(r["parse_failed"], bool):
            issues.append(f"row {i}: parse_failed not bool")
            break

    # Per-height stats
    heights: dict[int, list] = {}
    parse_fail_by_h = Counter()
    fm_by_h: dict[int, Counter] = {}
    for r in rows:
        heights.setdefault(r["height"], []).append(r)
        if r["parse_failed"]:
            parse_fail_by_h[r["height"]] += 1
            fm = r["failure_mode"]
            fm_by_h.setdefault(r["height"], Counter())[fm] += 1

    per_height = {}
    for h, hrows in sorted(heights.items()):
        strong = sum(1 for r in hrows if r["is_correct_strong"])
        parse_fail = parse_fail_by_h[h]
        per_height[h] = {
            "n": len(hrows),
            "positive_n": strong,
            "negative_n": len(hrows) - strong,
            "strong_accuracy": strong / len(hrows) if hrows else None,
            "parse_fail_rate": parse_fail / len(hrows) if hrows else None,
            "failure_modes": dict(fm_by_h.get(h, {})),
        }

    warnings = []
    if per_height:
        max_h = max(per_height)
        # Monotonic decrease check for infer_property strong accuracy
        task = rows[0]["task"]
        if task == "infer_property":
            heights_sorted = sorted(per_height)
            accs = [per_height[h]["strong_accuracy"] for h in heights_sorted]
            for a, b in zip(accs, accs[1:]):
                if a is not None and b is not None and b > a + 0.02:
                    warnings.append(f"non-monotonic strong accuracy: {accs}")
                    break

        # h=1 should be >70%
        if 1 in per_height and per_height[1]["strong_accuracy"] is not None:
            if per_height[1]["strong_accuracy"] < 0.70:
                warnings.append(f"h=1 strong accuracy <0.70: {per_height[1]['strong_accuracy']:.3f}")

        # parse_fail_rate < 5% per (height)
        for h, stats in per_height.items():
            pr = stats["parse_fail_rate"]
            if pr is not None and pr > 0.05:
                warnings.append(f"h={h} parse_fail_rate {pr:.3f} > 0.05")

    # has_direct_member rate check
    hdm = sum(1 for r in rows if r["structural"]["has_direct_member"]) / n if n else None

    return {
        "file": str(path),
        "n": n,
        "task": rows[0]["task"] if rows else None,
        "model": rows[0]["model"] if rows else None,
        "per_height": per_height,
        "has_direct_member_rate": hdm,
        "integrity_issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", nargs="+", type=Path, required=True)
    args = p.parse_args()

    out = {str(j): check_file(j) for j in args.jsonl}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
