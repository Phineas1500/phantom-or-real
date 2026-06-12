#!/usr/bin/env python3
"""Concept-resolved, row-paired analysis of hint-state interchange outputs.

Scores each generation by which concept its emitted hypotheses target
(hypothesis-subject match, not substring mention), so the wrong-donor arm is
tested on the metric that can actually detect misdirection: P(any emitted
hypothesis targets the hinted wrong concept). All arm comparisons are paired
within row with row-level cluster bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_steer_answer_property_margins import (  # noqa: E402
    hypothesis_parts_from_text,
    singularize_margin_name,
)
from src.gemma3_parse import parse_hypotheses  # noqa: E402


def subjects_of(reply: str) -> set[str]:
    out = set()
    for hyp in parse_hypotheses(reply or ""):
        parts = hypothesis_parts_from_text(hyp)
        if parts.subject:
            out.add(parts.subject.lower())
    return out


def canon(concept: str) -> str:
    return (singularize_margin_name(concept) or concept).lower()


def row_bootstrap_ci(deltas: np.ndarray, rng, n_boot: int = 10000) -> tuple[float, float]:
    boots = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/stage2/erasure/hint_state_interchange_27b_property_manifest.jsonl"))
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--output", type=Path, default=Path("docs/hint_state_interchange_27b_property_concept_analysis.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = [json.loads(line) for line in args.jsonl.open() if line.strip()]
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        subjects = subjects_of(r["model_output"])
        per[r["condition"]][int(r["source_row_index"])].append(
            {
                "strong": bool(r["is_correct_strong"]),
                "targets_gold": canon(r["gold_concept"]) in subjects,
                "targets_wrong": canon(r["wrong_concept"]) in subjects,
            }
        )

    def row_rates(cond: str, key: str) -> dict[int, float]:
        return {i: float(np.mean([s[key] for s in v])) for i, v in per[cond].items()}

    metrics = ("strong", "targets_gold", "targets_wrong")
    base = {key: row_rates("baseline", key) for key in metrics}
    idx = sorted(base["strong"])
    report_conditions: dict[str, Any] = {}
    for cond in sorted(per):
        entry: dict[str, Any] = {"n_rows": len(per[cond]), "n_generations": sum(len(v) for v in per[cond].values())}
        for key in metrics:
            rates = row_rates(cond, key)
            entry[f"{key}_rate"] = float(np.mean([rates[i] for i in sorted(rates)]))
            if cond != "baseline":
                deltas = np.array([rates[i] - base[key][i] for i in idx if i in rates])
                lo, hi = row_bootstrap_ci(deltas, rng)
                entry[f"{key}_paired_delta"] = float(np.mean(deltas))
                entry[f"{key}_paired_ci95"] = [lo, hi]
        report_conditions[cond] = entry
        line = f"{cond}: strong={entry['strong_rate']:.3f} gold_tgt={entry['targets_gold_rate']:.3f} wrong_tgt={entry['targets_wrong_rate']:.3f}"
        if cond != "baseline":
            ci = entry["targets_wrong_paired_ci95"]
            line += f" | d_wrong_tgt={entry['targets_wrong_paired_delta']:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(line, flush=True)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/stage2_interchange_concept_analysis.py",
        "jsonl": str(args.jsonl),
        "method": "concept_resolved_row_paired_analysis",
        "target_variable": "target_concept",
        "metric_note": (
            "targets_X = any emitted hypothesis whose parsed subject equals concept X "
            "(singularized); replaces the loose substring mention metric. Misdirection "
            "through activations requires patch_hint_wrong to raise targets_wrong over "
            "baseline with a row-paired CI excluding zero."
        ),
        "conditions": report_conditions,
        "n": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
