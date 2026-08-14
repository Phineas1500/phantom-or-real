#!/usr/bin/env python3
"""Item L' pooled verdict (docs/causal_handle_directions.md item L'):
verbatim baseline gate vs item-L1, pooled PRIMARY branch, within-row
protocol contrast vs L1's recorded gold branches. Writes
docs/selfaddress_prime_27b_property_pooled.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIME = [
    "results/stage2/erasure/selfaddress_prime_27b_property_shard0of2.jsonl",
    "results/stage2/erasure/selfaddress_prime_27b_property_shard1of2.jsonl",
]
L1 = [
    "results/stage2/erasure/selfaddress_27b_property_shard0of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard1of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard2of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard3of8.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard7of8.jsonl",
]
OUT = Path("docs/selfaddress_prime_27b_property_pooled.json")


def ci(deltas, rng, n_boot=10000):
    draws = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    p_base_text, p_base, p_fix = {}, defaultdict(list), defaultdict(list)
    parse = defaultdict(lambda: [0, 0])
    for path in PRIME:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                row, cond = r["source_row_index"], r["condition"]
                parse[cond][0] += 1
                parse[cond][1] += int(bool(r.get("parse_failed")))
                if cond == "unhinted_baseline":
                    p_base_text[(row, r["sample_index"])] = r["model_output"]
                    p_base[row].append(int(bool(r["is_correct_strong"])))
                else:
                    p_fix[row].append(int(bool(r["is_correct_strong"])))

    l1_base_text, l1_gold = {}, defaultdict(list)
    for path in L1:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["condition"] == "unhinted_baseline":
                    l1_base_text[(r["source_row_index"], r["sample_index"])] = r["model_output"]
                elif r["condition"].startswith("percand_fire") and r.get("fired_is_gold"):
                    l1_gold[r["source_row_index"]].append(int(bool(r["is_correct_strong"])))

    shared = sorted(set(p_base_text) & set(l1_base_text))
    mismatch = [k for k in shared if p_base_text[k] != l1_base_text[k]]
    gate = {
        "n_shared_generations": len(shared),
        "n_prime_baseline_generations": len(p_base_text),
        "text_mismatches": len(mismatch),
        "pass": len(mismatch) == 0 and len(shared) == len(p_base_text),
    }

    rows = sorted(set(p_base) & set(p_fix))
    rng = np.random.default_rng(20260704)
    base_rate = {r: float(np.mean(p_base[r])) for r in rows}
    fix_rate = {r: float(np.mean(p_fix[r])) for r in rows}
    deltas = np.array([fix_rate[r] - base_rate[r] for r in rows])
    lo, hi = ci(deltas, rng)
    primary = {
        "n_rows": len(rows),
        "baseline_p_strong": float(np.mean([base_rate[r] for r in rows])),
        "fixednorm_p_strong": float(np.mean([fix_rate[r] for r in rows])),
        "dp": float(np.mean(deltas)), "ci95": [lo, hi],
        "branch": "TRANSFERS" if lo > 0 else ("NEGATIVE" if hi < 0 else "NULL"),
        "pct_of_guard_anchor_0447": float(np.mean(deltas)) / 0.447,
    }

    contrast_rows = sorted(set(rows) & set(l1_gold))
    cdeltas = np.array([fix_rate[r] - float(np.mean(l1_gold[r])) for r in contrast_rows])
    clo, chi = ci(cdeltas, rng)
    contrast = {
        "n_rows": len(contrast_rows),
        "delta": float(np.mean(cdeltas)), "ci95": [clo, chi],
        "sign": "pos" if clo > 0 else ("neg" if chi < 0 else "straddle"),
        "note": "L' fixednorm (k=8) minus L1 recorded gold-branch fire (k=4), per-row means",
    }

    parse_out = {c: {"n": n, "parse_fail": pf / n,
                     "status": "pass" if pf / n < 0.05 else ("VOID" if pf / n > 0.20 else "flag")}
                 for c, (n, pf) in sorted(parse.items())}

    report = {"gate_baseline_verbatim": gate, "primary": primary,
              "within_row_protocol_contrast": contrast, "parse": parse_out}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
