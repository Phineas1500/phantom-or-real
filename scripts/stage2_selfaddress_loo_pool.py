#!/usr/bin/env python3
"""Item L-double-prime pooled verdict: the composition (gauge-selected branch x
in-job LOO write) over the recorded C0..C3 shards, with the registered
comparators (docs/causal_handle_directions.md, Self-Addressing Battery).

Baseline verbatim gate: the fresh in-job unhinted baselines must reproduce the
L1-recorded baselines token-for-token on shared (row, sample) pairs (row-keyed
seeds make this exact by construction). The best-of-N majority comparator is
the L1-recorded matched_bestofN_unsteered arm (cross-job reuse registered).
Writes docs/selfaddress_loo_27b_property_pooled.json.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

LOO_JSONLS = [f"results/stage2/erasure/selfaddress_loo_27b_property_shard{i}of4.jsonl" for i in range(4)]
L1_JSONLS = [
    "results/stage2/erasure/selfaddress_27b_property_shard0of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard1of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard2of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard3of8.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard7of8.jsonl",
]
OUT = Path("docs/selfaddress_loo_27b_property_pooled.json")


def ci(deltas: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    draws = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def sign(lo: float, hi: float) -> str:
    return "neg" if hi < 0 else ("pos" if lo > 0 else "straddle")


def main() -> int:
    baseline: dict[int, list[int]] = defaultdict(list)
    baseline_text: dict[tuple[int, int], str] = {}
    branches: dict[int, dict[int, dict]] = defaultdict(dict)
    parse: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for path in LOO_JSONLS:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                row = r["source_row_index"]
                cond = r["condition"]
                parse[cond][0] += 1
                parse[cond][1] += int(bool(r.get("parse_failed")))
                s = int(bool(r["is_correct_strong"]))
                if cond == "unhinted_baseline":
                    baseline[row].append(s)
                    baseline_text[(row, r["sample_index"])] = r["model_output"]
                elif cond.startswith("percand_loo_fire"):
                    b = branches[row].setdefault(
                        r["fired_candidate_index"],
                        {"gauge": r["gauge_score"], "gold": bool(r["fired_is_gold"]),
                         "strong": [], "targets_fired": []},
                    )
                    b["strong"].append(s)
                    b["targets_fired"].append(int(bool(r.get("targets_fired_concept"))))

    bestofn: dict[int, list[int]] = defaultdict(list)
    l1_base_text: dict[tuple[int, int], str] = {}
    for path in L1_JSONLS:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["condition"] == "matched_bestofN_unsteered":
                    bestofn[r["source_row_index"]].append(int(bool(r["is_correct_strong"])))
                elif r["condition"] == "unhinted_baseline":
                    l1_base_text[(r["source_row_index"], r["sample_index"])] = r["model_output"]

    shared = sorted(set(baseline_text) & set(l1_base_text))
    verbatim = sum(baseline_text[k] == l1_base_text[k] for k in shared)

    rows = sorted(set(baseline) & set(branches) & set(bestofn))
    rows_with_gold = [r for r in rows if any(b["gold"] for b in branches[r].values())]
    base_rate = {r: float(np.mean(baseline[r])) for r in rows}
    rng = np.random.default_rng(20260704)

    def branch_mean(r, ci_idx):
        return float(np.mean(branches[r][ci_idx]["strong"]))

    policies: dict[str, dict[int, float]] = {}
    policies["oracle"] = {
        r: branch_mean(r, next(i for i, b in branches[r].items() if b["gold"])) for r in rows_with_gold
    }
    policies["gauge_select"] = {
        r: branch_mean(r, max(branches[r], key=lambda i: branches[r][i]["gauge"])) for r in rows
    }
    rand_draws = []
    for d in range(20):
        pr = random.Random(20260812 + d)
        rand_draws.append({r: branch_mean(r, pr.choice(sorted(branches[r]))) for r in rows})
    policies["random_select"] = {r: float(np.mean([dr[r] for dr in rand_draws])) for r in rows}
    policies["self_ratify"] = {
        r: branch_mean(
            r,
            max(sorted(branches[r]), key=lambda i: (float(np.mean(branches[r][i]["targets_fired"])), -i)),
        )
        for r in rows
    }
    policies["bestofn_majority"] = {r: float(np.mean(bestofn[r]) > 0.5) for r in rows}
    policies["bestofn_anycorrect"] = {r: float(any(bestofn[r])) for r in rows}

    report: dict = {
        "n_rows": len(rows),
        "n_rows_with_gold_branch": len(rows_with_gold),
        "baseline_p_strong": float(np.mean([base_rate[r] for r in rows])),
        "mean_candidates_per_row": float(np.mean([len(branches[r]) for r in rows])),
        "baseline_verbatim_gate": {
            "shared_generations": len(shared), "verbatim": verbatim,
            "pass": bool(shared) and verbatim == len(shared),
        },
    }

    arms_out = {}
    for name, vals in policies.items():
        rs = sorted(vals)
        deltas = np.array([vals[r] - base_rate[r] for r in rs])
        lo, hi = ci(deltas, rng)
        arms_out[name] = {
            "p_strong": float(np.mean([vals[r] for r in rs])),
            "dp": float(np.mean(deltas)), "ci95": [lo, hi], "sign": sign(lo, hi), "n_rows": len(rs),
        }
    report["policies"] = arms_out

    paired_out = {}
    for name, (a, b) in {
        "gauge_minus_bestofn_majority": (policies["gauge_select"], policies["bestofn_majority"]),
        "gauge_minus_bestofn_anycorrect": (policies["gauge_select"], policies["bestofn_anycorrect"]),
        "gauge_minus_random": (policies["gauge_select"], policies["random_select"]),
        "gauge_minus_selfratify": (policies["gauge_select"], policies["self_ratify"]),
        "gauge_minus_oracle": (policies["gauge_select"], policies["oracle"]),
    }.items():
        rs = sorted(set(a) & set(b))
        deltas = np.array([a[r] - b[r] for r in rs])
        lo, hi = ci(deltas, rng)
        paired_out[name] = {"delta": float(np.mean(deltas)), "ci95": [lo, hi], "sign": sign(lo, hi)}
    report["paired"] = paired_out

    parse_out = {}
    for c, (n, pf) in sorted(parse.items()):
        frac = pf / n
        parse_out[c] = {"n": n, "parse_fail": frac,
                        "status": "pass" if frac < 0.05 else ("VOID" if frac > 0.20 else "flag")}
    report["parse"] = parse_out

    oracle = arms_out["oracle"]
    report["oracle_gate"] = {"pass": oracle["sign"] == "pos", **oracle}
    gs, pm = arms_out["gauge_select"], paired_out["gauge_minus_bestofn_majority"]
    report["primary"] = {
        "conditions": "gauge_select dP CI > 0 AND paired (gauge - bestofn_majority) CI > 0",
        "pass": bool(gs["sign"] == "pos" and pm["sign"] == "pos"),
    }
    picks_gold = [
        float(max(branches[r], key=lambda i: branches[r][i]["gauge"])
              == next(i for i, b in branches[r].items() if b["gold"]))
        for r in rows_with_gold
    ]
    report["selector_texture"] = {
        "gauge_picks_gold_rate": float(np.mean(picks_gold)),
        "selector_write_interference_branch": bool(
            np.mean(picks_gold) >= 0.9 and report["oracle_gate"]["pass"] is False
        ),
    }
    gold_fires = [branch_mean(r, i) for r in rows for i, b in branches[r].items() if b["gold"]]
    nong_fires = [branch_mean(r, i) for r in rows for i, b in branches[r].items() if not b["gold"]]
    report["fire_texture"] = {
        "gold_branch_mean_p_strong": float(np.mean(gold_fires)) if gold_fires else None,
        "nongold_branch_mean_p_strong": float(np.mean(nong_fires)) if nong_fires else None,
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
