#!/usr/bin/env python3
"""Item L″ pooled verdict: the composition (gauge-selected addressing × the
transferring LOO write), per docs/causal_handle_directions.md item L″.

Order of operations is part of the registration: the baseline verbatim gate
vs item L1's recorded baselines runs FIRST; any mismatch aborts before the
selector suite is computed (debug, no unblinding). The compute-matched
sampling comparator is NOT re-run — L1's recorded matched_bestofN_unsteered
rows are the registered cross-job comparator. Writes
docs/selfaddress_loo_27b_property_pooled.json.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

LOO_JSONLS = [
    f"results/stage2/erasure/selfaddress_loo_27b_property_shard{i}of4.jsonl" for i in range(4)
]
L1_JSONLS = [
    "results/stage2/erasure/selfaddress_27b_property_shard0of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard1of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard2of4.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard3of8.jsonl",
    "results/stage2/erasure/selfaddress_27b_property_shard7of8.jsonl",
]
OUT = Path("docs/selfaddress_loo_27b_property_pooled.json")
LPRIME_GOLD_ANCHOR = 0.279


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
                elif cond == "percand_loo_fire_L30":
                    b = branches[row].setdefault(
                        r["fired_candidate_index"],
                        {"gauge": r["gauge_score"], "gold": bool(r["fired_is_gold"]),
                         "strong": [], "targets_fired": []},
                    )
                    b["strong"].append(s)
                    b["targets_fired"].append(int(bool(r.get("targets_fired_concept"))))

    l1_base_text: dict[tuple[int, int], str] = {}
    bestofn: dict[int, list[int]] = defaultdict(list)
    for path in L1_JSONLS:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                row = r["source_row_index"]
                if r["condition"] == "unhinted_baseline":
                    l1_base_text[(row, r["sample_index"])] = r["model_output"]
                elif r["condition"] == "matched_bestofN_unsteered":
                    bestofn[row].append(int(bool(r["is_correct_strong"])))

    shared = sorted(set(baseline_text) & set(l1_base_text))
    mismatches = [k for k in shared if baseline_text[k] != l1_base_text[k]]
    gate = {
        "n_compared": len(shared),
        "n_verbatim": len(shared) - len(mismatches),
        "pass": len(shared) > 0 and not mismatches,
        "mismatch_keys": mismatches[:20],
    }
    if not gate["pass"]:
        OUT.write_text(json.dumps({"baseline_verbatim_gate": gate, "verdict": "GATE FAIL - debug, no unblinding"}, indent=2) + "\n")
        print(json.dumps(gate, indent=2))
        print("BASELINE VERBATIM GATE FAILED - selector suite not computed.")
        return 1

    rows = sorted(set(baseline) & set(branches) & set(bestofn))
    rows_with_gold = [r for r in rows if any(b["gold"] for b in branches[r].values())]
    base_rate = {r: float(np.mean(baseline[r])) for r in rows}
    rng = np.random.default_rng(20260704)

    def branch_mean(r, ci_idx):
        return float(np.mean(branches[r][ci_idx]["strong"]))

    policies: dict[str, dict[int, float]] = {}
    policies["gold_branch"] = {
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
    policies["bestofn_majority_l1recorded"] = {r: float(np.mean(bestofn[r]) > 0.5) for r in rows}
    policies["bestofn_anycorrect_l1recorded"] = {r: float(any(bestofn[r])) for r in rows}

    report: dict = {
        "baseline_verbatim_gate": gate,
        "n_rows": len(rows),
        "n_rows_with_gold_branch": len(rows_with_gold),
        "baseline_p_strong": float(np.mean([base_rate[r] for r in rows])),
        "mean_candidates_per_row": float(np.mean([len(branches[r]) for r in rows])),
        "lprime_gold_anchor_dp": LPRIME_GOLD_ANCHOR,
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
        if arms_out[name]["sign"] == "straddle":
            arms_out[name]["mde_observed_halfwidth"] = float((hi - lo) / 2)
    report["policies"] = arms_out

    paired_out = {}
    for name, (a, b) in {
        "gauge_minus_bestofn_majority": (policies["gauge_select"], policies["bestofn_majority_l1recorded"]),
        "gauge_minus_bestofn_anycorrect": (policies["gauge_select"], policies["bestofn_anycorrect_l1recorded"]),
        "gauge_minus_random": (policies["gauge_select"], policies["random_select"]),
        "gauge_minus_selfratify": (policies["gauge_select"], policies["self_ratify"]),
        "gauge_minus_gold": (policies["gauge_select"], policies["gold_branch"]),
    }.items():
        rs = sorted(set(a) & set(b))
        deltas = np.array([a[r] - b[r] for r in rs])
        lo, hi = ci(deltas, rng)
        paired_out[name] = {"delta": float(np.mean(deltas)), "ci95": [lo, hi], "sign": sign(lo, hi), "n_rows": len(rs)}
        if paired_out[name]["sign"] == "straddle":
            paired_out[name]["mde_observed_halfwidth"] = float((hi - lo) / 2)
    report["paired"] = paired_out

    nongold = {}
    for r in rows:
        vals = [branch_mean(r, i) for i, b in branches[r].items() if not b["gold"]]
        if vals:
            nongold[r] = float(np.mean(vals))
    rs = sorted(nongold)
    deltas = np.array([nongold[r] - base_rate[r] for r in rs])
    lo, hi = ci(deltas, rng)
    report["wrong_address_collateral"] = {
        "dp": float(np.mean(deltas)), "ci95": [lo, hi], "sign": sign(lo, hi), "n_rows": len(rs),
    }

    parse_out = {}
    for c, (n, pf) in sorted(parse.items()):
        frac = pf / n
        parse_out[c] = {"n": n, "parse_fail": frac,
                        "status": "pass" if frac < 0.05 else ("VOID" if frac > 0.20 else "flag")}
    report["parse"] = parse_out

    gs = arms_out["gauge_select"]
    gold = arms_out["gold_branch"]
    pm = paired_out["gauge_minus_bestofn_majority"]
    primary_pass = gs["sign"] == "pos" and pm["sign"] == "pos"
    report["l2primes_primary"] = {
        "conditions": "gauge_select dP vs in-job baseline CI > 0 AND paired (gauge_select - L1-recorded bestofN majority) CI > 0",
        "gauge_select_sign": gs["sign"], "paired_sign": pm["sign"], "pass": primary_pass,
    }
    if primary_pass:
        branch = "PASS: the answer-free loop closes on the adjudication rows (L''' fresh-draw confirmation required before any generalization language)"
    elif gs["sign"] == "pos":
        branch = "branch 1: repairs but sampling-parity unresolved (paired vs bestofN-majority straddles; MDE stated)"
    elif gs["sign"] != "pos" and gold["sign"] == "pos":
        branch = "branch 2: SELECTOR-WRITE INTERFERENCE (gauge-select null while own gold branch CI > 0; compare gauge-vs-random to separate selector failure from branch degradation)"
    elif gold["sign"] != "pos":
        branch = "branch 3: protocol instability across arm structure (own gold branch fails to reproduce L-prime's +0.279); verdict confined to reporting this"
    else:
        branch = "branch 5: catch-all, descriptive"
    report["registered_branch"] = branch
    if report["wrong_address_collateral"]["sign"] == "neg":
        report["registered_branch_addendum"] = "branch 4 rider: wrong-address collateral CI < 0 (misaddressed LOO fires actively harm)"

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
