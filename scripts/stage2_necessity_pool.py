#!/usr/bin/env python3
"""Item K pooled verdict: three correct-side shards + anchor, evaluated against
the registered branch partition in docs/causal_handle_directions.md item K.

Reads the shard jsonls, pools rows keyed (shard, source_row_index), computes
row-cluster bootstrap CIs (10k, seed 20260704) for every arm and registered
paired contrast, evaluates the gates and branch conditions mechanically, and
writes docs/necessity_27b_property_pooled.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SHARDS = [0, 1, 2]
JSONL = "results/stage2/erasure/necessity_27b_property_shard{s}of3.jsonl"
ANCHOR_JSON = Path("docs/necessity_anchor_27b_property_shard0of1.json")
OUT = Path("docs/necessity_27b_property_pooled.json")

BASELINE = "correct_unhinted_baseline"
ABLATE = "correct_ablate_rank8_gold_L30"
RAND8 = ["correct_ablate_rand8_gold_L30_d1", "correct_ablate_rand8_gold_L30_d2"]
PERM8 = "correct_ablate_perm8_gold_L30"
SIGNFLIP100 = "correct_signflip_fixednorm_100"
SIGNFLIP200 = "correct_signflip_fixednorm_200"
RANDNORM = ["correct_rand_norm_gold_d1", "correct_rand_norm_gold_d2"]
COLLATERAL = "correct_fixednorm_100"

ANCHOR_DP = None
ANCHOR_BASELINE = None


def ci(deltas: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    draws = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def sign_status(lo: float, hi: float) -> str:
    if hi < 0:
        return "neg"
    if lo > 0:
        return "pos"
    return "straddle"


def main() -> int:
    per_row: dict[str, dict[tuple[int, int], list[dict]]] = defaultdict(lambda: defaultdict(list))
    parse: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for s in SHARDS:
        with open(JSONL.format(s=s)) as f:
            for line in f:
                r = json.loads(line)
                key = (s, r["source_row_index"])
                per_row[r["condition"]][key].append(r)
                p = parse[r["condition"]]
                p[0] += 1
                p[1] += int(bool(r.get("parse_failed")))
                p[2] += int(bool(r.get("is_correct_strong")))

    rate = {c: {k: float(np.mean([x["is_correct_strong"] for x in v])) for k, v in rows.items()} for c, rows in per_row.items()}
    base = rate[BASELINE]
    rows_all = sorted(base)
    rng = np.random.default_rng(20260704)

    def fam(conds: list[str]) -> dict[tuple[int, int], float]:
        return {k: float(np.mean([rate[c][k] for c in conds if k in rate[c]])) for k in rows_all}

    report: dict = {"n_rows": len(rows_all), "rows": [list(k) for k in rows_all]}

    arms_out = {}
    fam_rates = {"rand8_family": fam(RAND8), "rand_norm_family": fam(RANDNORM)}
    singles = {c: rate[c] for c in [ABLATE, PERM8, SIGNFLIP100, SIGNFLIP200, COLLATERAL] + RAND8 + RANDNORM}
    for name, r in {**singles, **fam_rates}.items():
        deltas = np.array([r[k] - base[k] for k in rows_all])
        lo, hi = ci(deltas, rng)
        arms_out[name] = {
            "p_strong": float(np.mean([r[k] for k in rows_all])),
            "dp": float(np.mean(deltas)),
            "ci95": [lo, hi],
            "sign": sign_status(lo, hi),
            "ci_half_width": (hi - lo) / 2,
        }
    report["baseline_p_strong"] = float(np.mean([base[k] for k in rows_all]))
    report["arms"] = arms_out

    paired_out = {}
    for name, (a, b) in {
        "ablate_minus_rand8fam": (rate[ABLATE], fam_rates["rand8_family"]),
        "ablate_minus_perm8": (rate[ABLATE], rate[PERM8]),
        "signflip100_minus_randnormfam": (rate[SIGNFLIP100], fam_rates["rand_norm_family"]),
        "signflip200_minus_signflip100": (rate[SIGNFLIP200], rate[SIGNFLIP100]),
    }.items():
        deltas = np.array([a[k] - b[k] for k in rows_all])
        lo, hi = ci(deltas, rng)
        paired_out[name] = {"delta": float(np.mean(deltas)), "ci95": [lo, hi], "sign": sign_status(lo, hi)}
    report["paired"] = paired_out

    gates = {"anchor_verbatim": None}
    anchor = json.loads(ANCHOR_JSON.read_text())
    anchor_summary = anchor["summary"]["fixednorm_proj_add_L30"]
    anchor_base = anchor["summary"]["unhinted_baseline"]["strong_accuracy"]
    gates["anchor_dp"] = anchor_summary["paired_delta_vs_reference"]
    gates["anchor_ci95"] = anchor_summary["paired_ci95"]
    gates["anchor_baseline"] = anchor_base
    gates["baseline_gate_055"] = report["baseline_p_strong"] >= 0.55
    parse_out = {}
    for c, (n, pf, strong) in sorted(parse.items()):
        frac = pf / n
        entry = {"n": n, "parse_fail": frac, "status": "pass" if frac < 0.05 else ("VOID" if frac > 0.20 else "flag")}
        if frac >= 0.05:
            parsed = [x for rows in per_row[c].values() for x in rows if not x.get("parse_failed")]
            entry["p_strong_given_parsed"] = float(np.mean([x["is_correct_strong"] for x in parsed]))
        parse_out[c] = entry
    gates["parse"] = parse_out
    report["gates"] = gates

    ab, pr = arms_out[ABLATE], paired_out["ablate_minus_rand8fam"]
    r8 = arms_out["rand8_family"]
    if ab["sign"] == "neg" and pr["sign"] == "neg":
        branch = "CHANNEL-IN-USE (a: rand8 family straddles)" if r8["sign"] != "neg" else "CHANNEL-IN-USE (b: rand8 family also negative)"
    elif ab["sign"] == "straddle":
        branch = "INVERSE-SPECIFICITY" if r8["sign"] == "neg" else "WRITE-ONLY PORT"
    elif ab["sign"] == "neg" and r8["sign"] == "neg" and pr["sign"] == "straddle":
        branch = "PROJECTION-DAMAGE CONFOUND"
    elif ab["sign"] == "neg" and pr["sign"] == "straddle" and r8["sign"] == "straddle":
        branch = "BREAKS-SPECIFICITY-UNRESOLVED"
    else:
        branch = "CATCH-ALL (descriptive)"
    flag_pass = paired_out["ablate_minus_perm8"]["sign"] == "neg"
    report["k_primary"] = {"branch": branch, "perm8_flag_layer_passes": flag_pass}

    sf, sfp = arms_out[SIGNFLIP100], paired_out["signflip100_minus_randnormfam"]
    report["prediction_i"] = {
        "confirmed": sf["sign"] == "neg" and sfp["sign"] == "neg",
        "signflip100": sf, "paired_vs_randnorm_family": sfp,
    }

    b_frac = abs(sf["dp"]) / report["baseline_p_strong"]
    p_frac = gates["anchor_dp"] / (1 - gates["anchor_baseline"])
    report["asymmetry"] = {
        "break_fraction_of_downroom": b_frac,
        "repair_fraction_of_uproom": p_frac,
        "label": "attractor-compatible (B > P)" if b_frac > p_frac else "symmetric-or-repair-favored",
        "dose_monotone": paired_out["signflip200_minus_signflip100"]["delta"] < 0,
    }

    col = arms_out[COLLATERAL]
    report["collateral"] = {
        "replicates_direction": col["sign"] == "pos",
        "headroom_fraction": col["dp"] / (1 - report["baseline_p_strong"]),
        "original_headroom_fraction": 0.266 / 0.273,
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
