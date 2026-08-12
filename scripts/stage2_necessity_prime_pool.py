#!/usr/bin/env python3
"""Item K' pooled verdict per the registered branch partition
(docs/causal_handle_directions.md item K'). Writes
docs/necessity_prime_27b_property_pooled.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

JSONL = "results/stage2/erasure/necessity_prime_27b_property_shard{s}of3.jsonl"
OUT = Path("docs/necessity_prime_27b_property_pooled.json")

BASELINE = "correct_unhinted_baseline"
RANK8 = "correct_ablate_rank8_gold_L30"
MEAN = "correct_meanablate_gold_L30"
ARMS = [
    RANK8, MEAN,
    "correct_statepca8_ablate_gold_L30",
    "correct_ablate_rank1_gold_L30",
    "correct_ablate_rank2_gold_L30",
    "correct_ablate_rank4_gold_L30",
    "correct_ablate_dose012_gold_L30",
    "correct_keeponly8_gold_L30",
]


def ci(deltas: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    draws = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def sign_status(lo: float, hi: float) -> str:
    return "neg" if hi < 0 else ("pos" if lo > 0 else "straddle")


def main() -> int:
    per_row: dict[str, dict[tuple[int, int], list[dict]]] = defaultdict(lambda: defaultdict(list))
    parse: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    parsed_strong: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in range(3):
        with open(JSONL.format(s=s)) as f:
            for line in f:
                r = json.loads(line)
                per_row[r["condition"]][(s, r["source_row_index"])].append(r)
                parse[r["condition"]][0] += 1
                parse[r["condition"]][1] += int(bool(r.get("parse_failed")))
                if not r.get("parse_failed"):
                    parsed_strong[r["condition"]][0] += 1
                    parsed_strong[r["condition"]][1] += int(bool(r["is_correct_strong"]))

    rate = {c: {k: float(np.mean([x["is_correct_strong"] for x in v])) for k, v in rows.items()} for c, rows in per_row.items()}
    base = rate[BASELINE]
    rows_all = sorted(base)
    rng = np.random.default_rng(20260704)

    report: dict = {"n_rows": len(rows_all), "baseline_p_strong": float(np.mean([base[k] for k in rows_all]))}

    arms_out = {}
    for c in ARMS:
        deltas = np.array([rate[c][k] - base[k] for k in rows_all])
        lo, hi = ci(deltas, rng)
        arms_out[c] = {"p_strong": float(np.mean([rate[c][k] for k in rows_all])), "dp": float(np.mean(deltas)), "ci95": [lo, hi], "sign": sign_status(lo, hi)}
    report["arms"] = arms_out

    paired_out = {}
    for name, (a, b) in {
        "meanablate_minus_rank8": (rate[MEAN], rate[RANK8]),
        "statepca8_minus_rank8": (rate["correct_statepca8_ablate_gold_L30"], rate[RANK8]),
    }.items():
        deltas = np.array([a[k] - b[k] for k in rows_all])
        lo, hi = ci(deltas, rng)
        paired_out[name] = {"delta": float(np.mean(deltas)), "ci95": [lo, hi], "sign": sign_status(lo, hi)}
    report["paired"] = paired_out

    parse_out = {}
    for c, (n, pf) in sorted(parse.items()):
        frac = pf / n
        entry = {"n": n, "parse_fail": frac, "status": "pass" if frac < 0.05 else ("VOID" if frac > 0.20 else "flag")}
        pn, ps = parsed_strong[c]
        entry["p_strong_given_parsed"] = ps / pn if pn else None
        parse_out[c] = entry
    report["parse"] = parse_out
    report["baseline_gate_055"] = report["baseline_p_strong"] >= 0.55

    mean_void = parse_out[MEAN]["status"] == "VOID"
    ma, pr = arms_out[MEAN], paired_out["meanablate_minus_rank8"]
    if mean_void:
        branch = "VOID (meanablate parse-fail > 20% pooled) -> registered debug path"
    elif ma["sign"] == "neg" and pr["sign"] in ("neg", "straddle"):
        branch = "CONTENT-NECESSITY (meanablate breaks as hard as zero-ablation)"
    elif ma["sign"] == "neg":
        branch = "PARTIAL (breaks, but less than zero-ablation)"
    elif ma["sign"] == "straddle":
        branch = "ENERGY-ACCOUNT (meanablate null)"
    else:
        branch = "CATCH-ALL"
    report["k_prime_primary"] = {"branch": branch, "meanablate_parse_void": mean_void}

    keep = arms_out["correct_keeponly8_gold_L30"]
    report["riders"] = {
        "keeponly8_label": "SUFFICIENT-AT-SITE" if keep["ci95"][0] >= -0.15 else "INSUFFICIENT-AT-SITE",
        "ladder": {c: arms_out[c]["dp"] for c in ARMS if "rank" in c or c == RANK8},
        "dose012_vs_K_randnorm": {"dose012": arms_out["correct_ablate_dose012_gold_L30"]["dp"], "K_rand_norm_family": -0.351, "K_signflip100": -0.696},
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
