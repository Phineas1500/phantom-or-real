#!/usr/bin/env python3
"""Item WH gate readings (registered: docs/causal_handle_directions.md item WH).
(a) TEXT CEILING: hint-first gold text correct rate − baseline, row-bootstrap CI > 0.
(b) ORACLE WRITE per rung: gold-δ write dP CI > 0 AND non-gold fingerprint lift CI > 0.
(c) descriptive: recovered fraction, non-gold text answers-fired lift, |δ|, gauge shifts."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci, r_norm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/loop_screen/wikihop_wh.jsonl"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wh_gates.json"))
    args = p.parse_args()
    recs = [json.loads(l) for l in open(args.jsonl)]
    rows = sorted({r["id"] for r in recs})
    base = defaultdict(list)
    for r in recs:
        if r["condition"] == "baseline":
            base[r["id"]].append(r)
    base_rate = {i: float(np.mean([b["correct"] for b in base[i]])) for i in rows}
    base_ans = {i: defaultdict(float) for i in rows}
    for i in rows:
        for b in base[i]:
            base_ans[i][b["normalized_output"]] += 1 / len(base[i])

    def rate_stats(cond, rung, gold: bool):
        per_row = defaultdict(list)
        for r in recs:
            if r["condition"] == cond and r["rung"] == rung and r["fired_is_gold"] == gold:
                key = r["id"] if gold else (r["id"], r["fired_candidate"])
                per_row[key].append(r["correct"] if gold else r["answers_fired"])
        if gold:
            ids = sorted(per_row)
            dp = [float(np.mean(per_row[i])) - base_rate[i] for i in ids]
            return {"n_rows": len(ids), "rate": float(np.mean([np.mean(per_row[i]) for i in ids])) if ids else None,
                    "baseline": float(np.mean([base_rate[i] for i in ids])) if ids else None,
                    "dP": float(np.mean(dp)) if dp else None, "ci": boot_ci(dp), "pass": bool(dp and boot_ci(dp)[0] > 0)}
        by_row = defaultdict(list)
        for (i, c), v in per_row.items():
            by_row[i].append(float(np.mean(v)) - base_ans[i].get(r_norm(c), 0.0))
        lifts = [float(np.mean(by_row[i])) for i in sorted(by_row)]
        return {"n_rows": len(lifts), "n_fires": len(per_row),
                "answers_fired_rate": float(np.mean([np.mean(v) for v in per_row.values()])) if per_row else None,
                "lift": float(np.mean(lifts)) if lifts else None, "ci": boot_ci(lifts), "pass": bool(lifts and boot_ci(lifts)[0] > 0)}

    writes = [r for r in recs if r["condition"] == "delta_write"]
    audit = {"n_fired_records": len(writes),
             "n_zero_prefill_hook_calls": sum((r["hook_prefill_calls"] or 0) < 1 for r in writes),
             "n_zero_positions_written": sum((r["hook_positions_written"] or 0) < 1 for r in writes),
             "n_gauge_forward_unhooked": sum((r["gauge_forward_hook_calls"] or 0) < 1 for r in writes)}
    audit["valid"] = all(v == 0 for k, v in audit.items() if k.startswith("n_zero") or k.startswith("n_gauge"))
    text_gold = rate_stats("text_hint", None, True)
    text_non = rate_stats("text_hint", None, False)
    rungs = sorted({r["rung"] for r in writes})
    per_rung = {}
    for rung in rungs:
        g = rate_stats("delta_write", rung, True)
        n = rate_stats("delta_write", rung, False)
        gs = [r["gauge_score"] - r["base_gauge_score"] for r in writes if r["rung"] == rung and r["fired_is_gold"]]
        ns = [r["gauge_score"] - r["base_gauge_score"] for r in writes if r["rung"] == rung and not r["fired_is_gold"]]
        per_rung[str(rung)] = {"gold": g, "nongold": n, "positive_control_pass": bool(g["pass"] and n["pass"]),
                               "recovered_fraction": (g["dP"] / text_gold["dP"]) if (text_gold["dP"] and g["dP"] is not None) else None,
                               "gauge_shift_gold": float(np.mean(gs)) if gs else None, "gauge_shift_nongold": float(np.mean(ns)) if ns else None}
    dn = [r["delta_mean_position_norm"] for r in recs if r["condition"] == "text_hint" and r["fired_is_gold"] and r["sample_index"] == 0]
    passing = [k for k, v in per_rung.items() if v["positive_control_pass"]]
    if not audit["valid"]:
        verdict = "EXECUTION-INVALID (delivery audit)"
    elif not text_gold["pass"]:
        verdict = "NO-CEILING (hint-first text does not repair these rows; write test uninformative)"
    elif passing:
        verdict = f"HINT-DELTA-TRANSFERS (oracle, reading-driven scope; rungs {passing})"
    else:
        verdict = "HINT-DELTA-DOES-NOT-TRANSFER"
    out = {"n_rows": len(rows), "delivery_audit": audit, "text_ceiling_gold": text_gold, "text_nongold_fingerprint_ceiling": text_non,
           "per_rung": per_rung, "gold_delta_mean_position_norm": {"mean": float(np.mean(dn)) if dn else None, "n": len(dn)},
           "rungs_passing": passing, "verdict": verdict}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
