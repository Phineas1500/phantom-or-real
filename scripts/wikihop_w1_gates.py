#!/usr/bin/env python3
"""Item W1 gate readings (registered: docs/causal_handle_directions.md item W).

From the W1 row-level JSONL: (b) POSITIVE CONTROL per rung — gold-fire repair
direction (gold correct rate minus in-job baseline correct rate) AND the
in-frame delivery fingerprint at non-gold fires (answers-fired rate minus that
candidate's baseline rate, row-bootstrap CI > 0); (c) SELECTION SIGNAL at the
per-candidate rung (gold-branch gauge score minus mean non-gold gauge score,
row-bootstrap CI > 0). Also the delivery telemetry audit (every fired branch
must show >= 1 prefill hook call inside generate).
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def boot_ci(values, seed=20260822, draws=4000):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(draws)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/loop_screen/wikihop_w1.jsonl"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_w1_gates.json"))
    args = p.parse_args()
    all_recs = [json.loads(l) for l in open(args.jsonl)]
    layers = sorted({r["write_layer"] for r in all_recs})
    if len(layers) > 1:
        per_layer = {}
        for L in layers:
            per_layer[str(L)] = gates_for([r for r in all_recs if r["write_layer"] == L])
        out = {"write_layers": layers, "per_layer": per_layer,
               "layers_passing_positive_control": [L for L in per_layer if per_layer[L]["rungs_passing_positive_control"]]}
    else:
        out = gates_for(all_recs)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


def gates_for(recs):
    rows = sorted({r["id"] for r in recs})
    base = defaultdict(list)
    for r in recs:
        if r["condition"] == "baseline":
            base[r["id"]].append(r)
    base_rate = {i: np.mean([b["correct"] for b in base[i]]) for i in rows}
    base_ans = {i: defaultdict(float) for i in rows}
    for i in rows:
        for b in base[i]:
            base_ans[i][b["normalized_output"]] += 1 / len(base[i])

    fired = [r for r in recs if r["fired_candidate"] is not None]
    delivery_audit = {"n_fired_records": len(fired),
                      "n_zero_prefill_hook_calls": sum((r["hook_prefill_calls"] or 0) < 1 for r in fired),
                      "n_zero_positions_written": sum((r["hook_positions_written"] or 0) < 1 for r in fired),
                      "n_gauge_forward_unhooked": sum((r.get("gauge_forward_hook_calls") or 0) < 1 for r in fired)}
    delivery_audit["valid"] = all(v == 0 for k, v in delivery_audit.items() if k.startswith("n_zero") or k.startswith("n_gauge"))

    rungs = sorted({r["rung"] for r in fired})
    per_rung = {}
    for rung in rungs:
        gold = defaultdict(list)
        nongold = defaultdict(list)
        for r in fired:
            if r["rung"] != rung:
                continue
            if r["fired_is_gold"]:
                gold[r["id"]].append(r["correct"])
            else:
                nongold[(r["id"], r["fired_candidate"])].append(r["answers_fired"])
        gold_rows = sorted(gold)
        dp = [np.mean(gold[i]) - base_rate[i] for i in gold_rows]
        gold_rate = [np.mean(gold[i]) for i in gold_rows]
        lifts_by_row = defaultdict(list)
        for (i, c), v in nongold.items():
            key = [k for k in base_ans[i] if k == r_norm(c)]
            b = base_ans[i][key[0]] if key else 0.0
            lifts_by_row[i].append(np.mean(v) - b)
        lift_rows = sorted(lifts_by_row)
        lift = [np.mean(lifts_by_row[i]) for i in lift_rows]
        raw_af = [np.mean(v) for v in nongold.values()]
        pr = {"n_rows_gold": len(gold_rows), "gold_correct_rate": float(np.mean(gold_rate)) if gold_rate else None,
              "baseline_correct_rate": float(np.mean([base_rate[i] for i in gold_rows])) if gold_rows else None,
              "dP_gold": float(np.mean(dp)) if dp else None, "dP_gold_ci": boot_ci(dp),
              "repair_direction": bool(dp and np.mean(dp) > 0),
              "n_rows_nongold": len(lift_rows), "n_nongold_fires": len(nongold),
              "nongold_answers_fired_rate": float(np.mean(raw_af)) if raw_af else None,
              "fingerprint_lift": float(np.mean(lift)) if lift else None, "fingerprint_lift_ci": boot_ci(lift),
              "fingerprint_pass": bool(lift and boot_ci(lift)[0] > 0)}
        pr["positive_control_pass"] = bool(pr["repair_direction"] and pr["fingerprint_pass"])
        per_rung[str(rung)] = pr

    percand = [r for r in fired if r["condition"] == "percand"]
    pc_rung = percand[0]["rung"] if percand else None
    sel = []
    argmax_gold = []
    for i in rows:
        g = {}
        for r in percand:
            if r["id"] == i:
                g[r["fired_candidate"]] = (r["gauge_score"], r["fired_is_gold"])
        if not g or not any(v[1] for v in g.values()):
            continue
        gold_s = [v[0] for v in g.values() if v[1]][0]
        non = [v[0] for v in g.values() if not v[1]]
        if non:
            sel.append(gold_s - float(np.mean(non)))
            argmax_gold.append(max(g.items(), key=lambda kv: kv[1][0])[1][1])
    selection = {"rung": pc_rung, "n_rows": len(sel), "gold_minus_nongold_gauge": float(np.mean(sel)) if sel else None,
                 "ci": boot_ci(sel), "argmax_is_gold": int(sum(argmax_gold)), "pass": bool(sel and boot_ci(sel)[0] > 0)}
    gauge_select_dp = []
    sc_dp = []
    for i in rows:
        branch = defaultdict(list)
        for r in percand:
            if r["id"] == i:
                branch[r["fired_candidate"]].append((r["gauge_score"], r["correct"]))
        if not branch:
            continue
        best = max(branch, key=lambda c: branch[c][0][0])
        gauge_select_dp.append(np.mean([c for _, c in branch[best]]) - base_rate[i])
    passing = [str(r) for r in rungs if per_rung[str(r)]["positive_control_pass"]]
    out = {"n_rows": len(rows), "delivery_audit": delivery_audit, "per_rung": per_rung,
           "selection_signal": selection,
           "gauge_select_dP_descriptive": {"mean": float(np.mean(gauge_select_dp)) if gauge_select_dp else None,
                                           "ci": boot_ci(gauge_select_dp)},
           "rungs_passing_positive_control": passing,
           "pinned_rung_for_w2": (percand[0]["rung"] if (percand and str(percand[0]["rung"]) in passing)
                                  else (float(passing[0]) if passing else None)),
           "selector_demoted_to_descriptive": not selection["pass"]}
    return out


def r_norm(s):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wikihop_common import normalize_answer
    return normalize_answer(s)


if __name__ == "__main__":
    raise SystemExit(main())
