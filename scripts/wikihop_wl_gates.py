#!/usr/bin/env python3
"""Item WL gate readings (registered: docs/causal_handle_directions.md item WL).
(a) TEXT CEILING; (b) REPLICATION at 1x/2x: gold-δ dP CI>0 AND SPECIFICITY (per-row gold-write
correct − mean non-gold-write correct) CI>0; (c) LOOP at the loop rung: gauge-select over ALL
candidate branches vs baseline and vs self-consistency@8. Fingerprint descriptive."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci, r_norm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wl_gates.json"))
    p.add_argument("--loop-rung", type=float, default=2.0)
    args = p.parse_args()
    recs = [json.loads(l) for f in args.jsonl for l in open(f)]
    rows = sorted({r["id"] for r in recs})
    base = defaultdict(list)
    for r in recs:
        if r["condition"] == "baseline":
            base[r["id"]].append(r)
    base_rate = {i: float(np.mean([b["correct"] for b in base[i]])) for i in rows}
    sc8 = {}
    for i in rows:
        cnt = Counter(b["normalized_output"] for b in base[i])
        modal = cnt.most_common(1)[0][0]
        sc8[i] = float(next(b["correct"] for b in base[i] if b["normalized_output"] == modal))
    writes = [r for r in recs if r["condition"] == "delta_write"]
    audit = {"n_fired_records": len(writes),
             "n_zero_prefill_hook_calls": sum((r["hook_prefill_calls"] or 0) < 1 for r in writes),
             "n_zero_positions_written": sum((r["hook_positions_written"] or 0) < 1 for r in writes),
             "n_gauge_forward_unhooked": sum((r["gauge_forward_hook_calls"] or 0) < 1 for r in writes)}
    audit["valid"] = all(audit[k] == 0 for k in ("n_zero_prefill_hook_calls", "n_zero_positions_written", "n_gauge_forward_unhooked"))

    tg = defaultdict(list)
    for r in recs:
        if r["condition"] == "text_hint" and r["fired_is_gold"]:
            tg[r["id"]].append(r["correct"])
    tdp = [float(np.mean(tg[i])) - base_rate[i] for i in sorted(tg)]
    text = {"n_rows": len(tdp), "rate": float(np.mean([np.mean(tg[i]) for i in tg])) if tg else None,
            "dP": float(np.mean(tdp)) if tdp else None, "ci": boot_ci(tdp), "pass": bool(tdp and boot_ci(tdp)[0] > 0)}

    per_rung = {}
    for rung in sorted({r["rung"] for r in writes}):
        g = defaultdict(list)
        n = defaultdict(lambda: defaultdict(list))
        for r in writes:
            if r["rung"] != rung:
                continue
            if r["fired_is_gold"]:
                g[r["id"]].append(r["correct"])
            else:
                n[r["id"]][r["fired_candidate"]].append(r["correct"])
        ids = sorted(i for i in g if i in n)
        dp = [float(np.mean(g[i])) - base_rate[i] for i in sorted(g)]
        spec = [float(np.mean(g[i])) - float(np.mean([np.mean(v) for v in n[i].values()])) for i in ids]
        non_rate = [float(np.mean([np.mean(v) for v in n[i].values()])) for i in ids]
        af = defaultdict(list)
        for r in writes:
            if r["rung"] == rung and not r["fired_is_gold"]:
                af[(r["id"], r["fired_candidate"])].append(r["answers_fired"])
        per_rung[str(rung)] = {"n_rows_gold": len(dp), "gold_rate": float(np.mean([np.mean(g[i]) for i in g])) if g else None,
                               "dP": float(np.mean(dp)) if dp else None, "dP_ci": boot_ci(dp), "dP_pass": bool(dp and boot_ci(dp)[0] > 0),
                               "n_rows_spec": len(spec), "nongold_write_gold_rate": float(np.mean(non_rate)) if non_rate else None,
                               "specificity": float(np.mean(spec)) if spec else None, "specificity_ci": boot_ci(spec),
                               "specificity_pass": bool(spec and boot_ci(spec)[0] > 0),
                               "n_nongold_fires": len(af), "nongold_answers_fired_rate": float(np.mean([np.mean(v) for v in af.values()])) if af else None}
        per_rung[str(rung)]["replication_pass"] = bool(per_rung[str(rung)]["dP_pass"] and per_rung[str(rung)]["specificity_pass"])
    passing = [k for k, v in per_rung.items() if v["replication_pass"]]
    pinned = max(passing, key=lambda k: per_rung[k]["dP"]) if passing else None

    lr = args.loop_rung
    loop_rows, loop_c, oracle_c, argmax_gold, chance, sel = [], [], [], [], [], []
    for i in rows:
        br = defaultdict(list)
        for r in writes:
            if r["id"] == i and r["rung"] == lr:
                br[r["fired_candidate"]].append(r)
        if len(br) < 2 or not any(v[0]["fired_is_gold"] for v in br.values()):
            continue
        best = max(br, key=lambda c: br[c][0]["gauge_score"])
        loop_rows.append(i)
        loop_c.append(float(np.mean([x["correct"] for x in br[best]])))
        gold_c = [c for c in br if br[c][0]["fired_is_gold"]][0]
        oracle_c.append(float(np.mean([x["correct"] for x in br[gold_c]])))
        argmax_gold.append(best == gold_c)
        chance.append(1 / len(br))
        gs = br[gold_c][0]["gauge_score"]
        ns = [br[c][0]["gauge_score"] for c in br if c != gold_c]
        sel.append(gs - float(np.mean(ns)))
    d_base = [loop_c[k] - base_rate[i] for k, i in enumerate(loop_rows)]
    d_sc = [loop_c[k] - sc8[i] for k, i in enumerate(loop_rows)]
    loop = {"rung": lr, "n_rows": len(loop_rows), "gauge_select_rate": float(np.mean(loop_c)) if loop_c else None,
            "oracle_rate": float(np.mean(oracle_c)) if oracle_c else None,
            "baseline_rate": float(np.mean([base_rate[i] for i in loop_rows])) if loop_rows else None,
            "sc8_rate": float(np.mean([sc8[i] for i in loop_rows])) if loop_rows else None,
            "dP_vs_baseline": float(np.mean(d_base)) if d_base else None, "dP_vs_baseline_ci": boot_ci(d_base),
            "dP_vs_sc8": float(np.mean(d_sc)) if d_sc else None, "dP_vs_sc8_ci": boot_ci(d_sc),
            "argmax_gold_rate": float(np.mean(argmax_gold)) if argmax_gold else None, "chance_rate": float(np.mean(chance)) if chance else None,
            "selection_signal": float(np.mean(sel)) if sel else None, "selection_signal_ci": boot_ci(sel),
            "mean_n_branches": float(np.mean([1 / c for c in chance])) if chance else None}
    loop["pass_i"] = bool(d_base and boot_ci(d_base)[0] > 0)
    loop["pass_ii"] = bool(d_sc and boot_ci(d_sc)[0] > 0)

    if not audit["valid"]:
        verdict = "EXECUTION-INVALID (delivery audit)"
    elif not text["pass"]:
        verdict = "NO-CEILING"
    elif not passing:
        verdict = "WRITE-SIDE-CLOSED (replication fails: dP or specificity CI not > 0 at 1x and 2x)"
    elif loop["pass_i"] and loop["pass_ii"]:
        verdict = f"HINT-DELTA-TRANSFERS (pinned rung {pinned}x) + LOOP-CLOSES-ON-NATURAL-DATA"
    elif loop["pass_i"]:
        verdict = f"HINT-DELTA-TRANSFERS (pinned rung {pinned}x) + LOOP-BEATS-BASELINE-NOT-SC"
    else:
        verdict = f"HINT-DELTA-TRANSFERS (pinned rung {pinned}x) + SELECTOR-FAILS"
    out = {"n_rows": len(rows), "delivery_audit": audit, "text_ceiling": text, "per_rung": per_rung,
           "rungs_passing_replication": passing, "pinned_rung": pinned, "loop": loop, "verdict": verdict}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
