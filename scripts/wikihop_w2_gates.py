#!/usr/bin/env python3
"""Item W2 gate readings (registered: docs/causal_handle_directions.md item W).

From the concatenated W2 shard JSONLs (W2 mode of wikihop_w1_job.py: k=8
baselines + per-candidate fires k=4 at the W1-pinned rung, plus the gold
ladder branch at that rung):
  * DELIVERY HARD GATE — the W1-validated fingerprint: non-gold answers-fired
    rate minus that candidate's baseline rate, row-bootstrap CI > 0;
  * ORACLE GATE (10th registered directional prediction) — gold-branch dP =
    gold-fire correct rate minus in-job baseline correct rate, CI > 0;
  * PRIMARY (if oracle passes and the selector is not demoted) — gauge-select
    dP CI > 0 AND paired (gauge-select correct − self-consistency@8 correct)
    CI > 0;
  * branch texture: argmax-gold rate, selector-write interference (gauge-
    selected non-gold branches), gold-instability (gold branch answers-fired).
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from wikihop_w1_gates import boot_ci, r_norm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_w2_gates.json"))
    p.add_argument("--selector-demoted", action="store_true")
    args = p.parse_args()
    recs = [json.loads(l) for f in args.jsonl for l in open(f)]
    rows = sorted({r["id"] for r in recs})
    base = defaultdict(list)
    for r in recs:
        if r["condition"] == "baseline":
            base[r["id"]].append(r)
    fired = [r for r in recs if r["fired_candidate"] is not None]
    audit = {"n_rows": len(rows), "n_fired_records": len(fired),
             "n_zero_prefill_hook_calls": sum((r["hook_prefill_calls"] or 0) < 1 for r in fired),
             "n_zero_positions_written": sum((r["hook_positions_written"] or 0) < 1 for r in fired),
             "n_gauge_forward_unhooked": sum((r.get("gauge_forward_hook_calls") or 0) < 1 for r in fired),
             "rungs_seen": sorted({r["rung"] for r in fired})}
    audit["valid"] = all(audit[k] == 0 for k in ("n_zero_prefill_hook_calls", "n_zero_positions_written", "n_gauge_forward_unhooked"))

    base_rate, base_ans, sc_correct = {}, {}, {}
    for i in rows:
        b = base[i]
        base_rate[i] = float(np.mean([x["correct"] for x in b])) if b else float("nan")
        cnt = Counter(x["normalized_output"] for x in b)
        base_ans[i] = {k: v / len(b) for k, v in cnt.items()} if b else {}
        modal = cnt.most_common(1)[0][0] if b else None
        sc_correct[i] = float(next(x["correct"] for x in b if x["normalized_output"] == modal)) if b else float("nan")

    gold_c, nongold_af, branches = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in fired:
        if r["fired_is_gold"]:
            gold_c[r["id"]].append(r["correct"])
        else:
            nongold_af[(r["id"], r["fired_candidate"])].append(r["answers_fired"])
        if r["condition"] == "percand":
            branches[r["id"]][r["fired_candidate"]].append(r)
    lift_rows = defaultdict(list)
    for (i, c), v in nongold_af.items():
        lift_rows[i].append(float(np.mean(v)) - base_ans[i].get(r_norm(c), 0.0))
    lifts = [float(np.mean(lift_rows[i])) for i in sorted(lift_rows)]
    delivery = {"n_rows": len(lifts), "fingerprint_lift": float(np.mean(lifts)) if lifts else None,
                "ci": boot_ci(lifts), "pass": bool(lifts and boot_ci(lifts)[0] > 0),
                "nongold_answers_fired_rate": float(np.mean([np.mean(v) for v in nongold_af.values()])) if nongold_af else None}

    oracle_rows = sorted(i for i in gold_c if i in base_rate)
    dp = [float(np.mean(gold_c[i])) - base_rate[i] for i in oracle_rows]
    oracle = {"n_rows": len(dp), "gold_correct_rate": float(np.mean([np.mean(gold_c[i]) for i in oracle_rows])) if oracle_rows else None,
              "baseline_correct_rate": float(np.mean([base_rate[i] for i in oracle_rows])) if oracle_rows else None,
              "dP": float(np.mean(dp)) if dp else None, "ci": boot_ci(dp), "pass": bool(dp and boot_ci(dp)[0] > 0)}

    gs_dp, gs_minus_sc, argmax_gold, sel_signal, interference, gold_af = [], [], [], [], [], []
    for i in rows:
        br = branches.get(i)
        if not br:
            continue
        best = max(br, key=lambda c: br[c][0]["gauge_score"])
        best_rate = float(np.mean([x["correct"] for x in br[best]]))
        gs_dp.append(best_rate - base_rate[i])
        gs_minus_sc.append(best_rate - sc_correct[i])
        is_gold = br[best][0]["fired_is_gold"]
        argmax_gold.append(bool(is_gold))
        golds = [c for c in br if br[c][0]["fired_is_gold"]]
        non = [br[c][0]["gauge_score"] for c in br if not br[c][0]["fired_is_gold"]]
        if golds and non:
            sel_signal.append(br[golds[0]][0]["gauge_score"] - float(np.mean(non)))
            gold_af.append(float(np.mean([x["answers_fired"] for x in br[golds[0]]])))
        if not is_gold:
            interference.append(float(np.mean([x["answers_fired"] for x in br[best]])))
    primary = {"n_rows": len(gs_dp), "gauge_select_dP": float(np.mean(gs_dp)) if gs_dp else None, "gauge_select_dP_ci": boot_ci(gs_dp),
               "gauge_minus_sc8": float(np.mean(gs_minus_sc)) if gs_minus_sc else None, "gauge_minus_sc8_ci": boot_ci(gs_minus_sc),
               "sc8_correct_rate": float(np.mean([sc_correct[i] for i in rows if i in branches])) if branches else None,
               "argmax_gold_rate": float(np.mean(argmax_gold)) if argmax_gold else None,
               "selection_signal": float(np.mean(sel_signal)) if sel_signal else None, "selection_signal_ci": boot_ci(sel_signal),
               "selected_nongold_answers_fired_rate": float(np.mean(interference)) if interference else None,
               "gold_branch_answers_fired_rate": float(np.mean(gold_af)) if gold_af else None,
               "selector_demoted": bool(args.selector_demoted)}
    primary["pass"] = bool(oracle["pass"] and not args.selector_demoted and gs_dp
                           and boot_ci(gs_dp)[0] > 0 and boot_ci(gs_minus_sc)[0] > 0)
    if not audit["valid"]:
        verdict = "EXECUTION-INVALID (delivery audit)"
    elif not delivery["pass"]:
        verdict = "DELIVERY-GATE-FAIL (fingerprint absent — execution-invalid by rule)"
    elif not oracle["pass"]:
        verdict = "WRITE-FAILS (10th prediction refuted: oracle dP CI straddles/negative)"
    elif args.selector_demoted:
        verdict = "ORACLE-PASS / SELECTOR-DEMOTED (descriptive gauge-select reported)"
    elif primary["pass"]:
        verdict = "PRIMARY PASS (fresh-draw replication obligation on a disjoint dev draw)"
    else:
        verdict = "ORACLE-PASS / SELECTOR-FAILS-OR-WEAK"
    out = {"delivery_audit": audit, "delivery_gate": delivery, "oracle_gate": oracle, "primary": primary, "verdict": verdict}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
