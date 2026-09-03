#!/usr/bin/env python3
"""Item WD — blind yield and collateral. Inputs: the WD job records (Y1, Y2 =
unrepairable doc-dependent rows; C = correct-majority rows) and the WO records
(the 47 hint-repairable rows' branches, reused). Selector = output-first
(answers-fired, real-text gauge tie-break). BLIND YIELD = stratum-weighted
(loop − baseline) over the doc-dependent pool (16th prediction: CI > 0);
COLLATERAL = loop − baseline on correct rows (non-inferiority bound −0.10);
ABSTENTION variant: answer only when the top answers-fired is unique and
>= 0.5, else keep the baseline answer."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def per_row(recs, rung=2.0, tie_key="second_L38"):
    rows = sorted({r["id"] for r in recs})
    out = {}
    for i in rows:
        base = [r for r in recs if r["id"] == i and r["condition"] == "baseline"]
        base_rate = float(np.mean([x["correct"] for x in base]))
        br = defaultdict(list)
        for r in recs:
            if r["id"] == i and r["condition"] == "delta_write" and r["rung"] == rung:
                br[r["fired_candidate"]].append(r)
        cands = list(br)
        corr = lambda c: float(np.mean([x["correct"] for x in br[c]]))
        af = lambda c: float(np.mean([x["answers_fired"] for x in br[c]]))
        tsc = lambda c: (br[c][0].get("gauge_scores") or {}).get(tie_key, br[c][0]["gauge_score"])
        best = max(cands, key=lambda c: (af(c), tsc(c)))
        top = max(af(c) for c in cands); uniq = sum(1 for c in cands if af(c) == top) == 1
        abstain = not (uniq and top >= 0.5)
        out[i] = {"base": base_rate, "loop": corr(best), "random": float(np.mean([corr(c) for c in cands])),
                  "abst": base_rate if abstain else corr(best), "abstained": abstain, "n": len(cands)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yield-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--collateral-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--repairable-jsonl", type=Path, nargs="+", required=True, help="WO records (the 47 hint-repairable rows)")
    p.add_argument("--pins", type=Path, default=Path("docs/wikihop_wd_pinned.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wd_gates.json"))
    p.add_argument("--tie-key", default="second_L38", help="gauge_scores key for the tie-break; falls back to gauge_score")
    args = p.parse_args()
    pins = json.load(open(args.pins)); w_rep, w_unrep = pins["pool"]["weight_repairable"], pins["pool"]["weight_unrepairable"]
    Y = per_row([json.loads(l) for f in args.yield_jsonl for l in open(f)], tie_key=args.tie_key)
    C = per_row([json.loads(l) for f in args.collateral_jsonl for l in open(f)], tie_key=args.tie_key)
    R = per_row([json.loads(l) for f in args.repairable_jsonl for l in open(f)], tie_key=args.tie_key)
    def stratum(d, key):
        return [d[i][key] - d[i]["base"] for i in sorted(d)]
    d_rep, d_unrep = stratum(R, "loop"), stratum(Y, "loop")
    a_rep, a_unrep = stratum(R, "abst"), stratum(Y, "abst")
    rng = np.random.default_rng(20260846)
    def weighted_ci(a, b, draws=4000):
        a, b = np.array(a), np.array(b); vals = []
        for _ in range(draws):
            vals.append(w_rep * a[rng.integers(0, len(a), len(a))].mean() + w_unrep * b[rng.integers(0, len(b), len(b))].mean())
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    yield_pt = w_rep * np.mean(d_rep) + w_unrep * np.mean(d_unrep)
    yield_ci = weighted_ci(d_rep, d_unrep)
    abst_pt = w_rep * np.mean(a_rep) + w_unrep * np.mean(a_unrep)
    abst_ci = weighted_ci(a_rep, a_unrep)
    d_col = stratum(C, "loop"); a_col = stratum(C, "abst")
    out = {"weights": {"repairable": w_rep, "unrepairable": w_unrep}, "n": {"repairable": len(R), "unrepairable": len(Y), "correct": len(C)},
           "strata": {"repairable": {"baseline": float(np.mean([R[i]["base"] for i in R])), "loop": float(np.mean([R[i]["loop"] for i in R])), "random": float(np.mean([R[i]["random"] for i in R])), "dP": float(np.mean(d_rep)), "ci": boot_ci(d_rep), "abstained_frac": float(np.mean([R[i]["abstained"] for i in R])), "abst_dP": float(np.mean(a_rep)), "abst_ci": boot_ci(a_rep)},
                      "unrepairable": {"baseline": float(np.mean([Y[i]["base"] for i in Y])), "loop": float(np.mean([Y[i]["loop"] for i in Y])), "random": float(np.mean([Y[i]["random"] for i in Y])), "dP": float(np.mean(d_unrep)), "ci": boot_ci(d_unrep), "abstained_frac": float(np.mean([Y[i]["abstained"] for i in Y])), "abst_dP": float(np.mean(a_unrep)), "abst_ci": boot_ci(a_unrep)}},
           "blind_yield": {"dP_weighted": float(yield_pt), "ci": yield_ci, "pass": bool(yield_ci[0] > 0)},
           "blind_yield_abstention": {"dP_weighted": float(abst_pt), "ci": abst_ci},
           "collateral": {"baseline": float(np.mean([C[i]["base"] for i in C])), "loop": float(np.mean([C[i]["loop"] for i in C])), "dP": float(np.mean(d_col)), "ci": boot_ci(d_col),
                          "no_collateral": bool(boot_ci(d_col)[0] > -0.10), "abstained_frac": float(np.mean([C[i]["abstained"] for i in C])), "abst_dP": float(np.mean(a_col)), "abst_ci": boot_ci(a_col)}}
    out["verdict"] = ("BLIND-LOOP-HELPS" if out["blind_yield"]["pass"] else "BLIND-LOOP-NO-YIELD") + " / " + ("NO-COLLATERAL" if out["collateral"]["no_collateral"] else "COLLATERAL-HARM")
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
