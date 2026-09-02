#!/usr/bin/env python3
"""Item WO — the output-level selector. On frozen-write loop records (WX/WA-rider
design with multi-gauge scores), read per row: OUTPUT selection = the branch
whose k outputs most often equal its own fired candidate (ties → primary gauge);
GAUGE selection (primary = anonymized-fit gauge; second_L38 = real-text);
oracle; random branch; baseline; SC@8. 15th prediction: paired (output − primary
gauge) CI > 0. Also the write consistency (gold-address dP, specificity)."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wo_gates.json"))
    p.add_argument("--rung", type=float, default=2.0)
    p.add_argument("--real-key", default="second_L38")
    p.add_argument("--tie-key", default="second_L38", help="gauge used to break answers-fired ties (registered: the real-text L38 gauge)")
    args = p.parse_args()
    recs = [json.loads(l) for f in args.jsonl for l in open(f)]
    rows = sorted({r["id"] for r in recs})
    R = {"out": [], "gauge": [], "real": [], "oracle": [], "random": [], "base": [], "sc8": [], "out_gold": [], "gauge_gold": [], "real_gold": [], "nb": []}
    gold_dp, spec = {1.0: [], 2.0: []}, {1.0: [], 2.0: []}
    for i in rows:
        base = [r for r in recs if r["id"] == i and r["condition"] == "baseline"]
        base_rate = float(np.mean([x["correct"] for x in base]))
        modal = Counter(x["normalized_output"] for x in base).most_common(1)[0][0]
        sc8 = float(next(x["correct"] for x in base if x["normalized_output"] == modal))
        for rung in (1.0, 2.0):
            g = [r for r in recs if r["id"] == i and r["condition"] == "delta_write" and r["rung"] == rung and r["fired_is_gold"]]
            n = [r for r in recs if r["id"] == i and r["condition"] == "delta_write" and r["rung"] == rung and not r["fired_is_gold"]]
            if g:
                gold_dp[rung].append(float(np.mean([x["correct"] for x in g])) - base_rate)
                if n:
                    by = defaultdict(list)
                    for x in n:
                        by[x["fired_candidate"]].append(x["correct"])
                    spec[rung].append(float(np.mean([x["correct"] for x in g])) - float(np.mean([np.mean(v) for v in by.values()])))
        br = defaultdict(list)
        for r in recs:
            if r["id"] == i and r["condition"] == "delta_write" and r["rung"] == args.rung:
                br[r["fired_candidate"]].append(r)
        cands = list(br)
        gold = [c for c in cands if br[c][0]["fired_is_gold"]][0]
        corr = lambda c: float(np.mean([x["correct"] for x in br[c]]))
        gsc = lambda c: br[c][0]["gauge_score"]
        rsc = lambda c: br[c][0]["gauge_scores"][args.real_key] if br[c][0].get("gauge_scores") else br[c][0]["gauge_score"]
        af = lambda c: float(np.mean([x["answers_fired"] for x in br[c]]))
        tsc = (lambda c: br[c][0]["gauge_scores"][args.tie_key]) if args.tie_key != "primary" else gsc
        o_best = max(cands, key=lambda c: (af(c), tsc(c)))
        g_best = max(cands, key=gsc)
        r_best = max(cands, key=rsc)
        R["out"].append(corr(o_best)); R["gauge"].append(corr(g_best)); R["real"].append(corr(r_best))
        R["oracle"].append(corr(gold)); R["random"].append(float(np.mean([corr(c) for c in cands]))); R["base"].append(base_rate); R["sc8"].append(sc8)
        R["out_gold"].append(o_best == gold); R["gauge_gold"].append(g_best == gold); R["real_gold"].append(r_best == gold); R["nb"].append(len(cands))
    m = lambda k: float(np.mean(R[k]))
    d_or = [a - b for a, b in zip(R["out"], R["real"])]   # 15th prediction: vs the real-text gauge (the chain's registered selector)
    d_og = [a - b for a, b in zip(R["out"], R["gauge"])]  # secondary: vs the primary (anonymized-fit) gauge
    out = {"n_rows": len(rows), "mean_branches": m("nb"),
           "write_consistency": {str(k): {"gold_dP": float(np.mean(v)), "ci": boot_ci(v), "specificity": float(np.mean(spec[k])), "specificity_ci": boot_ci(spec[k])} for k, v in gold_dp.items() if v},
           "selectors": {"output_level": {"select": m("out"), "gold_argmax": m("out_gold"), "vs_baseline_ci": boot_ci([a - b for a, b in zip(R["out"], R["base"])]),
                                          "vs_random_ci": boot_ci([a - b for a, b in zip(R["out"], R["random"])]), "vs_sc8_ci": boot_ci([a - b for a, b in zip(R["out"], R["sc8"])]),
                                          "oracle_recovered": m("out") / max(m("oracle"), 1e-9)},
                         "primary_gauge": {"select": m("gauge"), "gold_argmax": m("gauge_gold"), "oracle_recovered": m("gauge") / max(m("oracle"), 1e-9)},
                         "real_text_gauge": {"select": m("real"), "gold_argmax": m("real_gold"), "oracle_recovered": m("real") / max(m("oracle"), 1e-9)},
                         "oracle": m("oracle"), "random_branch": m("random"), "baseline": m("base"), "sc8": m("sc8")},
           "prediction_15": {"paired_output_minus_real_gauge": float(np.mean(d_or)), "ci": boot_ci(d_or), "pass": bool(boot_ci(d_or)[0] > 0)},
           "secondary_output_minus_primary_gauge": {"mean": float(np.mean(d_og)), "ci": boot_ci(d_og)},
           "coverage": {"rows_decided_by_output_signal_alone": int(sum(1 for i in range(len(rows)) if R["nb"][i] and R["out"][i] is not None)) }}
    out["verdict"] = "OUTPUT-SELECTOR-BEATS-GAUGE" if out["prediction_15"]["pass"] else "OUTPUT-SELECTOR-NOT-BETTER"
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
