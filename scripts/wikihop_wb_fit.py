#!/usr/bin/env python3
"""Item WB — the branch gauge. Inputs: two capture-only artifacts (branch
final-token states under the frozen write, per shard) and the WG records
(the same branches' outputs). Consistency: recompute the real-text L38 gauge
score from each captured state and match WG's recorded 'second_L38'. Fit:
cross-fit by shard (label = branch correct rate >= 0.5), centered logistic
C=1.0, layer = argmax donor-shard 5-fold CV AUC; BRANCH NATURAL GATE >= 0.65.
Readings: loop selection on the WG branches under the branch gauge vs the
real-text gauge (14th prediction, paired CI > 0) and vs the anonymized-fit
gauge (descriptive); LOO-over-rows variant (descriptive)."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def fit_gauge(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    mean = X.mean(axis=0)
    clf = LogisticRegression(max_iter=3000, C=C, solver="liblinear").fit(X - mean, y)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0]), mean


def cv_auc(X, y, seed=20260837, folds=5):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y):
        w, b, m = fit_gauge(X[tr], y[tr])
        oof[te] = (X[te] - m) @ w + b
    return float(roc_auc_score(y, oof))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, nargs=2, required=True, help="capture npz for job A and job B")
    p.add_argument("--manifest", type=Path, nargs=2, required=True)
    p.add_argument("--wg-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--real-pins", type=Path, default=Path("results/loop_screen/wikihop_w0_pinned.npz"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wb_gates.json"))
    p.add_argument("--loop-rung", type=float, default=2.0)
    args = p.parse_args()
    # --- branches from WG records: outputs, correctness, recorded scores
    recs = [json.loads(l) for f in args.wg_jsonl for l in open(f)]
    br = defaultdict(lambda: defaultdict(list))
    base_rate, sc8 = {}, {}
    for r in recs:
        if r["condition"] == "delta_write" and r["rung"] == args.loop_rung:
            br[r["id"]][r["fired_candidate"]].append(r)
    for i in br:
        b = [r for r in recs if r["id"] == i and r["condition"] == "baseline"]
        base_rate[i] = float(np.mean([x["correct"] for x in b]))
        from collections import Counter
        modal = Counter(x["normalized_output"] for x in b).most_common(1)[0][0]
        sc8[i] = float(next(x["correct"] for x in b if x["normalized_output"] == modal))
    # --- captured states
    real = np.load(args.real_pins)
    rw, rb, rm = real["gauge_w_L38"].astype(np.float64), float(real["gauge_b_L38"][0]), real["gauge_mean_L38"].astype(np.float64)
    shards = {}
    for cap, man in zip(args.capture, args.manifest):
        z = np.load(cap); m = json.load(open(man))
        layers = [int(x) for x in z["cap_layers"]]
        X = z["branch_states"].astype(np.float64)  # [n, n_layers, D]
        rows_here = sorted({b["id"] for b in m["branches"]})
        keys, y, rec_real, mism = [], [], [], 0
        for k, b in enumerate(m["branches"]):
            rr = br[b["id"]].get(b["fired_candidate"])
            if rr is None:
                mism += 1
                continue
            keys.append((b["id"], b["fired_candidate"], k))
            y.append(int(np.mean([x["correct"] for x in rr]) >= 0.5))
            s_rec = rr[0]["gauge_scores"]["second_L38"]
            s_cap = float((X[k, layers.index(38)] - rm) @ rw + rb)
            rec_real.append((s_rec, s_cap))
        diffs = np.array([abs(a - b) for a, b in rec_real])
        shards[m["wx_job"]] = {"X": X, "layers": layers, "keys": keys, "y": np.array(y), "rows": rows_here,
                               "consistency": {"n": len(diffs), "max_abs_diff": float(diffs.max()), "median_abs_diff": float(np.median(diffs)),
                                               "rel_max": float(diffs.max() / max(np.abs([a for a, _ in rec_real]).mean(), 1e-9)), "n_unmatched": mism}}
    consistency_ok = all(v["consistency"]["rel_max"] < 1e-2 and v["consistency"]["n_unmatched"] == 0 for v in shards.values())
    # --- cross-fit by shard (job A tested shard 1 with donors shard 0; the branch gauge for A's rows is fit on B's rows and vice versa)
    per_layer, sel = {}, {}
    out_rows = {}
    for test_job, donor_job in (("A", "B"), ("B", "A")):
        D, T = shards[donor_job], shards[test_job]
        best = None
        cvs = {}
        for li, L in enumerate(D["layers"]):
            Xd = D["X"][[k for _, _, k in D["keys"]], li]
            cvs[L] = cv_auc(Xd, D["y"])
        Lstar = max(cvs, key=cvs.get)
        li = D["layers"].index(Lstar)
        w, b, m = fit_gauge(D["X"][[k for _, _, k in D["keys"]], li], D["y"])
        per_layer[test_job] = {"donor_cv_auc": cvs, "layer": Lstar, "n_donor_branches": int(len(D["y"])), "n_donor_pos": int(D["y"].sum()),
                               "branch_natural_gate_pass": bool(cvs[Lstar] >= 0.65)}
        for (rid, cand, k) in T["keys"]:
            sel.setdefault(rid, {})[cand] = float((T["X"][k, T["layers"].index(Lstar)] - m) @ w + b)
    def loop_with(score_of):
        rows = sorted(sel)
        outc, argmax_gold, oracle, rnd = [], [], [], []
        for i in rows:
            cands = list(br[i])
            best = max(cands, key=lambda c: score_of(i, c))
            gold = [c for c in cands if br[i][c][0]["fired_is_gold"]][0]
            outc.append(float(np.mean([x["correct"] for x in br[i][best]])))
            argmax_gold.append(best == gold)
            oracle.append(float(np.mean([x["correct"] for x in br[i][gold]])))
            rnd.append(float(np.mean([np.mean([x["correct"] for x in br[i][c]]) for c in cands])))
        return rows, outc, argmax_gold, oracle, rnd
    rows, bg, bg_am, orc, rnd = loop_with(lambda i, c: sel[i][c])
    _, rt, rt_am, _, _ = loop_with(lambda i, c: br[i][c][0]["gauge_scores"]["second_L38"])
    _, an, an_am, _, _ = loop_with(lambda i, c: br[i][c][0]["gauge_score"])  # WG primary = anonymized-fit L48
    paired_real = [a - b for a, b in zip(bg, rt)]
    paired_anon = [a - b for a, b in zip(bg, an)]
    readings = {"n_rows": len(rows),
                "branch_gauge": {"gauge_select": float(np.mean(bg)), "argmax_gold": float(np.mean(bg_am)), "oracle": float(np.mean(orc)),
                                 "oracle_recovered": float(np.mean(bg) / max(np.mean(orc), 1e-9)),
                                 "vs_baseline_ci": boot_ci([bg[k] - base_rate[i] for k, i in enumerate(rows)]),
                                 "vs_random_ci": boot_ci([bg[k] - rnd[k] for k in range(len(rows))]),
                                 "vs_sc8_ci": boot_ci([bg[k] - sc8[i] for k, i in enumerate(rows)])},
                "real_text_gauge": {"gauge_select": float(np.mean(rt)), "argmax_gold": float(np.mean(rt_am))},
                "anonymized_fit_gauge": {"gauge_select": float(np.mean(an)), "argmax_gold": float(np.mean(an_am))},
                "paired_branch_minus_real": {"mean": float(np.mean(paired_real)), "ci": boot_ci(paired_real), "pass": bool(boot_ci(paired_real)[0] > 0)},
                "paired_branch_minus_anon": {"mean": float(np.mean(paired_anon)), "ci": boot_ci(paired_anon), "pass": bool(boot_ci(paired_anon)[0] > 0)}}
    # descriptive: leave-one-row-out over all 60 rows, at each layer
    allX, ally, allkeys, layers = [], [], [], shards["A"]["layers"]
    for j in ("A", "B"):
        S = shards[j]
        for (rid, cand, k) in S["keys"]:
            allX.append(S["X"][k]); allkeys.append((rid, cand))
        ally.extend(S["y"].tolist())
    allX, ally = np.stack(allX), np.array(ally)
    loo = {}
    for li, L in enumerate(layers):
        scores = {}
        for i in rows:
            tr = np.array([kk for kk, (rid, _) in enumerate(allkeys) if rid != i])
            w, b, m = fit_gauge(allX[tr, li], ally[tr])
            for kk, (rid, cand) in enumerate(allkeys):
                if rid == i:
                    scores.setdefault(i, {})[cand] = float((allX[kk, li] - m) @ w + b)
        _, lc, lam, _, _ = loop_with(lambda i, c: scores[i][c])
        loo[f"L{L}"] = {"gauge_select": float(np.mean(lc)), "argmax_gold": float(np.mean(lam)),
                        "paired_minus_real_ci": boot_ci([lc[k] - rt[k] for k in range(len(rows))])}
    gates = {k: v["branch_natural_gate_pass"] for k, v in per_layer.items()}
    if not consistency_ok:
        verdict = "EXECUTION-INVALID (captured states do not reproduce WG's recorded gauge scores)"
    elif readings["paired_branch_minus_real"]["pass"]:
        verdict = "BRANCH-GAUGE-CLOSES-THE-GAP"
    else:
        verdict = "SELECTOR-CEILING"
    out = {"consistency": {j: shards[j]["consistency"] for j in shards}, "consistency_ok": consistency_ok,
           "fit": per_layer, "branch_natural_gate": gates, "readings": readings, "loo_by_layer_descriptive": loo, "verdict": verdict}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
