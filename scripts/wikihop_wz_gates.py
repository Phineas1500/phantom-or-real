#!/usr/bin/env python3
"""Item WZ reader: label-free detectors from the literature, scored as
registered. Per frame: labels from stage-1 rows (correct-majority ≥ 5/8 = 0,
0/8 failure = 1; others excluded), features from the WZ job (P(True) of the
modal answer; answer-token states per layer × pooling). Probes are logistic
regressions fit on the frame's non-blind rows and scored on its blind rows
(also cross-frame). Each detector reports AUROC and recall on failures at
the threshold giving ≤ 5% false positives on the training rows. The
deployment reading runs the two-stage rule on the blind rows' loop records
with flag = groundedness ∨ detector, selectors = output vote / argmax P(True)
among non-baseline branches; rerankers without the loop = argmax P(True) and
the CAD score over all candidates. Verdicts: 33rd (pooled conflict rows),
34th (WE rows, stratum-weighted)."""
from __future__ import annotations
import argparse, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402
from wikihop_common import normalize_answer  # noqa: E402
from wikihop_wk_gates import per_row, whole_word, best  # noqa: E402

RS = Path("results/loop_screen")
FRAMES = {
    "nqswap": dict(rows="nqswap_rows.jsonl", frame="wk_stage2_input.jsonl.gz", scores="wz_nqswap_scores.jsonl", states="wz_nqswap_states.npz",
                   loop=["wikihop_wk_xa.jsonl", "wikihop_wk_xb.jsonl", "wikihop_wkp_xa.jsonl", "wikihop_wkp_xb.jsonl"], weighted=None),
    "squadcf": dict(rows="squad_cf_rows.jsonl", frame="ws_stage2_input.jsonl.gz", scores="wz_squadcf_scores.jsonl", states="wz_squadcf_states.npz",
                    loop=["wikihop_ws_xa.jsonl", "wikihop_ws_xb.jsonl"], weighted=None),
    "wikihop": dict(rows="wikihop_wf_rows.jsonl", frame="wikihop_fresh_input.jsonl.gz", scores="wz_wikihop_scores.jsonl", states="wz_wikihop_states.npz",
                    loop=["wikihop_wx_a.jsonl", "wikihop_wx_b.jsonl", "wikihop_we_y1.jsonl", "wikihop_we_y2.jsonl", "wikihop_we_c.jsonl"], weighted="docs/wikihop_we_pinned.json"),
}
POOLINGS = ("mean", "last", "prompt")


def load_frame(name):
    cfg = FRAMES[name]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(RS / cfg["rows"])}
    frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(RS / cfg["frame"], "rt")}
    scores = {json.loads(l)["id"]: json.loads(l) for l in open(RS / cfg["scores"])}
    z = np.load(RS / cfg["states"]); sid = list(z["ids"]); states = {k: z[k].astype(np.float32) for k in z.files if k != "ids"}
    idx = {i: k for k, i in enumerate(sid)}
    lab = {}
    for i, r in rows.items():
        if r["std_n_correct"] >= 5: lab[i] = 0
        elif r["std_n_correct"] == 0: lab[i] = 1
    ids = [i for i in scores if i in lab and i in idx]
    blind = [i for i in ids if scores[i]["blind"]]; train = [i for i in ids if not scores[i]["blind"]]
    return dict(rows=rows, frame=frame, scores=scores, states=states, idx=idx, lab=lab, ids=ids, blind=blind, train=train, cfg=cfg)


def feat(D, key, ids):
    if key == "p_true":
        return np.array([[-D["scores"][i]["p_true_modal"]] for i in ids])
    return np.stack([D["states"][key][D["idx"][i]] for i in ids])


def fit_score(Dtr, Dte, key, tr_ids, te_ids):
    Xtr, ytr = feat(Dtr, key, tr_ids), np.array([Dtr["lab"][i] for i in tr_ids]); Xte, yte = feat(Dte, key, te_ids), np.array([Dte["lab"][i] for i in te_ids])
    if key == "p_true":
        s_tr, s_te = Xtr[:, 0], Xte[:, 0]
    else:
        sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(Xtr), ytr)
        s_tr, s_te = m.decision_function(sc.transform(Xtr)), m.decision_function(sc.transform(Xte))
    thr = float(np.percentile(s_tr[ytr == 0], 95))
    auc = float(roc_auc_score(yte, s_te)) if len(set(yte)) > 1 else float("nan")
    rec = float(np.mean(s_te[yte == 1] >= thr)) if (yte == 1).any() else float("nan"); fpr = float(np.mean(s_te[yte == 0] >= thr)) if (yte == 0).any() else float("nan")
    return dict(auroc=auc, recall_at_5fpr=rec, fpr=fpr, thr=thr, scores=dict(zip(te_ids, s_te.tolist())))


def two_stage(R, flagged, selector, docs, pt):
    vals = []
    for i, r in R.items():
        c = r["cands"]
        if not flagged[i] or not c:
            vals.append(0.0); continue
        non_base = [x for x in c if normalize_answer(x) != r["base_modal"]]
        if not non_base:
            vals.append(0.0); continue
        if selector == "vote":
            top = max(v["af"] for v in c.values()); tied = [x for x in non_base if c[x]["af"] == top]
            if top == 0 or not tied:
                vals.append(0.0); continue
            pick = tied[0] if len(tied) == 1 else max(tied, key=lambda x: c[x]["gauge"])
        else:
            ptc = pt.get(i, {})
            scored = [(ptc.get(normalize_answer(x), {}).get("p_true", -1e9), x) for x in non_base]
            pick = max(scored)[1]
        vals.append(c[pick]["corr"] - r["base"])
    return np.array(vals)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wz_gates.json"))
    args = p.parse_args()
    D = {n: load_frame(n) for n in FRAMES}
    out = {"detectors": {}, "deployment": {}}
    keys = ["p_true"] + [f"L{L}_{pl}" for L in (20, 30, 38, 43, 48, 53) for pl in POOLINGS]
    print("| frame (train → test) | detector | AUROC | recall on failures @≤5% FPR (train thr) | FPR on test |\n|---|---|---|---|---|")
    for n in FRAMES:
        out["detectors"][n] = {}
        for key in keys:
            if key != "p_true" and key not in D[n]["states"]:
                continue
            r = fit_score(D[n], D[n], key, D[n]["train"], D[n]["blind"]); out["detectors"][n][key] = {k: v for k, v in r.items() if k != "scores"}; out["detectors"][n][key]["scores"] = r["scores"]
            print(f"| {n} (own train → blind) | {key} | {r['auroc']:.3f} | {r['recall_at_5fpr']:.2f} | {r['fpr']:.2f} |")
    for a, b in (("nqswap", "squadcf"), ("squadcf", "nqswap"), ("nqswap", "wikihop")):
        for key in ("p_true", "L30_mean", "L38_mean", "L43_last", "L48_mean"):
            if key != "p_true" and key not in D[a]["states"]:
                continue
            r = fit_score(D[a], D[b], key, D[a]["train"], D[b]["blind"]); out["detectors"][f"{a}->{b}"] = out["detectors"].get(f"{a}->{b}", {}); out["detectors"][f"{a}->{b}"][key] = {k: v for k, v in r.items() if k != "scores"}
            print(f"| {a} → {b} blind | {key} | {r['auroc']:.3f} | {r['recall_at_5fpr']:.2f} | {r['fpr']:.2f} |")
    # best detector on the pooled conflict blind rows (own-frame fits), by AUROC
    pooled_lab, pooled_scores = {}, defaultdict(dict)
    for n in ("nqswap", "squadcf"):
        for i in D[n]["blind"]:
            pooled_lab[(n, i)] = D[n]["lab"][i]
            for key, r in out["detectors"][n].items():
                if i in r["scores"]: pooled_scores[key][(n, i)] = r["scores"][i]
    pooled_auc = {key: float(roc_auc_score([pooled_lab[k] for k in s], [s[k] for k in s])) for key, s in pooled_scores.items() if len(s) == len(pooled_lab)}
    best_key = max(pooled_auc, key=pooled_auc.get); out["pooled_conflict_auroc"] = pooled_auc; out["best_detector"] = best_key
    print(f"\nbest detector on pooled conflict blind rows: {best_key} AUROC {pooled_auc[best_key]:.3f}; P(True) {pooled_auc.get('p_true', float('nan')):.3f}")
    # deployment
    print("\n| frame | rule | frame net [CI] | up / down | paired vs grounded-vote [CI] |\n|---|---|---|---|---|")
    pooled_pairs = []
    for n in FRAMES:
        cfg = D[n]["cfg"]; R = per_row([json.loads(l) for f in cfg["loop"] for l in open(RS / f)])
        R = {i: r for i, r in R.items() if i in D[n]["scores"] and D[n]["scores"][i]["blind"]}
        docs = {i: D[n]["frame"][i]["docs"].lower() for i in R}
        pt = {i: {normalize_answer(c): v for c, v in D[n]["scores"][i].get("candidates", {}).items()} for i in R}
        ung = {i: not whole_word(R[i]["base_modal"], docs[i]) for i in R}
        det = out["detectors"][n][best_key]; thr = det["thr"]; dscore = det["scores"]
        flags = {"grounded": ung, "grounded∨detector": {i: ung[i] or dscore.get(i, -1e9) >= thr for i in R}, "detector only": {i: dscore.get(i, -1e9) >= thr for i in R},
                 "oracle": {i: D[n]["lab"].get(i, 0) == 1 for i in R}}
        weights = None
        if cfg["weighted"]:
            pins = json.load(open(cfg["weighted"])); pool = pins["pool"]; strat = {}
            for i in R:
                strat[i] = "rep" if i in set(pins["repairable_rows_from_WX"]) else "unrep" if i in set(pins["yield_rows"]) else "cor"
            def net(vals):
                v = {s: [vals[k] for k, i in enumerate(R) if strat[i] == s] for s in ("rep", "unrep", "cor")}
                y = pool["weight_repairable"] * np.mean(v["rep"]) + pool["weight_unrepairable"] * np.mean(v["unrep"])
                return pool["n_doc_dependent"] / pool["n_rows"] * y + pool["n_correct_majority"] / pool["n_rows"] * np.mean(v["cor"])
            weights = net
        ref = two_stage(R, flags["grounded"], "vote", docs, pt)
        for fname, fl in flags.items():
            for sel in ("vote", "p_true"):
                vals = two_stage(R, fl, sel, docs, pt)
                if weights:
                    rng = np.random.default_rng(20260921); ids = list(R)
                    draws = []
                    for _ in range(2000):
                        # stratified bootstrap
                        samp = []
                        for s in ("rep", "unrep", "cor"):
                            members = [k for k, i in enumerate(ids) if strat[i] == s]; samp += list(rng.choice(members, len(members)))
                        vv = np.zeros(len(ids)); 
                        vsub = {}
                        draws.append(np.mean([0]))  # placeholder replaced below
                    fn = weights(vals); ci = None
                    # stratified CI via per-stratum bootstrap of means
                    draws = []
                    for _ in range(3000):
                        vs = {}
                        for s in ("rep", "unrep", "cor"):
                            m = np.array([vals[k] for k, i in enumerate(ids) if strat[i] == s]); vs[s] = m[rng.integers(0, len(m), len(m))].mean()
                        y = pool["weight_repairable"] * vs["rep"] + pool["weight_unrepairable"] * vs["unrep"]
                        draws.append(pool["n_doc_dependent"] / pool["n_rows"] * y + pool["n_correct_majority"] / pool["n_rows"] * vs["cor"])
                    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
                else:
                    fn = float(vals.mean()); ci = boot_ci(list(vals))
                d = vals - ref; pci = boot_ci(list(d))
                key = f"{fname} / {sel}"
                out["deployment"].setdefault(n, {})[key] = {"frame_net": float(fn), "ci95": ci, "up": int((vals > 0).sum()), "down": int((vals < 0).sum()), "paired_vs_grounded_vote": float(d.mean()), "paired_ci": pci}
                if n != "wikihop" and fname == "grounded∨detector" and sel == "vote": pooled_pairs += list(d)
                print(f"| {n} | {key} | **{fn:+.3f}** [{ci[0]:+.3f}, {ci[1]:+.3f}] | {int((vals > 0).sum())} / {int((vals < 0).sum())} | {d.mean():+.3f} [{pci[0]:+.3f}, {pci[1]:+.3f}] |")
        # rerankers
        base = np.array([R[i]["base"] for i in R])
        rr = {}
        for label, score in (("argmax P(True)", lambda c, v: v["p_true"]), ("CAD α=0.5", lambda c, v: 1.5 * v["logp_ctx"] - 0.5 * v["logp_noctx"])):
            acc = []
            for i in R:
                cands = D[n]["scores"][i].get("candidates", {})
                if not cands: acc.append(R[i]["base"]); continue
                pick = max(cands, key=lambda c: score(c, cands[c])); acc.append(float(normalize_answer(pick) == normalize_answer(D[n]["frame"][i]["answer"])))
            acc = np.array(acc); rr[label] = {"accuracy": float(acc.mean()), "vs_baseline": float((acc - base).mean()), "ci": boot_ci(list(acc - base))}
            print(f"| {n} | reranker: {label} (no loop) | {acc.mean():.3f} (baseline {base.mean():.3f}) | | {rr[label]['vs_baseline']:+.3f} [{rr[label]['ci'][0]:+.3f}, {rr[label]['ci'][1]:+.3f}] |")
        out["deployment"][n]["rerankers"] = rr
    pooled_ci = boot_ci(pooled_pairs) if pooled_pairs else [float("nan")] * 2
    out["prediction_33"] = {"best_detector": best_key, "pooled_auroc": pooled_auc[best_key], "pooled_paired_gain_vs_grounded": float(np.mean(pooled_pairs)) if pooled_pairs else None, "ci": pooled_ci,
                            "verdict": "CONFIRMED" if pooled_auc[best_key] >= 0.90 and pooled_ci[0] > 0 else "NOT CONFIRMED"}
    wh = out["deployment"].get("wikihop", {}).get("grounded∨detector / vote")
    if wh:
        out["prediction_34"] = {"frame_net": wh["frame_net"], "ci": wh["ci95"], "verdict": "CONFIRMED" if wh["ci95"][0] > 0 else "NOT CONFIRMED"}
    print(f"\n33rd: best detector {best_key} pooled AUROC {pooled_auc[best_key]:.3f}; paired gain vs grounded rule on pooled conflict rows {np.mean(pooled_pairs):+.3f} {pooled_ci} → {out['prediction_33']['verdict']}")
    if wh: print(f"34th: WikiHop (WE rows) grounded∨detector / vote frame net {wh['frame_net']:+.3f} {wh['ci95']} → {out['prediction_34']['verdict']}")
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
