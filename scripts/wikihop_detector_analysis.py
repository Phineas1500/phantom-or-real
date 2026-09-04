#!/usr/bin/env python3
"""Descriptive (post-hoc, unregistered): how far can a blind failure detector
close the gap to the oracle two-stage rule on the WD (anonymized) and WE
(real-text) blind frames? Per-row features use only what a deployment has
before any label: the unsteered gauge scores, std self-consistency,
closed-book agreement, whether the baseline answer is a whole-word span of
the documents / a listed candidate, and (optionally) the loop's own branch
acceptance profile. Detectors are logistic regressions weighted to frame
stratum proportions, scored (a) cross-frame (fit on one frame, applied to
the other) and (b) within-frame five-fold out-of-fold. Each detector then
drives the two-stage rule of scripts/wikihop_two_stage_selector.py and is
scored by frame net against the oracle."""
from __future__ import annotations
import argparse, collections, gzip, json, re, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import normalize_answer  # noqa: E402
from wikihop_loop_descriptives import RS, load, rows_from  # noqa: E402
from wikihop_two_stage_selector import stage2  # noqa: E402

FRAMES = {
    "WD": dict(pins="docs/wikihop_wd_pinned.json", frame=RS / "wikihop_anon2_input.jsonl.gz", grades=RS / "wikihop_wo_grades.jsonl",
               rep=["wikihop_wo_a.jsonl", "wikihop_wo_b.jsonl"], unrep=["wikihop_wd_y1.jsonl", "wikihop_wd_y2.jsonl"], col=["wikihop_wd_c.jsonl"],
               rep_key="repairable_rows_from_WO", n_tot=507, gauge_main="primary_L48", gauge_real="second_L38"),
    "WE": dict(pins="docs/wikihop_we_pinned.json", frame=RS / "wikihop_fresh_input.jsonl.gz", grades=RS / "wikihop_wf_grades.jsonl",
               rep=["wikihop_wx_a.jsonl", "wikihop_wx_b.jsonl"], unrep=["wikihop_we_y1.jsonl", "wikihop_we_y2.jsonl"], col=["wikihop_we_c.jsonl"],
               rep_key="repairable_rows_from_WX", n_tot=800, gauge_main="primary_L38", gauge_real="primary_L38"),
}
BASE_FEATS = ["gauge_main", "gauge_real", "sc_share", "sc_distinct", "cb_agree", "cb_share", "in_docs", "in_cands", "log_n_cands"]
LOOP_FEATS = ["af_base", "af_max_other", "af_margin", "n_top_tied"]


def whole_word(cand, text):
    return re.search(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE) is not None


def build(name):
    cfg = FRAMES[name]; pins = json.load(open(cfg["pins"]))
    frame = {r["id"]: r for r in (json.loads(l) for l in gzip.open(cfg["frame"], "rt"))}
    grades = collections.defaultdict(list)
    for l in open(cfg["grades"]):
        r = json.loads(l); grades[r["id"], r["arm"]].append(normalize_answer(r["model_output"]))
    strata = {"repairable": (cfg["rep"], set(pins[cfg["rep_key"]])), "unrepairable": (cfg["unrep"], set(pins["yield_rows"])), "correct": (cfg["col"], set(pins["collateral_rows"]))}
    rows, keys_seen = {}, collections.Counter()
    for stratum, (files, ids) in strata.items():
        recs = load([RS / f for f in files])
        scalar = {r["id"]: r.get("base_gauge_score") for r in recs if r["condition"] == "baseline"}
        R = rows_from(recs, tie_key="primary_L38" if name == "WE" else "second_L38")
        for i, row in R.items():
            if i not in ids:
                continue
            fr = frame[i]; bg = row["base_gauge"]; keys_seen.update(bg.keys())
            std, closed = grades[i, "std"], grades[i, "closed"]
            modal = collections.Counter(std).most_common(1)[0][0]
            cands_norm = {normalize_answer(c): c for c in fr["candidates"]}
            af_by_norm = {normalize_answer(c): v["af"] for c, v in row["cands"].items()}
            af_base = af_by_norm.get(modal, 0.0)
            others = [v for k, v in af_by_norm.items() if k != modal]
            af_max_other = max(others) if others else 0.0
            top = max(af_by_norm.values()) if af_by_norm else 0.0
            f = dict(gauge_main=bg.get(cfg["gauge_main"], scalar.get(i) or 0.0), gauge_real=bg.get(cfg["gauge_real"], scalar.get(i) or 0.0),
                     sc_share=collections.Counter(std).most_common(1)[0][1] / len(std), sc_distinct=len(set(std)) / len(std),
                     cb_agree=sum(1 for x in closed if x == modal) / len(closed), cb_share=collections.Counter(closed).most_common(1)[0][1] / len(closed),
                     in_docs=float(whole_word(modal, fr["docs"])) if modal else 0.0, in_cands=float(modal in cands_norm), log_n_cands=float(np.log(len(fr["candidates"]))),
                     af_base=af_base, af_max_other=af_max_other, af_margin=af_base - af_max_other, n_top_tied=float(sum(1 for v in af_by_norm.values() if v == top and top > 0)))
            rows[i] = dict(row, stratum=stratum, feats=f, fail=int(stratum != "correct"), rep=int(stratum == "repairable"))
    pool = pins["pool"]
    n_frame = {"repairable": pool["n_hint_repairable"], "unrepairable": pool["n_unrepairable"], "correct": pool["n_correct_majority"]}
    n_samp = collections.Counter(r["stratum"] for r in rows.values())
    for r in rows.values():
        r["w"] = n_frame[r["stratum"]] / n_samp[r["stratum"]]
    return rows, pool, cfg, dict(keys_seen), n_samp


def matrix(rows, feats, target="fail"):
    ids = sorted(rows)
    X = np.array([[rows[i]["feats"][f] for f in feats] for i in ids]); y = np.array([rows[i][target] for i in ids]); w = np.array([rows[i]["w"] for i in ids])
    return ids, X, y, w


def fit(X, y, w):
    sc = StandardScaler().fit(X); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y, sample_weight=w)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1], dict(zip(range(X.shape[1]), m.coef_[0].round(3).tolist()))


def oof_scores(rows, feats, target="fail", seed=20260901):
    ids, X, y, w = matrix(rows, feats, target); rng = np.random.default_rng(seed); folds = np.zeros(len(ids), int)
    for s in ("repairable", "unrepairable", "correct"):
        idx = [k for k, i in enumerate(ids) if rows[i]["stratum"] == s]; rng.shuffle(idx)
        for j, k in enumerate(idx): folds[k] = j % 5
    p = np.zeros(len(ids))
    for f in range(5):
        tr, te = folds != f, folds == f
        pred, _ = fit(X[tr], y[tr], w[tr]); p[te] = pred(X[te])
    return dict(zip(ids, p))


def frame_net(rows, pool, n_tot, flagged, variant, rng=None):
    w_rep, w_unrep = pool["weight_repairable"], pool["weight_unrepairable"]
    d = {s: [] for s in ("repairable", "unrepairable", "correct")}
    for i, r in rows.items():
        if flagged[i]:
            a, _ = stage2(r, variant); d[r["stratum"]].append(a - r["base"])
        else:
            d[r["stratum"]].append(0.0)
    d = {s: np.array(v) for s, v in d.items()}
    def net(dd):
        y = w_rep * dd["repairable"].mean() + w_unrep * dd["unrepairable"].mean()
        return pool["n_doc_dependent"] / n_tot * y + pool["n_correct_majority"] / n_tot * dd["correct"].mean(), y, dd["correct"].mean()
    point = net(d)
    if rng is None:
        return point, None
    draws = [net({s: v[rng.integers(0, len(v), len(v))] for s, v in d.items()})[0] for _ in range(3000)]
    return point, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def flag_rates(rows, flagged):
    c = collections.defaultdict(list)
    for i, r in rows.items(): c[r["stratum"]].append(flagged[i])
    return {s: float(np.mean(v)) for s, v in c.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_detector_analysis.json"))
    args = p.parse_args()
    data = {n: build(n) for n in FRAMES}
    out = {"frames": {}, "auroc": {}, "detectors": []}
    for n, (rows, pool, cfg, keys, n_samp) in data.items():
        out["frames"][n] = {"n_rows": len(rows), "strata": dict(n_samp), "gauge_keys": keys}
        print(f"\n== {n}: {len(rows)} rows {dict(n_samp)}; gauge keys {keys}")
        print("| feature | mean failing (rep / unrep) | mean correct | weighted AUROC (failure vs correct) |\n|---|---|---|---|")
        ids, X, y, w = matrix(rows, BASE_FEATS + LOOP_FEATS)
        out["auroc"][n] = {}
        for j, f in enumerate(BASE_FEATS + LOOP_FEATS):
            x = X[:, j]
            if np.std(x) == 0:
                continue
            auc = roc_auc_score(y, x if f in ("gauge_main", "gauge_real", "cb_agree", "cb_share", "sc_share", "in_docs", "in_cands", "af_base", "af_margin") else x, sample_weight=w)
            auc = max(auc, 1 - auc)
            rep = np.mean([rows[i]["feats"][f] for i in ids if rows[i]["stratum"] == "repairable"]); un = np.mean([rows[i]["feats"][f] for i in ids if rows[i]["stratum"] == "unrepairable"])
            co = np.mean([rows[i]["feats"][f] for i in ids if rows[i]["stratum"] == "correct"])
            out["auroc"][n][f] = float(auc); print(f"| {f} | {rep:+.3f} / {un:+.3f} | {co:+.3f} | {auc:.3f} |")
    rng = np.random.default_rng(20260902)
    TAUS = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]
    print("\n| frame | detector | flags rep / unrep / correct | yield weighted | collateral | frame net [CI] answer | frame net abstain | oracle answer / abstain |\n|---|---|---|---|---|---|---|---|")
    for n, (rows, pool, cfg, keys, n_samp) in data.items():
        other = "WE" if n == "WD" else "WD"; orows, opool, ocfg = data[other][0], data[other][1], data[other][2]
        oracle = {v: frame_net(rows, pool, cfg["n_tot"], {i: rows[i]["fail"] == 1 for i in rows}, v)[0][0] for v in ("answer", "abstain")}
        oracle_rep = frame_net(rows, pool, cfg["n_tot"], {i: rows[i]["rep"] == 1 for i in rows}, "answer")[0][0]
        entries = [("gauge<0 (two-stage baseline)", {i: rows[i]["feats"]["gauge_main"] < 0 for i in rows}, None),
                   ("oracle repairable-only flag", {i: rows[i]["rep"] == 1 for i in rows}, None)]
        for target in ("fail", "rep"):
            for label, feats in (("baseline-only", BASE_FEATS), ("baseline+loop", BASE_FEATS + LOOP_FEATS)):
                tag = f"{label} → {target}"
                oids, oX, oy, ow = matrix(orows, feats, target); pred, coef = fit(oX, oy, ow)
                ooof = oof_scores(orows, feats, target)
                tau_star = max(TAUS, key=lambda t: frame_net(orows, opool, ocfg["n_tot"], {i: ooof[i] >= t for i in oids}, "answer")[0][0])
                ids, X, y, w = matrix(rows, feats, target); px = dict(zip(ids, pred(X)))
                entries.append((f"{tag} · cross-frame fit on {other}, τ*={tau_star} chosen there", {i: px[i] >= tau_star for i in ids},
                                {"coef": dict(zip(feats, coef.values())), "auroc": float(roc_auc_score(y, [px[i] for i in ids], sample_weight=w)), "tau": tau_star}))
                oof = oof_scores(rows, feats, target)
                entries.append((f"{tag} · within-frame 5-fold, τ=0.5", {i: oof[i] >= 0.5 for i in ids}, {"auroc": float(roc_auc_score(y, [oof[i] for i in ids], sample_weight=w))}))
                best = max(TAUS, key=lambda t: frame_net(rows, pool, cfg["n_tot"], {i: oof[i] >= t for i in ids}, "answer")[0][0])
                entries.append((f"{tag} · within-frame 5-fold, best τ={best} (optimistic)", {i: oof[i] >= best for i in ids}, {"tau": best}))
        for label, flagged, extra in entries:
            fr = flag_rates(rows, flagged)
            (net_a, y_a, c_a), ci = frame_net(rows, pool, cfg["n_tot"], flagged, "answer", rng)
            (net_b, y_b, c_b), _ = frame_net(rows, pool, cfg["n_tot"], flagged, "abstain")
            rec = {"frame": n, "detector": label, "flag_rates": fr, "yield_weighted_answer": y_a, "collateral_answer": c_a, "frame_net_answer": net_a, "frame_net_answer_ci": ci,
                   "frame_net_abstain": net_b, "oracle_answer": oracle["answer"], "oracle_abstain": oracle["abstain"], "oracle_repairable_only": oracle_rep, "extra": extra}
            out["detectors"].append(rec)
            au = f" (AUROC {extra['auroc']:.3f})" if extra and "auroc" in extra else ""
            print(f"| {n} | {label}{au} | {fr['repairable']:.0%} / {fr['unrepairable']:.0%} / {fr['correct']:.0%} | {y_a:+.3f} | {c_a:+.3f} | **{net_a:+.3f}** [{ci[0]:+.3f}, {ci[1]:+.3f}] | {net_b:+.3f} | {oracle['answer']:+.3f} / {oracle['abstain']:+.3f} (rep-only {oracle_rep:+.3f}) |")
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
