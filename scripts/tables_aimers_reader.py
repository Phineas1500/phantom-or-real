#!/usr/bin/env python3
"""Item WN reader: rank the rows of each document under every label-free
aimer and read gold-row recall@1/@3 per stratum; retrieval heads are chosen
cross-fit (shard A's correct rows pick the heads scored on shard B, and vice
versa); calibrated attention divides each span's per-token mass by the
frame-average mass at its relative position. Also the detector reading
(retrieval-head peak mass, attention entropy: AUROC failure vs correct)."""
import argparse
import gzip
import json
import re
from collections import Counter, defaultdict

import numpy as np

BAR = 0.50; REFERENCE_BM25 = 0.23; N_HEADS = 16; N_BINS = 20


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def f1(a, b):
    ca, cb = Counter(a), Counter(b); common = sum((ca & cb).values())
    if not common:
        return 0.0
    p, r = common / max(1, sum(ca.values())), common / max(1, sum(cb.values())); return 2 * p * r / (p + r)


def rank_of(scores, gold):
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable"); return int(np.where(order == gold)[0][0]) + 1


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if not len(x):
        return [None, None]
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]; return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", nargs="+", required=True); ap.add_argument("--pins", default="docs/wn_pinned.json")
    ap.add_argument("--frame", default="results/loop_screen/wtq_haystack_input.jsonl.gz"); ap.add_argument("--out", default="docs/wn_gates.json"); a = ap.parse_args()
    pins = json.load(open(a.pins)); strata = pins["strata"]; shard_of = {i: s for s, ids in pins["jobs"].items() for i in ids}
    frame = {}
    for line in gzip.open(a.frame, "rt"):
        r = json.loads(line); frame[r["id"]] = r
    recs = {}
    for p in a.records:
        for line in open(p):
            r = json.loads(line); recs[r["id"]] = r
    print(f"records {len(recs)} of {len(strata)} pinned rows")
    # frame-average positional profile of per-token global-mean attention (label-free)
    per_tok = {}; bins = defaultdict(list)
    for rid, r in recs.items():
        G = r["attn_global_heads"]
        if not G:
            continue
        gm = np.mean([np.mean(np.array(v), 0) for v in G.values()], 0)  # spans, mean over global layers and heads
        cnt = np.maximum(np.array(r["span_token_counts"], float), 1); pt = gm / cnt; per_tok[rid] = (gm, pt)
        for s, rp in enumerate(r["span_rel_pos"]):
            if rp is not None:
                bins[min(N_BINS - 1, int(rp * N_BINS))].append(pt[s])
    profile = {b: float(np.mean(v)) for b, v in bins.items()}
    # retrieval heads, cross-fit on correct rows: heads by mean gold-span share of attention
    heads = {}
    for fit_shard, apply_shard in (("A", "B"), ("B", "A")):
        share = defaultdict(list)
        for rid, r in recs.items():
            if shard_of.get(rid) != fit_shard or strata[rid] != "correct" or not r["attn_global_heads"]:
                continue
            g = r["gold_span_index"]
            for L, v in r["attn_global_heads"].items():
                v = np.array(v)
                for h in range(v.shape[0]):
                    tot = v[h].sum(); share[(L, h)].append(v[h][g] / tot if tot > 0 else 0.0)
        ranked = sorted(share, key=lambda k: -np.mean(share[k])); heads[apply_shard] = ranked[:N_HEADS]
        print(f"retrieval heads fit on shard {fit_shard} correct rows (n={len(next(iter(share.values()), []))}): top {[(L, h, round(float(np.mean(share[(L, h)])), 3)) for L, h in ranked[:5]]}")
    ranks = defaultdict(dict); detector = {}
    for rid, r in recs.items():
        fr = frame[rid]; g = r["gold_span_index"]; n = r["n_spans"]
        bm = fr["retrieved_rows_in_top_table"]; ranks["bm25"][rid] = (bm.index(g) + 1) if g in bm else n
        span_text = [fr["docs"][sp["start"]:sp["end"]] for sp in fr["spans"]]; st = toks(r["selection"])
        ranks["self_selection"][rid] = rank_of([f1(st, toks(t)) for t in span_text], g) if st else n
        ranks["cp"][rid] = rank_of(np.array(r["cp_with_q"]) - np.array(r["cp_without_q"]), g)
        lm = np.array(r["attn_layer_mean"]); ranks["attn_all_layers"][rid] = rank_of(lm.mean(0), g) if lm.shape[0] > 2 else n
        if rid in per_tok:
            gm, pt = per_tok[rid]; ranks["attn_raw"][rid] = rank_of(gm, g)
            cal = [pt[s] / profile[min(N_BINS - 1, int(rp * N_BINS))] if rp is not None else 0.0 for s, rp in enumerate(r["span_rel_pos"])]; ranks["attn_calibrated"][rid] = rank_of(cal, g)
            hs = heads.get(shard_of.get(rid), [])
            if hs:
                m = np.mean([np.array(r["attn_global_heads"][L])[h] for L, h in hs], 0); ranks["retrieval_heads"][rid] = rank_of(m, g)
                tot = m.sum(); detector[rid] = {"rh_peak": float(m.max() / tot) if tot > 0 else 0.0, "rh_gold_share": float(m[g] / tot) if tot > 0 else 0.0}
            p = gm / gm.sum() if gm.sum() > 0 else np.ones(n) / n; detector.setdefault(rid, {})["entropy"] = float(-(p * np.log(p + 1e-12)).sum())
            detector[rid]["doc_mass"] = float(gm.sum())
        else:
            ranks["attn_raw"][rid] = ranks["attn_calibrated"][rid] = ranks["retrieval_heads"][rid] = n
    aimers = ["bm25", "self_selection", "attn_raw", "attn_calibrated", "attn_all_layers", "retrieval_heads", "cp"]
    out = {"n": len(recs), "profile_bins": profile, "retrieval_heads": {s: [[L, h] for L, h in v] for s, v in heads.items()}, "recall": {}, "union": {}, "detector": {}}
    for stratum in ("fixable", "unfixable", "correct"):
        ids = [i for i in recs if strata[i] == stratum]; out["recall"][stratum] = {"n": len(ids)}
        for am in aimers:
            rk = np.array([ranks[am][i] for i in ids]); r1, r3 = (rk <= 1).astype(float), (rk <= 3).astype(float)
            out["recall"][stratum][am] = {"recall_at_1": round(float(r1.mean()), 3), "recall_at_3": round(float(r3.mean()), 3), "ci_at_3": boot_ci(r3), "median_rank": float(np.median(rk))}
        u3 = np.array([min(ranks[am][i] for am in aimers[1:]) <= 3 for i in ids], float); out["union"][stratum] = {"recall_at_3_any_model_aimer": round(float(u3.mean()), 3)}
        print(f"{stratum:10s} n={len(ids):3d} " + " ".join(f"{am}={out['recall'][stratum][am]['recall_at_1']:.2f}/{out['recall'][stratum][am]['recall_at_3']:.2f}" for am in aimers) + f" union@3={u3.mean():.2f}")
    for key in ("rh_peak", "rh_gold_share", "entropy", "doc_mass"):
        val = {s: [detector[i][key] for i in recs if strata[i] == s and key in detector.get(i, {})] for s in ("fixable", "unfixable", "correct")}
        out["detector"][key] = {"mean": {s: round(float(np.mean(v)), 4) if v else None for s, v in val.items()},
                                "auroc_failure_vs_correct": auroc(val["fixable"] + val["unfixable"], val["correct"]), "auroc_fixable_vs_correct": auroc(val["fixable"], val["correct"]), "auroc_fixable_vs_unfixable": auroc(val["fixable"], val["unfixable"])}
        print(f"detector {key}: means {out['detector'][key]['mean']} AUROC fail/correct {out['detector'][key]['auroc_failure_vs_correct']}")
    fx = out["recall"]["fixable"]; best = max(aimers[1:], key=lambda am: fx[am]["recall_at_3"]); lo = fx[best]["ci_at_3"][0]
    out["verdict_37"] = {"best_aimer": best, "recall_at_3": fx[best]["recall_at_3"], "ci": fx[best]["ci_at_3"], "bar": BAR, "reference_bm25": REFERENCE_BM25,
                         "confirmed": bool(fx[best]["recall_at_3"] >= BAR and lo is not None and lo > REFERENCE_BM25)}
    print("37th:", json.dumps(out["verdict_37"]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
