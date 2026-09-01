#!/usr/bin/env python3
"""Item W0 offline fits (registered: docs/causal_handle_directions.md item W).

Inputs: the 800-row frame, the W0a grades (std+closed k=8) and the W0b capture
(final-token states at L38/43/48/53, candidate-mention-mean states at L30).
Pins, before W1: labels + pools (failing / doc-dependent failing), the gauge
(logistic on final-token states vs std-majority labels; primary layer = argmax
5-fold CV AUC; natural gate CV >= 0.65), the class-mean write (gold-candidate
mention-mean L30 state, correct-majority minus incorrect-majority donors OUTSIDE
the doc-dependent pool, balanced), the amplitude base (mean L30 candidate-mention
state norm), and the seeded W1 row draw (seed 20260822).
"""
from __future__ import annotations
import argparse, gzip, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import contains_match, exact_match  # noqa: E402

CAP_LAYERS = [38, 43, 48, 53]
SEED_FRAME, SEED_W1 = 20260821, 20260822


def majority_label(n_correct: int, k: int = 8):
    if n_correct * 2 > k:
        return 1
    if n_correct * 2 < k:
        return 0
    return None


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int, folds: int = 5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y):
        mean = X[tr].mean(axis=0)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear").fit(X[tr] - mean, y[tr])
        oof[te] = clf.decision_function(X[te] - mean)
    return float(roc_auc_score(y, oof)), oof


def fit_full(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    mean = X.mean(axis=0)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear").fit(X - mean, y)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0]), mean


def bootstrap_mean_ci(values, seed=SEED_FRAME, draws=2000):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(draws)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", type=Path, default=Path("results/loop_screen/wikihop_port_input.jsonl.gz"))
    p.add_argument("--grades", type=Path, default=Path("results/loop_screen/wikihop_w0_grades.jsonl"))
    p.add_argument("--capture", type=Path, default=Path("results/loop_screen/wikihop_w0_capture.npz"))
    p.add_argument("--manifest", type=Path, default=Path("results/loop_screen/wikihop_w0_capture_manifest.json"))
    p.add_argument("--out-rows", type=Path, default=Path("results/loop_screen/wikihop_w0_rows.jsonl"))
    p.add_argument("--out-npz", type=Path, default=Path("results/loop_screen/wikihop_w0_pinned.npz"))
    p.add_argument("--out-json", type=Path, default=Path("docs/wikihop_w0_pinned.json"))
    p.add_argument("--w1-rows", type=int, default=12)
    p.add_argument("--sweep-capture", type=Path, default=None,
                   help="multi-layer candidate capture npz (W0_WRITE_LAYERS job); adds class_vector_L*/base_norm_L* pins")
    p.add_argument("--sweep-manifest", type=Path, default=None)
    args = p.parse_args()

    frame = {}
    for line in gzip.open(args.frame, "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    order = list(frame)
    per = defaultdict(lambda: {"std": [], "closed": [], "std_c": [], "closed_c": [], "std_out": [], "closed_out": []})
    n_gen = 0
    for line in open(args.grades):
        g = json.loads(line)
        gold = frame[g["id"]]["answer"]
        per[g["id"]][g["arm"]].append(bool(exact_match(g["model_output"], gold)))
        per[g["id"]][g["arm"] + "_c"].append(bool(contains_match(g["model_output"], gold)))
        per[g["id"]][g["arm"] + "_out"].append(g["model_output"])
        n_gen += 1
    assert all(len(per[i]["std"]) == 8 and len(per[i]["closed"]) == 8 for i in order), "grades incomplete"

    rows = []
    for i in order:
        d = per[i]
        s, c = sum(d["std"]), sum(d["closed"])
        rows.append({"id": i, "std_n_correct": s, "closed_n_correct": c,
                     "std_n_correct_contains": sum(d["std_c"]), "closed_n_correct_contains": sum(d["closed_c"]),
                     "std_majority": majority_label(s), "failing": s == 0,
                     "doc_dependent_failing": s == 0 and c == 0,
                     "std_outputs": d["std_out"], "closed_outputs": d["closed_out"]})
    n = len(rows)
    std_acc = float(np.mean([r["std_n_correct"] / 8 for r in rows]))
    closed_acc = float(np.mean([r["closed_n_correct"] / 8 for r in rows]))
    failing = [r["id"] for r in rows if r["failing"]]
    doc_dep = [r["id"] for r in rows if r["doc_dependent_failing"]]
    below_half = [r for r in rows if r["std_n_correct"] < 4]
    behavior = {
        "n_rows": n, "n_generations": n_gen,
        "std_accuracy_exact": std_acc, "closed_accuracy_exact": closed_acc,
        "std_accuracy_contains": float(np.mean([r["std_n_correct_contains"] / 8 for r in rows])),
        "closed_accuracy_contains": float(np.mean([r["closed_n_correct_contains"] / 8 for r in rows])),
        "failing_below_half_frac": len(below_half) / n,
        "any_correct_at_8_on_below_half": float(np.mean([r["std_n_correct"] > 0 for r in below_half])) if below_half else None,
        "n_failing_zero_of_8": len(failing), "n_doc_dependent_failing": len(doc_dep),
        "doc_dependent_pool_ok": len(doc_dep) >= 150,
        "std_majority_counts": {"correct": sum(r["std_majority"] == 1 for r in rows),
                                "incorrect": sum(r["std_majority"] == 0 for r in rows),
                                "tie": sum(r["std_majority"] is None for r in rows)},
    }
    print(json.dumps(behavior, indent=1), flush=True)

    cap = np.load(args.capture)
    man = json.load(open(args.manifest))
    man_rows = man["rows"]
    assert [m["id"] for m in man_rows] == order, "capture manifest order != frame order"
    idx = {i: k for k, i in enumerate(order)}
    finite_final = np.ones(len(order), dtype=bool)
    for L in CAP_LAYERS:
        finite_final &= np.isfinite(cap[f"L{L}_final"].astype(np.float32)).all(axis=1)
    n_nonfinite_final = int((~finite_final).sum())
    lab_ids = [r["id"] for r in rows if r["std_majority"] is not None and finite_final[idx[r["id"]]]]
    y = np.array([rows[idx[i]]["std_majority"] for i in lab_ids], dtype=int)
    sel = np.array([idx[i] for i in lab_ids])
    gauge = {}
    oofs = {}
    for L in CAP_LAYERS:
        X = cap[f"L{L}_final"][sel].astype(np.float64)
        auc, oof = cv_auc(X, y, SEED_FRAME)
        gauge[L] = auc
        oofs[L] = oof
        print(f"gauge L{L}: 5-fold CV AUC {auc:.4f} (n={len(y)})", flush=True)
    primary = max(CAP_LAYERS, key=lambda L: gauge[L])
    natural_pass = gauge[primary] >= 0.65
    X = cap[f"L{primary}_final"][sel].astype(np.float64)
    w, b, mean = fit_full(X, y)
    from sklearn.metrics import roc_auc_score
    dd_mask = np.array([rows[idx[i]]["doc_dependent_failing"] for i in lab_ids])
    pos_mask = y == 1
    keep = dd_mask | pos_mask
    auc_dd_vs_correct = float(roc_auc_score(y[keep], oofs[primary][keep]))
    print(f"PRIMARY gauge layer L{primary} CV AUC {gauge[primary]:.4f} -> NATURAL GATE {'PASS' if natural_pass else 'FAIL'}; "
          f"OOF AUC doc-dependent-vs-correct-majority {auc_dd_vs_correct:.4f}", flush=True)

    cand_vecs = cap["cand_L30"].astype(np.float64)
    finite_cand = np.isfinite(cand_vecs).all(axis=1)
    n_nonfinite_cand = int((~finite_cand).sum())
    gold_vec = {}
    for m in man_rows:
        for c in m["candidates"]:
            if c["candidate"] == frame[m["id"]]["answer"] and finite_cand[c["vec_index"]]:
                gold_vec[m["id"]] = cand_vecs[c["vec_index"]]
    donors_pos = sorted(r["id"] for r in rows if r["std_majority"] == 1 and not r["doc_dependent_failing"] and r["id"] in gold_vec)
    donors_neg = sorted(r["id"] for r in rows if r["std_majority"] == 0 and not r["doc_dependent_failing"] and r["id"] in gold_vec)
    rng = random.Random(SEED_FRAME)
    n_bal = min(len(donors_pos), len(donors_neg))
    donors_pos_b = sorted(rng.sample(donors_pos, n_bal))
    donors_neg_b = sorted(rng.sample(donors_neg, n_bal))
    class_vec = np.stack([gold_vec[i] for i in donors_pos_b]).mean(0) - np.stack([gold_vec[i] for i in donors_neg_b]).mean(0)
    all_norms = np.linalg.norm(cand_vecs[finite_cand], axis=1)
    single_pos = np.array([c["n_positions"] == 1 for m in man_rows for c in m["candidates"]])[finite_cand]
    measured = [c["mean_position_norm"] for m in man_rows for c in m["candidates"] if "mean_position_norm" in c]
    literal_base = float(np.mean(measured)) if measured else float(all_norms.mean())
    if "L30_sq_mean_cand" in cap:
        sq = cap["L30_sq_mean_cand"].astype(np.float64)
        pooled = sq.mean(axis=0)
        massive = [int(d) for d in np.argsort(-pooled)[:16] if pooled[d] / pooled.sum() > 0.05]
        keep = np.ones(sq.shape[1], dtype=bool)
        keep[massive] = False
        base_norm = float(np.sqrt(sq[:, keep].sum(axis=1)).mean())
        literal_rms = float(np.sqrt(sq.sum(axis=1)).mean())
    else:
        sq2 = (cand_vecs[finite_cand] ** 2).mean(axis=0)
        massive = [int(d) for d in np.argsort(-sq2)[:16] if sq2[d] / sq2.sum() > 0.05]
        keep = np.ones(len(sq2), dtype=bool)
        keep[massive] = False
        base_norm = float(np.linalg.norm(cand_vecs[finite_cand][:, keep], axis=1).mean())
        literal_rms = None
    write = {"capture_dtype": str(cap["cand_L30"].dtype), "n_nonfinite_final_rows_dropped": n_nonfinite_final,
             "n_nonfinite_candidate_vectors_dropped": n_nonfinite_cand,
             "amplitude_base_source": ("mean per-position L30 RMS norm at candidate-mention positions EXCLUDING the massive-activation dims"
                                       if "L30_sq_mean_cand" in cap else "norm of mention-mean vectors excluding massive dims (fallback)"),
             "massive_dims_excluded": massive,
             "massive_dims_share_of_norm_sq": float(1 - (cap["L30_sq_mean_cand"].astype(np.float64).mean(0)[keep].sum() / cap["L30_sq_mean_cand"].astype(np.float64).mean(0).sum())) if "L30_sq_mean_cand" in cap else None,
             "literal_base_mean_position_norm": literal_base, "literal_base_rms": literal_rms,
             "literal_rung_multiples_of_base": [round(m * literal_base / base_norm, 4) for m in (0.25, 0.5, 1.0)],
             "mean_norm_of_mention_mean_vectors": float(all_norms.mean()),
             "n_donors_correct_available": len(donors_pos), "n_donors_incorrect_available": len(donors_neg),
             "n_per_class_balanced": n_bal, "class_vector_norm": float(np.linalg.norm(class_vec)),
             "class_vector_cos_with_gauge_w": float(class_vec @ w / (np.linalg.norm(class_vec) * np.linalg.norm(w) + 1e-12)),
             "amplitude_base_pinned": base_norm,
             "amplitude_base_single_position_subset_mean_norm": float(all_norms[single_pos].mean()) if single_pos.any() else None,
             "n_candidate_vectors": int(len(all_norms)), "n_single_position_vectors": int(single_pos.sum()),
             "class_vector_norm_over_base": float(np.linalg.norm(class_vec) / base_norm),
             "amplitude_ladder": [0.25 * base_norm, 0.5 * base_norm, 1.0 * base_norm]}
    print(json.dumps(write, indent=1), flush=True)

    w1_rows = sorted(random.Random(SEED_W1).sample(sorted(doc_dep), args.w1_rows))
    w2_pool = sorted(set(doc_dep) - set(w1_rows))
    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.out_rows.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    extra = {}
    for L in CAP_LAYERS:
        wL, bL, mL = fit_full(cap[f"L{L}_final"][sel].astype(np.float64), y)
        extra[f"gauge_w_L{L}"], extra[f"gauge_b_L{L}"], extra[f"gauge_mean_L{L}"] = wL, np.array([bL]), mL
    sweep = {}
    if args.sweep_capture is not None:
        scap = np.load(args.sweep_capture)
        sman = json.load(open(args.sweep_manifest))
        assert [m["id"] for m in sman["rows"]] == order, "sweep manifest order != frame order"
        for L in sman["write_layers"]:
            vecs = scap[f"cand_L{L}"].astype(np.float64)
            fin = np.isfinite(vecs).all(axis=1)
            gv = {}
            for m in sman["rows"]:
                for c in m["candidates"]:
                    if c["candidate"] == frame[m["id"]]["answer"] and fin[c["vec_index"]]:
                        gv[m["id"]] = vecs[c["vec_index"]]
            pos_b = [i for i in donors_pos_b if i in gv]
            neg_b = [i for i in donors_neg_b if i in gv]
            cvL = np.stack([gv[i] for i in pos_b]).mean(0) - np.stack([gv[i] for i in neg_b]).mean(0)
            sqL = scap[f"L{L}_sq_mean_cand"].astype(np.float64)
            pooledL = sqL.mean(axis=0)
            massiveL = [int(d) for d in np.argsort(-pooledL)[:16] if pooledL[d] / pooledL.sum() > 0.05]
            keepL = np.ones(sqL.shape[1], dtype=bool)
            keepL[massiveL] = False
            baseL = float(np.sqrt(sqL[:, keepL].sum(axis=1)).mean())
            litL = float(np.sqrt(sqL.sum(axis=1)).mean())
            extra[f"class_vector_L{L}"], extra[f"base_norm_L{L}"] = cvL, np.array([baseL])
            sweep[f"L{L}"] = {"n_donors_per_class": [len(pos_b), len(neg_b)], "class_vector_norm": float(np.linalg.norm(cvL)),
                              "massive_dims_excluded": massiveL, "massive_share_of_norm_sq": float(1 - pooledL[keepL].sum() / pooledL.sum()),
                              "base_norm_pinned": baseL, "literal_base_rms": litL, "middle_rung_0.5x": 0.5 * baseL,
                              "cos_with_L30_class_vector": float(cvL @ class_vec / (np.linalg.norm(cvL) * np.linalg.norm(class_vec) + 1e-12))}
            print(f"sweep L{L}: |cv|={np.linalg.norm(cvL):.1f} base={baseL:.1f} (literal {litL:.1f}, massive {massiveL})", flush=True)
    np.savez(args.out_npz, gauge_w=w, gauge_b=np.array([b]), gauge_mean=mean, gauge_layer=np.array([primary]),
             gauge_cv_auc=np.array([gauge[L] for L in CAP_LAYERS]), cap_layers=np.array(CAP_LAYERS),
             class_vector=class_vec, base_norm=np.array([base_norm]), write_layer=np.array([man["write_layer"]]), **extra)
    pinned = {"registered": "docs/causal_handle_directions.md item W (2026-08-19)",
              "frame_seed": SEED_FRAME, "w1_seed": SEED_W1,
              "behavior": behavior,
              "gauge": {"cv_auc_by_layer": {f"L{L}": gauge[L] for L in CAP_LAYERS}, "primary_layer": primary,
                        "natural_gate_threshold": 0.65, "natural_gate_pass": bool(natural_pass),
                        "n_fit_rows": int(len(y)), "n_positive": int(y.sum()),
                        "oof_auc_doc_dependent_vs_correct_majority": auc_dd_vs_correct,
                        "recipe": "center + liblinear logistic C=1.0, StratifiedKFold(5, shuffle, seed 20260821); ties (4/8) excluded"},
              "write": {**write, "donors_correct": donors_pos_b, "donors_incorrect": donors_neg_b,
                        "donor_vector": "gold-candidate mention-mean L30 state (case-insensitive span match, offset-mapping addressing)"},
              "sweep_layers": sweep,
              "pools": {"failing_zero_of_8": failing, "doc_dependent_failing": doc_dep,
                        "w1_rows": w1_rows, "w2_pool": w2_pool, "n_w2_pool": len(w2_pool)},
              "artifacts": {"rows": str(args.out_rows), "npz": str(args.out_npz)}}
    args.out_json.write_text(json.dumps(pinned, indent=1) + "\n")
    print(f"W1 rows: {w1_rows}\nW2 pool: {len(w2_pool)} rows\nwrote {args.out_npz} {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
