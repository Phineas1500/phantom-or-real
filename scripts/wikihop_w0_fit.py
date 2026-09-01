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
    lab_ids = [r["id"] for r in rows if r["std_majority"] is not None]
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
    gold_vec = {}
    for m in man_rows:
        for c in m["candidates"]:
            if c["candidate"] == frame[m["id"]]["answer"]:
                gold_vec[m["id"]] = cand_vecs[c["vec_index"]]
    donors_pos = sorted(r["id"] for r in rows if r["std_majority"] == 1 and not r["doc_dependent_failing"] and r["id"] in gold_vec)
    donors_neg = sorted(r["id"] for r in rows if r["std_majority"] == 0 and not r["doc_dependent_failing"] and r["id"] in gold_vec)
    rng = random.Random(SEED_FRAME)
    n_bal = min(len(donors_pos), len(donors_neg))
    donors_pos_b = sorted(rng.sample(donors_pos, n_bal))
    donors_neg_b = sorted(rng.sample(donors_neg, n_bal))
    class_vec = np.stack([gold_vec[i] for i in donors_pos_b]).mean(0) - np.stack([gold_vec[i] for i in donors_neg_b]).mean(0)
    all_norms = np.linalg.norm(cand_vecs, axis=1)
    single_pos = np.array([c["n_positions"] == 1 for m in man_rows for c in m["candidates"]])
    base_norm = float(all_norms.mean())
    write = {"n_donors_correct_available": len(donors_pos), "n_donors_incorrect_available": len(donors_neg),
             "n_per_class_balanced": n_bal, "class_vector_norm": float(np.linalg.norm(class_vec)),
             "class_vector_cos_with_gauge_w": float(class_vec @ w / (np.linalg.norm(class_vec) * np.linalg.norm(w) + 1e-12)),
             "amplitude_base_mean_candidate_state_norm": base_norm,
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
    np.savez(args.out_npz, gauge_w=w, gauge_b=np.array([b]), gauge_mean=mean, gauge_layer=np.array([primary]),
             gauge_cv_auc=np.array([gauge[L] for L in CAP_LAYERS]), cap_layers=np.array(CAP_LAYERS),
             class_vector=class_vec, base_norm=np.array([base_norm]), write_layer=np.array([man["write_layer"]]))
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
              "pools": {"failing_zero_of_8": failing, "doc_dependent_failing": doc_dep,
                        "w1_rows": w1_rows, "w2_pool": w2_pool, "n_w2_pool": len(w2_pool)},
              "artifacts": {"rows": str(args.out_rows), "npz": str(args.out_npz)}}
    args.out_json.write_text(json.dumps(pinned, indent=1) + "\n")
    print(f"W1 rows: {w1_rows}\nW2 pool: {len(w2_pool)} rows\nwrote {args.out_npz} {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
