"""Item M0 offline battery (registered 2026-08-14): gates + T1 identity test.

Inputs: the M0 capture npz/manifest + the frozen gauge (row 0 of
L53_inlp_stack). All analyses 4-fold CV stratified by (OC, SJ) cell,
fold seed 20260816. T1 thresholds and directions are always fit on train
folds and evaluated held-out. Differential erasure probes: standardized
logistic regression, C=1e-3 (registered as descriptive support).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CAP = Path("results/stage2/erasure/sjoc_m0_capture_27b_property.npz")
MAN = Path("results/stage2/erasure/sjoc_m0_capture_27b_property.manifest.json")
GAUGE = Path("results/stage2/erasure/inlp_direction_stacks_27b_property_5layer.npz")
OUT = Path("docs/sjoc_m0_battery.json")


def probe_auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1e-3, max_iter=3000, tol=1e-4)
    clf.fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.decision_function(sc.transform(Xte)))


def main() -> int:
    man = json.loads(MAN.read_text())["rows"]
    X = np.load(CAP)["L53_final"].astype(np.float64)
    oc = np.array([r["oc"] for r in man])
    sj = np.array([1 if r["sj"] == "yes" else 0 for r in man])
    cell = np.array([f"{a}{b}" for a, b in zip(oc, sj)])
    gauge_dir = np.load(GAUGE)["L53_inlp_stack"][0]
    gauge_dir = gauge_dir / np.linalg.norm(gauge_dir)
    g = X @ gauge_dir

    report: dict = {"n": len(man), "cells": {c: int((cell == c).sum()) for c in sorted(set(cell))}}
    report["gate_a_natural_sanity"] = {
        "gauge_auc_for_oc": round(float(roc_auc_score(oc, g)), 4),
        "pass": bool(roc_auc_score(oc, g) >= 0.75),
    }

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=20260816)
    agree_sj, agree_oc, per_row_diff = [], [], []
    auc_sj_dir, auc_oc_dir = [], []
    erase = {"sj_drop_on_sj": [], "sj_move_on_oc": [], "oc_drop_on_oc": [], "oc_move_on_sj": []}
    for tr, te in skf.split(X, cell):
        def cellmean(o, s, idx):
            m = (oc[idx] == o) & (sj[idx] == s)
            return X[idx][m].mean(axis=0)
        dir_sj = ((cellmean(1, 1, tr) - cellmean(1, 0, tr)) + (cellmean(0, 1, tr) - cellmean(0, 0, tr))) / 2
        dir_oc = ((cellmean(1, 1, tr) - cellmean(0, 1, tr)) + (cellmean(1, 0, tr) - cellmean(0, 0, tr))) / 2
        u_sj = dir_sj / np.linalg.norm(dir_sj)
        u_oc = dir_oc / np.linalg.norm(dir_oc)
        auc_sj_dir.append(roc_auc_score(sj[te], X[te] @ u_sj))
        auc_oc_dir.append(roc_auc_score(oc[te], X[te] @ u_oc))

        thr = g[tr].mean()
        conf = te[(oc[te] != sj[te])]
        pred = (g[conf] > thr).astype(int)
        a_sj = (pred == sj[conf]).astype(float)
        a_oc = (pred == oc[conf]).astype(float)
        agree_sj.extend(a_sj); agree_oc.extend(a_oc)
        per_row_diff.extend(a_sj - a_oc)

        for tag, u, lab_main, lab_other, k_drop, k_move in (
            ("sj", u_sj, sj, oc, "sj_drop_on_sj", "sj_move_on_oc"),
            ("oc", u_oc, oc, sj, "oc_drop_on_oc", "oc_move_on_sj"),
        ):
            Xe_tr = X[tr] - np.outer(X[tr] @ u, u)
            Xe_te = X[te] - np.outer(X[te] @ u, u)
            base_main = probe_auc(X[tr], lab_main[tr], X[te], lab_main[te])
            base_other = probe_auc(X[tr], lab_other[tr], X[te], lab_other[te])
            er_main = probe_auc(Xe_tr, lab_main[tr], Xe_te, lab_main[te])
            er_other = probe_auc(Xe_tr, lab_other[tr], Xe_te, lab_other[te])
            erase[k_drop].append(base_main - er_main)
            erase[k_move].append(abs(base_other - er_other))

    rng = np.random.default_rng(20260704)
    d = np.array(per_row_diff)
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    report["gate_e_sj_readability"] = {
        "heldout_auc_dir_sj": round(float(np.mean(auc_sj_dir)), 4),
        "pass": bool(np.mean(auc_sj_dir) >= 0.60),
    }
    report["dir_oc_heldout_auc"] = round(float(np.mean(auc_oc_dir)), 4)
    report["t1_identity"] = {
        "n_conflict_rows": len(d),
        "agreement_with_sj": round(float(np.mean(agree_sj)), 4),
        "agreement_with_oc": round(float(np.mean(agree_oc)), 4),
        "diff_sj_minus_oc": round(float(d.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
    }
    a_sj, diff = np.mean(agree_sj), d.mean()
    if a_sj >= 0.70 and lo > 0:
        branch = "GAUGE~SJ"
    elif np.mean(agree_oc) >= 0.70 and hi < 0:
        branch = "GAUGE~OC"
    else:
        branch = "GAUGE-MIXED"
    report["t1_branch"] = branch
    report["d_differential_erasure"] = {k: round(float(np.mean(v)), 4) for k, v in erase.items()}
    report["d_factorization_ok"] = bool(
        np.mean(erase["sj_drop_on_sj"]) >= 0.10 and np.mean(erase["sj_move_on_oc"]) <= 0.03
        and np.mean(erase["oc_drop_on_oc"]) >= 0.10 and np.mean(erase["oc_move_on_sj"]) <= 0.03)

    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
