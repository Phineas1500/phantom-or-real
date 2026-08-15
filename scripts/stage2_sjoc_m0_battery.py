#!/usr/bin/env python3
"""Item M0 offline battery (docs/causal_handle_directions.md, item M) — frozen
before the capture is read. Order enforced in code: (a) the natural sanity
gate STOPs the battery when it fails; (e) demotes T1 to descriptive; (d)
failure demotes T1 branches to descriptive (SJ/OC-ENTANGLED-AT-SITE).

Implementation choices frozen here, pre-data:
- Battery runs at L53 (primary); other stack layers reported descriptively.
- 4-fold CV stratified by the four cells, shuffle seed 20260816.
- dir_SJ per train fold = mean over OC levels of (mean SJ=yes - mean SJ=no)
  within that OC level; dir_OC symmetric. Decodability = held-out AUC of the
  projection onto the unit direction.
- T1 gauge polarity is the frozen L0 convention (higher score = predicted
  correct; asserted against train folds, never flipped). The yes/no threshold
  is the grand mean of train-fold gauge scores (cells balanced by design, so
  the threshold is neutral to SJ and OC).
- T1 agreement difference: per held-out conflict row, agree(SJ) - agree(OC)
  in {-1, +1}; plain bootstrap over conflict rows (one state per row), 10k
  draws, seed 20260704, percentile CI.
- (d) erasure re-fits directions on erased train states and evaluates on
  erased test states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def cell_dir(X: np.ndarray, a: np.ndarray, b: np.ndarray, within: np.ndarray) -> np.ndarray:
    parts = []
    for w in (True, False):
        m = within == w
        if (m & a).any() and (m & b).any():
            parts.append(X[m & a].mean(axis=0) - X[m & b].mean(axis=0))
    return np.mean(parts, axis=0)


def boot_ci(vals: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> list[float]:
    draws = [float(np.mean(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=Path("results/stage2/sjoc_m0_states.npz"))
    parser.add_argument("--gauge-npz", type=Path,
                        default=Path("results/stage2/erasure/inlp_direction_stacks_27b_property_5layer.npz"))
    parser.add_argument("--layer", type=int, default=53)
    parser.add_argument("--out", type=Path, default=Path("docs/sjoc_m0_battery.json"))
    args = parser.parse_args()

    d = np.load(args.npz)
    layers = list(d["layers"])
    li = layers.index(args.layer)
    X = d["states"][:, li, :].astype(np.float64)
    oc = d["oc"].astype(bool)
    sj = d["sj_yes"].astype(bool)
    cells = np.array([f"oc{int(o)}_sj{'yes' if s else 'no'}" for o, s in zip(oc, sj)])
    gauge_unit = unit(np.load(args.gauge_npz)[f"L{args.layer}_inlp_stack"][0].astype(np.float64))
    gscore = X @ gauge_unit

    report: dict = {"n": int(len(X)), "layer": args.layer,
                    "decision_rule": "item M, docs/causal_handle_directions.md"}

    auc_oc_frame = float(roc_auc_score(oc, gscore))
    report["gate_a_natural_sanity"] = {"gauge_auc_oc": auc_oc_frame, "threshold": 0.75,
                                       "pass": auc_oc_frame >= 0.75}
    if not report["gate_a_natural_sanity"]["pass"]:
        report["verdict"] = "STOP: natural sanity gate failed - pipeline suspect"
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=20260816)
    conflict = oc != sj
    sj_aucs, oc_aucs = [], []
    sj_aucs_erased, oc_aucs_moved = [], []
    oc_aucs_erased, sj_aucs_moved = [], []
    t1_rows: list[float] = []
    t1_agree_sj = []
    for tr, te in skf.split(X, cells):
        Xtr, Xte = X[tr], X[te]
        dsj = unit(cell_dir(Xtr, sj[tr], ~sj[tr], oc[tr]))
        doc = unit(cell_dir(Xtr, oc[tr], ~oc[tr], sj[tr]))
        sj_aucs.append(roc_auc_score(sj[te], Xte @ dsj))
        oc_aucs.append(roc_auc_score(oc[te], Xte @ doc))

        thr = float(np.mean(gscore[tr]))
        pred_correct = gscore[te] > thr
        cte = conflict[te]
        for p, o, s in zip(pred_correct[cte], oc[te][cte], sj[te][cte]):
            agree_sj = float(p == s)
            t1_agree_sj.append(agree_sj)
            t1_rows.append(agree_sj - float(p == o))

        def erased_auc(kill: np.ndarray, label_tr, label_te, within_tr):
            Etr = Xtr - np.outer(Xtr @ kill, kill)
            Ete = Xte - np.outer(Xte @ kill, kill)
            dd = unit(cell_dir(Etr, label_tr, ~label_tr, within_tr))
            return roc_auc_score(label_te, Ete @ dd)

        sj_aucs_erased.append(erased_auc(dsj, sj[tr], sj[te], oc[tr]))
        oc_aucs_moved.append(erased_auc(dsj, oc[tr], oc[te], sj[tr]))
        oc_aucs_erased.append(erased_auc(doc, oc[tr], oc[te], sj[tr]))
        sj_aucs_moved.append(erased_auc(doc, sj[tr], sj[te], oc[tr]))

    auc_sj = float(np.mean(sj_aucs))
    auc_oc = float(np.mean(oc_aucs))
    report["b_directions"] = {"heldout_auc_dir_sj": auc_sj, "heldout_auc_dir_oc": auc_oc}

    report["gate_e_sj_readability"] = {"threshold": 0.60, "auc": auc_sj, "pass": auc_sj >= 0.60}

    drop_sj = auc_sj - float(np.mean(sj_aucs_erased))
    move_oc = abs(auc_oc - float(np.mean(oc_aucs_moved)))
    drop_oc = auc_oc - float(np.mean(oc_aucs_erased))
    move_sj = abs(auc_sj - float(np.mean(sj_aucs_moved)))
    factorizes = drop_sj >= 0.10 and move_oc <= 0.03 and drop_oc >= 0.10 and move_sj <= 0.03
    report["d_differential_erasure"] = {
        "erase_dir_sj": {"sj_auc_drop": drop_sj, "oc_auc_move": move_oc},
        "erase_dir_oc": {"oc_auc_drop": drop_oc, "sj_auc_move": move_sj},
        "factorizes": factorizes,
        "note": None if factorizes else "SJ/OC-ENTANGLED-AT-SITE: T1 branches demoted to descriptive",
    }

    rng = np.random.default_rng(20260704)
    t1_vals = np.array(t1_rows)
    agree_sj_rate = float(np.mean(t1_agree_sj))
    agree_oc_rate = 1.0 - agree_sj_rate
    lo, hi = boot_ci(t1_vals, rng)
    t1 = {"n_conflict_rows": int(len(t1_vals)),
          "agreement_sj": agree_sj_rate, "agreement_oc": agree_oc_rate,
          "difference_sj_minus_oc": float(np.mean(t1_vals)), "ci95": [lo, hi]}
    if agree_sj_rate >= 0.70 and lo > 0:
        t1["branch"] = "GAUGE~SJ"
    elif agree_oc_rate >= 0.70 and hi < 0:
        t1["branch"] = "GAUGE~OC"
    else:
        t1["branch"] = "GAUGE-MIXED"
        t1["mde_observed_halfwidth"] = float((hi - lo) / 2)
    t1["status"] = ("descriptive"
                    if not report["gate_e_sj_readability"]["pass"] or not factorizes
                    else "registered")
    report["c_t1_identity_test"] = t1

    per_layer = {}
    for lj, lay in enumerate(layers):
        Xl = d["states"][:, lj, :].astype(np.float64)
        aucs_s, aucs_o = [], []
        for tr, te in skf.split(Xl, cells):
            ds = unit(cell_dir(Xl[tr], sj[tr], ~sj[tr], oc[tr]))
            do = unit(cell_dir(Xl[tr], oc[tr], ~oc[tr], sj[tr]))
            aucs_s.append(roc_auc_score(sj[te], Xl[te] @ ds))
            aucs_o.append(roc_auc_score(oc[te], Xl[te] @ do))
        per_layer[f"L{lay}"] = {"auc_sj": float(np.mean(aucs_s)), "auc_oc": float(np.mean(aucs_o))}
    report["descriptive_per_layer"] = per_layer

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
