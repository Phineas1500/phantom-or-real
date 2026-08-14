"""Local analysis for the Qwen3.8 gauge hunt (mission qwen38-gauge).

Scores the generation arms with the house scorer and trains the per-layer
logistic probe ladder on the 65-layer final-prompt-token capture. Labels come
from either the temp-0.7 k=1 arm (gauge_labels.jsonl) or the greedy rider
(gauge_labels_greedy.jsonl); the greedy protocol matches stage-1 and is the
comparable one. All inputs/outputs live in results/qwen38_gauge/.

  python -m scripts.qwen38_gauge_analysis --behavioral
  python -m scripts.qwen38_gauge_analysis --ladder --labels greedy --C 3e-4
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stage2_steering import score_reply

GAUGE_DIR = Path(__file__).resolve().parent.parent / "results" / "qwen38_gauge"
STAGE1 = (Path(__file__).resolve().parent.parent /
          "results/full/with_errortype/gemma3_27b_infer_property.jsonl")


def load_source_rows():
    meta = json.loads((GAUGE_DIR / "gauge_capture_meta.json").read_text())
    need = set(meta["order"])
    rows = {}
    with open(STAGE1) as f:
        for i, line in enumerate(f):
            if i in need:
                rows[i] = json.loads(line)
    return meta, rows


def score_arm(fname, src_rows):
    per = defaultdict(list)
    for line in open(GAUGE_DIR / fname):
        r = json.loads(line)
        s = score_reply(src_rows[r["row_index"]], r["model_output"])
        per[r["row_index"]].append(int(s["is_correct_strong"]))
    return per


def behavioral():
    meta, src = load_source_rows()
    lab = {ri: v[0] for ri, v in score_arm("gauge_labels.jsonl", src).items()}
    hin = {ri: sum(v) / len(v) for ri, v in score_arm("gauge_hinted.jsonl", src).items()}
    n = len(lab)
    fail = [ri for ri in hin if lab[ri] == 0]
    out = {
        "n_rows": n,
        "unhinted_p_strong": round(sum(lab.values()) / n, 4),
        "unhinted_by_height": {h: round(sum(lab[ri] for ri in lab if src[ri]["height"] == h)
                                        / sum(1 for ri in lab if src[ri]["height"] == h), 4)
                               for h in (3, 4)},
        "n_hinted_rows": len(hin),
        "hinted_p_strong": round(sum(hin.values()) / len(hin), 4),
        "n_unhinted_failing_in_hinted_set": len(fail),
        "hint_lift_on_failing": round(sum(hin[ri] for ri in fail) / len(fail), 4) if fail else None,
    }
    (GAUGE_DIR / "behavioral.json").write_text(json.dumps(out, indent=1))
    (GAUGE_DIR / "labels_by_row.json").write_text(json.dumps(lab))
    greedy_path = GAUGE_DIR / "gauge_labels_greedy.jsonl"
    if greedy_path.exists():
        gre = {ri: v[0] for ri, v in score_arm("gauge_labels_greedy.jsonl", src).items()}
        (GAUGE_DIR / "labels_greedy_by_row.json").write_text(json.dumps(gre))
        out["greedy_p_strong"] = round(sum(gre.values()) / len(gre), 4)
        out["greedy_temp_label_agreement"] = round(
            sum(1 for k in gre if gre[k] == lab[k]) / len(gre), 4)
    print(json.dumps(out, indent=1))


def ladder(label_kind, C):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    meta = json.loads((GAUGE_DIR / "gauge_capture_meta.json").read_text())
    lf = "labels_greedy_by_row.json" if label_kind == "greedy" else "labels_by_row.json"
    labels = json.loads((GAUGE_DIR / lf).read_text())
    y = np.array([labels[str(ri)] for ri in meta["order"]])
    H = np.load(GAUGE_DIR / "gauge_hidden_final.npy", mmap_mode="r")
    cfg = json.loads((GAUGE_DIR / "q38_config.json").read_text())
    lt = cfg.get("text_config", cfg)["layer_types"]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260815)
    results = []
    for li in range(H.shape[0]):
        X = np.asarray(H[li], dtype=np.float32)
        aucs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=3000, tol=1e-4)
            clf.fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
        rec = {"hs_index": li, "layer": li - 1,
               "type": "embed" if li == 0 else lt[li - 1],
               "auc_mean": round(float(np.mean(aucs)), 4),
               "auc_folds": [round(float(a), 4) for a in aucs]}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    out = GAUGE_DIR / f"probe_ladder_{label_kind}_C{C:g}.json"
    out.write_text(json.dumps(results, indent=1))
    print("BEST:", json.dumps(max(results, key=lambda r: r["auc_mean"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioral", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--labels", choices=["temp", "greedy"], default="greedy")
    ap.add_argument("--C", type=float, default=3e-4)
    args = ap.parse_args()
    if args.behavioral:
        behavioral()
    if args.ladder:
        ladder(args.labels, args.C)


if __name__ == "__main__":
    main()
