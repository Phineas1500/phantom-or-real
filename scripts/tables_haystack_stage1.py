#!/usr/bin/env python3
"""Item WM stage-1 reader: std / closed / hint accuracy against the cell,
0/8 failures, the fixable share (hint ≥ 4/8) with a CI, retriever recall
(table top-1; gold row within the top-3 of the top table) overall, on
failures, and on fixable failures; the instrument gate."""
from __future__ import annotations
import argparse, collections, gzip, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import exact_match  # noqa: E402
from wikihop_w1_gates import boot_ci  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grades", type=Path, required=True); p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--out-rows", type=Path, required=True)
    args = p.parse_args()
    frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.frame, "rt")}
    per = collections.defaultdict(lambda: collections.defaultdict(list)); outs = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(args.grades):
        g = json.loads(l); per[g["id"]][g["arm"]].append(bool(exact_match(g["model_output"], frame[g["id"]]["answer"]))); outs[g["id"]][g["arm"]].append(g["model_output"])
    ids = [i for i in frame if i in per]; arms = ("std", "closed", "hint_first")
    rows = []
    for i in ids:
        r = frame[i]; s, c, h = (sum(per[i][a]) for a in arms)
        rows.append({"id": i, "std_n_correct": s, "closed_n_correct": c, "hint_first_n_correct": h, "failing": s == 0, "hint_repairable": h >= 4, "correct_majority": s >= 5,
                     "table_top1": r["gold_table_rank"] == 1, "row_top3": bool(r["gold_row_rank_in_top_table"] and r["gold_row_rank_in_top_table"] <= 3),
                     "std_outputs": outs[i]["std"], "hint_outputs": outs[i]["hint_first"]})
    with open(args.out_rows, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = {a: float(np.mean([sum(per[i][a]) for i in ids]) / 8) for a in arms}
    fail = [r for r in rows if r["failing"]]; rep = [r for r in fail if r["hint_repairable"]]
    hist = collections.Counter(r["hint_first_n_correct"] for r in fail)
    out = {"n_rows": len(ids), "accuracy": acc, "n_failing_zero_of_8": len(fail), "failing_share": len(fail) / len(ids), "n_correct_majority": sum(r["correct_majority"] for r in rows),
           "n_hint_repairable_failures": len(rep), "fixable_share_of_failures": len(rep) / max(len(fail), 1), "fixable_share_ci95": boot_ci([float(r["hint_repairable"]) for r in fail]) if fail else None,
           "hint_hist_on_failures": {str(k): v for k, v in sorted(hist.items())},
           "retriever": {"table_top1_all": float(np.mean([r["table_top1"] for r in rows])), "row_top3_all": float(np.mean([r["row_top3"] for r in rows])),
                         "row_top3_on_failures": float(np.mean([r["row_top3"] for r in fail])) if fail else None, "row_top3_on_fixable": float(np.mean([r["row_top3"] for r in rep])) if rep else None,
                         "row_top3_on_correct": float(np.mean([r["row_top3"] for r in rows if r["correct_majority"]]))},
           "gate_std_in_[0.10,0.90]": 0.10 <= acc["std"] <= 0.90, "gate_failures_ge_10pct": len(fail) / len(ids) >= 0.10}
    args.out.write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
