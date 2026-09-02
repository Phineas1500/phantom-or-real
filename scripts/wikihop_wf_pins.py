#!/usr/bin/env python3
"""Item WF pins: score the fresh frame's grade_hint output (std / closed /
hint-first, k=8 each), define DOC-DEPENDENT failing (0/8 std ∧ 0/8 closed) and
HINT-REPAIRABLE (>= 4/8 hint-first) rows, pin ALL doc-dependent ∧ hint-repairable
rows (no sampling) with two seeded shards. Registered: item WF."""
from __future__ import annotations
import argparse, collections, gzip, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import exact_match, normalize_answer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grades", type=Path, default=Path("results/loop_screen/wikihop_wf_grades.jsonl"))
    p.add_argument("--frame", type=Path, default=Path("results/loop_screen/wikihop_fresh_input.jsonl.gz"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wf_pinned.json"))
    p.add_argument("--out-rows", type=Path, default=Path("results/loop_screen/wikihop_wf_rows.jsonl"))
    args = p.parse_args()
    frame = {}
    for line in gzip.open(args.frame, "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    outs = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(args.grades):
        g = json.loads(line)
        per[g["id"]][g["arm"]].append(bool(exact_match(g["model_output"], frame[g["id"]]["answer"])))
        outs[g["id"]][g["arm"]].append(g["model_output"])
    ids = list(frame)
    assert all(len(per[i][a]) == 8 for i in ids for a in ("std", "closed", "hint_first")), "grades incomplete"
    rows = []
    for i in ids:
        s, c, h = (sum(per[i][a]) for a in ("std", "closed", "hint_first"))
        rows.append({"id": i, "std_n_correct": s, "closed_n_correct": c, "hint_first_n_correct": h,
                     "doc_dependent_failing": s == 0 and c == 0, "hint_repairable": h >= 4,
                     "std_outputs": outs[i]["std"], "closed_outputs": outs[i]["closed"], "hint_outputs": outs[i]["hint_first"]})
    with open(args.out_rows, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dd = [r["id"] for r in rows if r["doc_dependent_failing"]]
    wf = [r["id"] for r in rows if r["doc_dependent_failing"] and r["hint_repairable"]]
    def modal(o): return collections.Counter(normalize_answer(x) for x in o).most_common(1)[0][0]
    reading = [i for i in wf if modal(outs[i]["std"]) != modal(outs[i]["closed"])]
    hist = collections.Counter(r["hint_first_n_correct"] for r in rows if r["doc_dependent_failing"])
    out = {"registered": "docs/causal_handle_directions.md item WF", "frame": str(args.frame), "frame_seed": 20260827, "grade_seed": 20260828,
           "n_rows": len(rows), "std_accuracy": sum(r["std_n_correct"] for r in rows) / (8 * len(rows)),
           "closed_accuracy": sum(r["closed_n_correct"] for r in rows) / (8 * len(rows)),
           "hint_first_accuracy": sum(r["hint_first_n_correct"] for r in rows) / (8 * len(rows)),
           "n_failing_zero_of_8": sum(r["std_n_correct"] == 0 for r in rows), "n_doc_dependent_failing": len(dd),
           "hint_first_hist_on_doc_dependent": {str(k): hist[k] for k in sorted(hist)},
           "n_hint_repairable_doc_dependent": len(wf), "hint_repairable_rate_of_doc_dependent": len(wf) / max(len(dd), 1),
           "n_reading_driven_among_wf": len(reading), "n_memory_driven_among_wf": len(wf) - len(reading),
           "wf_rows": wf, "n_wf_rows": len(wf), "underpowered": len(wf) < 20,
           "shards": {"0": wf[0::2], "1": wf[1::2]}, "pools": {"doc_dependent_failing": dd}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("wf_rows", "shards", "pools")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
