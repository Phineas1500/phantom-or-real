#!/usr/bin/env python3
"""Item WQ pins (the WikiHop loop on a second model). From the second
model's stage-1 grades on the WF real-text frame (std / closed /
hint-first, k=8 each): DOC-DEPENDENT failing (0/8 std ∧ 0/8 closed),
HINT-REPAIRABLE (≥ 4/8 hint-first), the pool counts the WD design would
use, and the WX cross-fit jobs A/B over the hint-repairable doc-dependent
rows (seeded shards; job A tests shard 1 with shard 0 as donors, B the
reverse)."""
from __future__ import annotations
import argparse, collections, gzip, json, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import exact_match, normalize_answer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grades", type=Path, default=Path("results/loop_screen/wikihop_wq_grades.jsonl"))
    p.add_argument("--frame", type=Path, default=Path("results/loop_screen/wikihop_fresh_input.jsonl.gz"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wq_pinned.json"))
    p.add_argument("--out-rows", type=Path, default=Path("results/loop_screen/wikihop_wq_rows.jsonl"))
    p.add_argument("--max-rows", type=int, default=60)
    p.add_argument("--draw-seed", type=int, default=20260855)
    p.add_argument("--shard-seed", type=int, default=20260856)
    p.add_argument("--label", default="item WQ")
    args = p.parse_args()
    frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.frame, "rt")}
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
    rep_all = [r["id"] for r in rows if r["doc_dependent_failing"] and r["hint_repairable"]]
    rep = sorted(random.Random(args.draw_seed).sample(rep_all, args.max_rows)) if 0 < args.max_rows < len(rep_all) else sorted(rep_all)
    shuf = list(rep); random.Random(args.shard_seed).shuffle(shuf)
    shard0, shard1 = sorted(shuf[0::2]), sorted(shuf[1::2])
    def modal(o): return collections.Counter(normalize_answer(x) for x in o).most_common(1)[0][0]
    reading = [i for i in rep if modal(outs[i]["std"]) != modal(outs[i]["closed"])]
    hist = collections.Counter(r["hint_first_n_correct"] for r in rows if r["doc_dependent_failing"])
    correct = [r["id"] for r in rows if r["std_n_correct"] >= 5]
    out = {"registered": f"docs/causal_handle_directions.md {args.label}", "frame": str(args.frame), "grades": str(args.grades),
           "n_rows": len(rows), "std_accuracy": sum(r["std_n_correct"] for r in rows) / (8 * len(rows)),
           "closed_accuracy": sum(r["closed_n_correct"] for r in rows) / (8 * len(rows)),
           "hint_first_accuracy": sum(r["hint_first_n_correct"] for r in rows) / (8 * len(rows)),
           "n_failing_zero_of_8": sum(r["std_n_correct"] == 0 for r in rows), "n_doc_dependent_failing": len(dd),
           "hint_first_hist_on_doc_dependent": {str(k): hist[k] for k in sorted(hist)},
           "n_hint_repairable_doc_dependent_all": len(rep_all), "hint_repairable_rate_of_doc_dependent": len(rep_all) / max(len(dd), 1),
           "max_rows": args.max_rows, "draw_seed": args.draw_seed, "shard_seed": args.shard_seed,
           "n_test_total": len(rep), "n_reading_driven": len(reading), "n_memory_driven": len(rep) - len(reading),
           "underpowered": len(rep) < 20, "n_correct_majority": len(correct),
           "rule": "cross-fit on the hint-repairable doc-dependent rows: job A tests shard 1 with shard 0 as donors; job B tests shard 0 with shard 1 as donors; frozen vector = mean per-position gold hint-delta over the donor rows, norm target = donor mean per-position |delta|",
           "jobs": {"A": {"test_rows": shard1, "donor_rows": shard0}, "B": {"test_rows": shard0, "donor_rows": shard1}},
           "pools": {"doc_dependent_failing": dd, "correct_majority": correct}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_rows", "std_accuracy", "closed_accuracy", "hint_first_accuracy", "n_failing_zero_of_8", "n_doc_dependent_failing",
                                          "n_hint_repairable_doc_dependent_all", "hint_repairable_rate_of_doc_dependent", "n_test_total", "n_reading_driven", "underpowered", "n_correct_majority")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
