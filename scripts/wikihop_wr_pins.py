#!/usr/bin/env python3
"""Item WR stage-1 pins: score the hint-first text screen of the W2 pool,
define HINT-REPAIRABLE rows (>= 4/8 exact-match gold), draw up to 24 with
seed 20260825 excluding the WH rows, and write docs/wikihop_wr_pinned.json.
Registered: docs/causal_handle_directions.md item WR."""
from __future__ import annotations
import argparse, collections, gzip, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import exact_match, normalize_answer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grades", type=Path, default=Path("results/loop_screen/wikihop_hint_grades.jsonl"))
    p.add_argument("--frame", type=Path, default=Path("results/loop_screen/wikihop_port_input.jsonl.gz"))
    p.add_argument("--w0-rows", type=Path, default=Path("results/loop_screen/wikihop_w0_rows.jsonl"))
    p.add_argument("--wh-pins", type=Path, default=Path("docs/wikihop_wh_pinned.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wr_pinned.json"))
    p.add_argument("--threshold", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260825)
    args = p.parse_args()
    frame = {}
    for line in gzip.open(args.frame, "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    w0 = {}
    for line in open(args.w0_rows):
        r = json.loads(line)
        w0[r["id"]] = r
    per = collections.defaultdict(list)
    for line in open(args.grades):
        g = json.loads(line)
        per[g["id"]].append(bool(exact_match(g["model_output"], frame[g["id"]]["answer"])))
    assert all(len(v) == 8 for v in per.values()), "screen incomplete"
    counts = {i: sum(v) for i, v in per.items()}
    wh = json.load(open(args.wh_pins))
    wh_rows = set(wh["wh_rows"])
    reading = set(wh["reading_driven_rows"])
    def modal(outs):
        return collections.Counter(normalize_answer(o) for o in outs).most_common(1)[0][0]
    repairable = sorted(i for i, c in counts.items() if c >= args.threshold)
    rate = len(repairable) / len(counts)
    by_kind = {"reading_driven": sum(i in reading for i in repairable), "memory_driven": sum(i not in reading for i in repairable)}
    pool = [i for i in repairable if i not in wh_rows]
    draw = sorted(random.Random(args.seed).sample(pool, min(args.max_rows, len(pool))))
    wh_recheck = {i: counts[i] for i in sorted(wh_rows) if i in counts}
    hist = collections.Counter(counts.values())
    out = {"registered": "docs/causal_handle_directions.md item WR", "screen_seed": 20260824, "draw_seed": args.seed,
           "threshold_of_8": args.threshold, "n_pool_screened": len(counts), "hint_first_correct_histogram": {str(k): hist[k] for k in sorted(hist)},
           "hint_first_mean_rate": sum(counts.values()) / (8 * len(counts)),
           "n_hint_repairable": len(repairable), "hint_repairable_rate": rate, "hint_repairable_by_kind": by_kind,
           "n_reading_driven_in_pool": sum(i in reading for i in counts), "n_memory_driven_in_pool": sum(i not in reading for i in counts),
           "wh_rows_recheck_hint_first_correct_of_8": wh_recheck,
           "hint_repairable_rows": repairable, "wr_rows": draw, "n_wr_rows": len(draw),
           "underpowered": len(draw) < 12}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("hint_repairable_rows",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
