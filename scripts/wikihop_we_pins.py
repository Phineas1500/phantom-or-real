#!/usr/bin/env python3
"""Item WE pins: the WD design on REAL text. Frame = the WF fresh frame
(800 real rows, results/loop_screen/wikihop_fresh_input.jsonl.gz; grades
in wikihop_wf_rows.jsonl). Strata: hint-repairable doc-dependent rows =
the 59 WF rows (their WX cross-fit frozen-write branches are reused);
unrepairable doc-dependent rows = doc-dependent minus those 59 (seeded
draw of 100, two shards); collateral = seeded 60 correct-majority rows
(std >= 5/8). Donors for the frozen direction = the 59 WF rows (disjoint
from every test row here)."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, default=Path("results/loop_screen/wikihop_wf_rows.jsonl"))
    p.add_argument("--wf-pins", type=Path, default=Path("docs/wikihop_wf_pinned.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_we_pinned.json"))
    p.add_argument("--n-yield", type=int, default=100)
    p.add_argument("--n-collateral", type=int, default=60)
    p.add_argument("--yield-seed", type=int, default=20260850)
    p.add_argument("--collateral-seed", type=int, default=20260851)
    p.add_argument("--job-seed", type=int, default=20260852)
    args = p.parse_args()
    rows = [json.loads(l) for l in open(args.rows)]
    wf = list(json.load(open(args.wf_pins))["wf_rows"])
    dd = [r["id"] for r in rows if r["doc_dependent_failing"]]
    unrep = sorted(i for i in dd if i not in set(wf))
    correct = sorted(r["id"] for r in rows if r["std_n_correct"] >= 5)
    y = sorted(random.Random(args.yield_seed).sample(unrep, args.n_yield))
    c = sorted(random.Random(args.collateral_seed).sample(correct, args.n_collateral))
    half = len(y) // 2
    donors = sorted(wf)
    out = {"registered": "docs/causal_handle_directions.md item WE (2026-09-03)", "frame": "results/loop_screen/wikihop_fresh_input.jsonl.gz",
           "pool": {"n_rows": len(rows), "n_doc_dependent": len(dd), "n_hint_repairable": len(wf), "n_unrepairable": len(unrep), "n_correct_majority": len(correct),
                    "weight_repairable": len(wf) / len(dd), "weight_unrepairable": len(unrep) / len(dd)},
           "jobs": {"Y1": {"test_rows": y[:half], "donor_rows": donors}, "Y2": {"test_rows": y[half:], "donor_rows": donors}, "C": {"test_rows": c, "donor_rows": donors}},
           "yield_rows": y, "collateral_rows": c, "repairable_rows_from_WX": donors,
           "seeds": {"yield_draw": args.yield_seed, "collateral_draw": args.collateral_seed, "job": args.job_seed}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["pool"]), "| Y1", len(out["jobs"]["Y1"]["test_rows"]), "Y2", len(out["jobs"]["Y2"]["test_rows"]), "C", len(out["jobs"]["C"]["test_rows"]), "donors", len(donors))
    assert not (set(y) | set(c)) & set(donors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
