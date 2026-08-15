#!/usr/bin/env python3
"""Item M frozen row frame (docs/causal_handle_directions.md, item M).

Seeded balanced factorial draw: 32 rows per (OC x SJ) cell, seed 20260816,
from the soft-census 2,000-row frame. SJ = k=4 majority under the soft
question (ties excluded); OC = stage-1 is_correct_strong. The M2 row sets are
the first 24 per OC=0 cell in seeded order. Writes docs/sjoc_frame_128.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CENSUS = Path("results/sj_census/sj_census_2k_soft.jsonl")
OUT = Path("docs/sjoc_frame_128.json")
SEED = 20260816
PER_CELL = 32
CELLS = [(True, "yes"), (True, "no"), (False, "yes"), (False, "no")]


def main() -> int:
    by_cell: dict[tuple[bool, str], list[int]] = {c: [] for c in CELLS}
    with CENSUS.open() as f:
        for line in f:
            r = json.loads(line)
            if r["sj_majority"] not in ("yes", "no"):
                continue
            by_cell[(bool(r["is_correct_strong"]), r["sj_majority"])].append(r["source_row_index"])

    rng = random.Random(SEED)
    frame = []
    m2_sets = {}
    for oc, sj in CELLS:
        pool = sorted(by_cell[(oc, sj)])
        drawn = rng.sample(pool, PER_CELL)
        for ri in drawn:
            frame.append({"row_index": ri, "oc": oc, "sj_yes": sj == "yes",
                          "cell": f"oc{int(oc)}_sj{sj}"})
        if not oc:
            m2_sets["confident_wrong" if sj == "yes" else "ordinary_wrong"] = drawn[:24]

    out = {
        "seed": SEED, "per_cell": PER_CELL, "n": len(frame),
        "census_source": str(CENSUS),
        "cell_pool_sizes": {f"oc{int(oc)}_sj{sj}": len(by_cell[(oc, sj)]) for oc, sj in CELLS},
        "rows": frame, "m2_sets": m2_sets,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
