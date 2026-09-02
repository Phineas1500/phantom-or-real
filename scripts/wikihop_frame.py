#!/usr/bin/env python3
"""WikiHop frame builder (item W family). Reconstructs the W0 construction from
the cached HF `qangaroo`/wikihop validation split: id WH_dev_<index>, docs =
supports joined by blank lines, relation = query's first token with '_'→' ',
subject = the rest; candidates/answer verbatim. Filters docs <= 14,000 chars,
excludes ids already used (given frames), draws n rows with a seed.
W0 frame: seed 20260821 (verified byte-identical on WH_dev_755). WF frame: seed 20260827."""
from __future__ import annotations
import argparse, gzip, json, random
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--max-doc-chars", type=int, default=14000)
    p.add_argument("--exclude", type=Path, nargs="*", default=[])
    p.add_argument("--verify-against", type=Path, default=None, help="frame to reproduce row-for-row (id set + fields)")
    args = p.parse_args()
    from datasets import load_dataset
    ds = load_dataset("qangaroo", "wikihop", split="validation")
    used = set()
    for f in args.exclude:
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if r.get("ds", "wikihop") == "wikihop":
                used.add(r["id"])

    def build(i):
        ex = ds[i]
        rel, subj = ex["query"].split(" ", 1)
        return {"id": f"WH_dev_{i}", "ds": "wikihop", "docs": "\n\n".join(ex["supports"]),
                "relation": rel.replace("_", " "), "subject": subj, "query": ex["query"],
                "candidates": list(ex["candidates"]), "answer": ex["answer"]}

    if args.verify_against is not None:
        ref = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.verify_against, "rt")}
        bad = 0
        for rid, r in ref.items():
            b = build(int(rid.split("_")[-1]))
            for k in ("docs", "relation", "subject", "candidates", "answer"):
                if b[k] != r[k]:
                    bad += 1
                    break
        print(f"verify: {len(ref)} rows, {bad} mismatches")
        return 0 if bad == 0 else 1
    eligible = [i for i in range(len(ds)) if f"WH_dev_{i}" not in used
                and len("\n\n".join(ds[i]["supports"])) <= args.max_doc_chars]
    draw = sorted(random.Random(args.seed).sample(eligible, args.n))
    with gzip.open(args.out, "wt") as f:
        for i in draw:
            f.write(json.dumps(build(i), ensure_ascii=False) + "\n")
    print(f"eligible {len(eligible)} → drew {len(draw)} rows (seed {args.seed}) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
