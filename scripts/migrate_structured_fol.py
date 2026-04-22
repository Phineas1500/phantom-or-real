"""Rewrite the ontology_fol_structured field to the plan's schema in place.

The first Phase 3 run used a nested {theories, observations, hypothesis}
shape; the plan's schema (BEHAVIORAL_DATA_PLAN.md §2.2) wants a flat KB
{membership, inheritance, properties, negated_properties, hypothesis}.
This script reads the existing JSONL (which already contains ontology_raw
so we have everything we need), swaps the field, and writes it back.

Usage:
    python scripts/migrate_structured_fol.py --in results/full/with_errortype --dry-run
    python scripts/migrate_structured_fol.py --in results/full/with_errortype
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import structured_fol  # noqa: E402


def migrate_file(path: Path, dry_run: bool) -> dict[str, int]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    changed = 0
    already_new = 0
    for r in rows:
        raw = r["ontology_raw"]
        new_struct = structured_fol(raw["theories"], raw["observations"], raw["hypotheses"])
        current = r.get("ontology_fol_structured")
        if isinstance(current, dict) and "membership" in current and "hypothesis" in current:
            already_new += 1
        r["ontology_fol_structured"] = new_struct
        changed += 1

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")
        tmp.replace(path)

    return {"path": str(path), "n_rows": len(rows), "rewrote": changed, "already_new_schema": already_new}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", type=Path, required=True)
    p.add_argument("--pattern", default="gemma3_*.jsonl")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = sorted(args.in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no JSONLs matching {args.pattern} in {args.in_dir}")

    summary = [migrate_file(f, args.dry_run) for f in files]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
