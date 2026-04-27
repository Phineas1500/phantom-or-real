"""Produce results/stage2/splits.jsonl + splits.meta.json.

One row per (example_id, task) with S1 and S2 partition assignments.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.splits import canonical_skeleton_key, make_s1_split, make_s2_split  # noqa: E402

DEFAULT_SEED = 42
JSONL_DIR = ROOT / "results" / "full" / "with_errortype"
OUT_DIR = ROOT / "results" / "stage2"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-dir", type=Path, default=JSONL_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--model-slug",
        default="gemma3_4b",
        help="Model slug prefix used to filter JSONL files (default: gemma3_4b)",
    )
    args = parser.parse_args()

    # Load all rows for the chosen model slug (both tasks).
    jsonl_paths = sorted(args.jsonl_dir.glob(f"{args.model_slug}_*.jsonl"))
    if not jsonl_paths:
        sys.exit(f"No JSONL files found for slug '{args.model_slug}' under {args.jsonl_dir}")

    print(f"Loading {len(jsonl_paths)} JSONL file(s):")
    all_rows: list[dict] = []
    for p in jsonl_paths:
        rows = load_jsonl(p)
        print(f"  {p.name}: {len(rows)} rows")
        all_rows.extend(rows)

    print(f"Total rows: {len(all_rows)}")

    # Compute skeleton keys once.
    for row in all_rows:
        row["_skeleton_key"] = canonical_skeleton_key(row)

    # S1 — per-task split (each task has its own train/val/test).
    # S2 — per-task split.
    s1_all: dict[str, str] = {}
    s2_all: dict[str, str] = {}
    s2_meta_by_task: dict[str, dict] = {}

    tasks = sorted({r["task"] for r in all_rows})
    for task in tasks:
        task_rows = [r for r in all_rows if r["task"] == task]
        s1 = make_s1_split(task_rows, seed=args.seed)
        s2, s2_meta = make_s2_split(task_rows, seed=args.seed)
        s1_all.update({eid: part for eid, part in s1.items()})
        s2_all.update({eid: part for eid, part in s2.items()})
        s2_meta_by_task[task] = s2_meta

    # Write splits.jsonl
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.out_dir / "splits.jsonl"
    with splits_path.open("w") as f:
        for row in all_rows:
            eid = row["example_id"]
            out = {
                "example_id": eid,
                "task": row["task"],
                "height": row["height"],
                "parse_failed": bool(row.get("parse_failed", False)),
                "is_correct_strong": row.get("is_correct_strong"),
                "skeleton_key": row["_skeleton_key"],
                "s1": s1_all.get(eid, "excluded"),
                "s2": s2_all.get(eid, "excluded"),
            }
            f.write(json.dumps(out) + "\n")
    print(f"Wrote {splits_path}")

    # Write splits.meta.json
    meta = {
        "seed": args.seed,
        "model_slug": args.model_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "s1_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "s2_method": "StratifiedGroupKFold(n_splits=7)",
        "s2_per_task": s2_meta_by_task,
        "s1_counts": _partition_counts(s1_all),
        "s2_counts": _partition_counts(s2_all),
    }
    meta_path = args.out_dir / "splits.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta_path}")


def _partition_counts(assignment: dict[str, str]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(assignment.values()))


if __name__ == "__main__":
    main()