"""Walk results/full/with_errortype/*.jsonl and emit docs/stage2_inventory.json."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOW_COUNT_THRESHOLD = 100


def load_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_inventory(jsonl_paths: list[Path]) -> dict:
    # raw cell counts: (model, task, height, parse_failed, is_correct_strong) -> int
    raw_cells: dict[tuple, int] = defaultdict(int)
    # per-(model, task, height) aggregate after parse_failed=False filter
    agg: dict[tuple, dict] = defaultdict(lambda: {"total": 0, "positive": 0, "negative": 0, "parse_failed": 0})

    for path in jsonl_paths:
        rows = load_rows(path)
        for row in rows:
            model = row.get("model", "unknown")
            task = row.get("task", "unknown")
            height = row.get("height")
            parse_failed = bool(row.get("parse_failed", False))
            is_correct = row.get("is_correct_strong")

            raw_cells[(model, task, height, parse_failed, is_correct)] += 1

            key = (model, task, height)
            agg[key]["total"] += 1
            if parse_failed:
                agg[key]["parse_failed"] += 1
            else:
                if is_correct:
                    agg[key]["positive"] += 1
                else:
                    agg[key]["negative"] += 1

    # Build per-cell list
    cell_list = []
    for (model, task, height, parse_failed, is_correct), count in sorted(raw_cells.items()):
        cell_list.append({
            "model": model,
            "task": task,
            "height": height,
            "parse_failed": parse_failed,
            "is_correct_strong": is_correct,
            "count": count,
        })

    # Build per-(model, task, height) summary with flags
    summary_list = []
    low_count_cells = []
    for (model, task, height), counts in sorted(agg.items()):
        pos = counts["positive"]
        neg = counts["negative"]
        entry = {
            "model": model,
            "task": task,
            "height": height,
            "total": counts["total"],
            "positive": pos,
            "negative": neg,
            "parse_failed": counts["parse_failed"],
            "below_threshold_positive": pos < LOW_COUNT_THRESHOLD,
            "below_threshold_negative": neg < LOW_COUNT_THRESHOLD,
        }
        summary_list.append(entry)
        if pos < LOW_COUNT_THRESHOLD or neg < LOW_COUNT_THRESHOLD:
            low_count_cells.append({"model": model, "task": task, "height": height,
                                    "positive": pos, "negative": neg})

    return {
        "low_count_threshold": LOW_COUNT_THRESHOLD,
        "low_count_cells": low_count_cells,
        "per_model_task_height": summary_list,
        "raw_cells": cell_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=ROOT / "results" / "full" / "with_errortype",
        help="Directory containing the *_infer_*.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "stage2_inventory.json",
        help="Where to write the inventory JSON",
    )
    args = parser.parse_args()

    jsonl_paths = sorted(args.jsonl_dir.glob("*.jsonl"))
    if not jsonl_paths:
        sys.exit(f"No JSONL files found under {args.jsonl_dir}")

    print(f"Found {len(jsonl_paths)} JSONL file(s):")
    for p in jsonl_paths:
        print(f"  {p.name}")

    inventory = build_inventory(jsonl_paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"\nWrote {args.output}")

    if inventory["low_count_cells"]:
        print("\nWARNING — cells below 100-row threshold:")
        for c in inventory["low_count_cells"]:
            print(f"  {c['model']} / {c['task']} / h={c['height']}: "
                  f"pos={c['positive']}, neg={c['negative']}")


if __name__ == "__main__":
    main()