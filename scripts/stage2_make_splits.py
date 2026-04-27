#!/usr/bin/env python3
"""Create Stage 2 S1/S2 split assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_phase0 import (  # noqa: E402
    load_stage1_records,
    make_split_assignments,
    stage1_jsonl_paths,
    summarize_split_assignments,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, default=Path("results/full/with_errortype"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("docs/stage2_splits_summary.json"))
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()

    records = load_stage1_records(
        stage1_jsonl_paths(args.jsonl_dir),
        models=args.models,
        tasks=args.tasks,
    )
    assignments = make_split_assignments(records, seed=args.seed)
    write_jsonl(args.output, assignments)
    summary = summarize_split_assignments(assignments)
    summary["seed"] = args.seed
    summary["output"] = str(args.output)
    summary["filters"] = {
        "models": args.models or "all",
        "tasks": args.tasks or "all",
    }
    write_json(args.summary, summary)
    print(args.output)
    print(args.summary)


if __name__ == "__main__":
    main()
