#!/usr/bin/env python3
"""Build Stage 2 inventory counts from shipped Stage 1 JSONLs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_phase0 import build_inventory, stage1_jsonl_paths, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, default=Path("results/full/with_errortype"))
    parser.add_argument("--output", type=Path, default=Path("docs/stage2_inventory.json"))
    parser.add_argument("--low-class-threshold", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()

    inventory = build_inventory(
        stage1_jsonl_paths(args.jsonl_dir),
        low_class_threshold=args.low_class_threshold,
        models=args.models,
        tasks=args.tasks,
    )
    write_json(args.output, inventory)
    print(args.output)


if __name__ == "__main__":
    main()
