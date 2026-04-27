#!/usr/bin/env python3
"""Train quick logistic probes on Stage 2 raw residual activations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list  # noqa: E402
from src.stage2_probes import run_probe_grid, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--keep-parse-failed", action="store_true")
    args = parser.parse_args()

    report = run_probe_grid(
        activation_dir=args.activation_dir,
        model_key=args.model_key,
        tasks=args.tasks,
        layers=args.layers,
        seed=args.seed,
        drop_parse_failed=not args.keep_parse_failed,
    )
    write_json(args.output, report)
    print(args.output)
    for task, best in report["best_by_task"].items():
        print(f"{task}: {best}")


if __name__ == "__main__":
    main()
