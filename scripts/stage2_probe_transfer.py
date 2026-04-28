#!/usr/bin/env python3
"""Train cross-task transfer probes on Stage 2 raw residual activations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list  # noqa: E402
from src.stage2_probes import run_cross_task_transfer_grid, write_json  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--keep-parse-failed", action="store_true")
    parser.add_argument("--c-values", type=parse_float_list, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    report = run_cross_task_transfer_grid(
        activation_dir=args.activation_dir,
        activation_site=args.activation_site,
        model_key=args.model_key,
        tasks=args.tasks,
        layers=args.layers,
        splits_path=args.splits,
        seed=args.seed,
        split_family=args.split_family,
        drop_parse_failed=not args.keep_parse_failed,
        c_values=args.c_values,
        max_iter=args.max_iter,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_json(args.output, report)
    print(args.output)
    for transfer_key, best in report["best_by_transfer"].items():
        print(f"{transfer_key}: {best}")


if __name__ == "__main__":
    main()
