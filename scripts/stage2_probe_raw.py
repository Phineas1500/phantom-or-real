#!/usr/bin/env python3
"""Train Stage 2 raw-activation probes (logistic or diffmeans)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list  # noqa: E402
from src.stage2_probes import run_probe_grid, write_json  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--keep-parse-failed", action="store_true")
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--probe-type", choices=("logistic", "diffmeans"), default="logistic")
    parser.add_argument("--save-probes-dir", type=Path, default=None)
    parser.add_argument("--c-values", type=parse_float_list, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    report = run_probe_grid(
        activation_dir=args.activation_dir,
        activation_site=args.activation_site,
        model_key=args.model_key,
        tasks=args.tasks,
        layers=args.layers,
        seed=args.seed,
        drop_parse_failed=not args.keep_parse_failed,
        splits_path=args.splits,
        split_family=args.split_family,
        shuffle_labels=args.shuffle_labels,
        c_values=args.c_values,
        max_iter=args.max_iter,
        bootstrap_samples=args.bootstrap_samples,
        probe_type=args.probe_type,
        save_probes_dir=args.save_probes_dir,
    )
    write_json(args.output, report)
    print(args.output)
    for task, best in report["best_by_task"].items():
        print(f"{task}: {best}")


if __name__ == "__main__":
    main()
