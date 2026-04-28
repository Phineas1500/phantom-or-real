#!/usr/bin/env python3
"""Train Stage 2 metadata-only B0 baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_phase0 import (  # noqa: E402
    add_prompt_length_fallback,
    add_prompt_token_counts,
    attach_splits,
    load_stage1_records,
    read_split_assignments,
    stage1_jsonl_paths,
    summarize_b0_results,
    train_metadata_baselines,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, default=Path("results/full/with_errortype"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2/baselines/b0_metadata.json"))
    parser.add_argument("--summary", type=Path, default=Path("docs/stage2_b0_summary.json"))
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--length-mode", choices=["tokenizer", "whitespace"], default="tokenizer")
    parser.add_argument("--split-families", nargs="+", choices=["s1", "s2", "s3"], default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()

    records = load_stage1_records(
        stage1_jsonl_paths(args.jsonl_dir),
        models=args.models,
        tasks=args.tasks,
    )
    attach_splits(records, read_split_assignments(args.splits))
    if args.length_mode == "tokenizer":
        add_prompt_token_counts(records, hf_cache=args.hf_cache)
    else:
        add_prompt_length_fallback(records)

    results = train_metadata_baselines(records, split_families=args.split_families)
    results["prompt_length_mode"] = records[0].get("prompt_length_mode") if records else None
    results["filters"] = {
        "models": args.models or "all",
        "tasks": args.tasks or "all",
        "split_families": args.split_families or "all",
    }
    write_json(args.output, results)
    summary = summarize_b0_results(results)
    summary["source"] = str(args.output)
    summary["filters"] = results["filters"]
    summary["split_families"] = results["split_families"]
    summary["prompt_length_mode"] = results["prompt_length_mode"]
    write_json(args.summary, summary)
    print(args.output)
    print(args.summary)


if __name__ == "__main__":
    main()
