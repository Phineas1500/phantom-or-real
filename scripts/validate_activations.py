#!/usr/bin/env python3
"""Validate Stage 2 activation inputs and optional written artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list, slugify_model_name  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.stage2_validation import build_validation_report, write_json  # noqa: E402


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HF model name, e.g. google/gemma-3-27b-it")
    parser.add_argument("--model-key", default=None, help="Activation artifact slug, e.g. gemma3_27b")
    parser.add_argument("--task", default=None, help="Override task slug; inferred from rows by default")
    parser.add_argument("--invariants", type=Path, default=Path("docs/stage2_invariants.json"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2/equivalence_report.json"))
    parser.add_argument("--hf-cache", type=Path, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    group.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--drop-parse-failed", action="store_true")
    parser.add_argument("--layers", type=parse_int_list, default=[], help="Comma-separated layers to validate")
    parser.add_argument("--activation-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails")
    args = parser.parse_args()

    model_key = args.model_key or slugify_model_name(args.model)
    report = build_validation_report(
        jsonl_path=args.jsonl,
        model_name=args.model,
        model_key=model_key,
        invariants_path=args.invariants,
        hf_cache=args.hf_cache,
        local_files_only=args.local_files_only,
        n_ctx=args.n_ctx,
        height=args.height,
        limit=args.limit,
        skip=args.skip,
        drop_parse_failed=args.drop_parse_failed,
        task=args.task,
        layers=args.layers,
        activation_dir=args.activation_dir,
    )
    write_json(args.output, report)
    print(args.output)
    print(report["status"])
    if args.strict and report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
