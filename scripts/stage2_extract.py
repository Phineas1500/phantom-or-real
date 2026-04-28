#!/usr/bin/env python3
"""Extract Stage 2 residual-stream activations from shipped Stage 1 JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list, run_extraction  # noqa: E402
from src.env_loader import load_env  # noqa: E402


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HF model name, e.g. google/gemma-3-27b-it")
    parser.add_argument("--model-key", default=None, help="Output slug, e.g. gemma3_27b")
    parser.add_argument("--task", default=None, help="Override task slug; inferred from rows by default")
    parser.add_argument("--layers", required=True, type=parse_int_list, help="Comma-separated layers, e.g. 15,30,45")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--hook-template", default="blocks.{layer}.hook_resid_post")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--drop-parse-failed", action="store_true")
    parser.add_argument("--load-mode", choices=["no-processing", "default"], default="no-processing")
    args = parser.parse_args()

    written = run_extraction(
        jsonl_path=args.jsonl,
        model_name=args.model,
        model_key=args.model_key,
        task=args.task,
        layers=args.layers,
        batch_size=args.batch_size,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        out_dir=args.out_dir,
        activation_site=args.activation_site,
        hook_template=args.hook_template,
        height=args.height,
        limit=args.limit,
        skip=args.skip,
        drop_parse_failed=args.drop_parse_failed,
        load_mode=args.load_mode,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
