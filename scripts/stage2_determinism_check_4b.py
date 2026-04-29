#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import (  # noqa: E402
    encode_stage1_rows,
    extract_residual_activations,
    load_tl_model,
    parse_int_list,
    read_stage1_rows,
)
from src.env_loader import load_env  # noqa: E402


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--load-mode", choices=["no-processing", "default"], default="no-processing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_stage1_rows(args.jsonl, height=args.height, limit=args.limit)
    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        load_mode=args.load_mode,
    )
    examples = encode_stage1_rows(rows, tokenizer=model.tokenizer, model_name=args.model)

    acts_a, sidecar_a, stats_a = extract_residual_activations(
        model,
        examples,
        layers=args.layers,
        batch_size=args.batch_size,
    )
    acts_b, sidecar_b, stats_b = extract_residual_activations(
        model,
        examples,
        layers=args.layers,
        batch_size=args.batch_size,
    )

    layer_results = {}
    ok = sidecar_a == sidecar_b
    for layer in args.layers:
        equal = torch.equal(acts_a[layer], acts_b[layer])
        layer_results[str(layer)] = {
            "torch_equal": equal,
            "shape": list(acts_a[layer].shape),
            "dtype": str(acts_a[layer].dtype),
        }
        ok = ok and equal

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if ok else "failed",
        "inputs": {
            "jsonl": str(args.jsonl),
            "model": args.model,
            "layers": args.layers,
            "limit": args.limit,
            "height": args.height,
            "batch_size": args.batch_size,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "load_mode": args.load_mode,
        },
        "sidecars_equal": sidecar_a == sidecar_b,
        "layers": layer_results,
        "stats_a": stats_a,
        "stats_b": stats_b,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(args.output)
    print(report["status"])
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()