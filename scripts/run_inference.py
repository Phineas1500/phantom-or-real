"""Batch orchestrator: run inference across heights for a (model, task) and concat JSONL.

Example — pilot, Gemma 3 4B via local vLLM:
    python scripts/run_inference.py \
        --examples-dir data/pilot \
        --tasks property ontology \
        --heights 1 2 3 4 \
        --model google/gemma-3-4b-it \
        --base-url http://localhost:8000/v1 \
        --concurrency 16 \
        --model-slug gemma3_4b \
        --output-dir results/pilot

Example — full, Gemma 3 27B via API:
    export OPENAI_API_KEY=...
    python scripts/run_inference.py \
        --examples-dir data/full \
        --tasks property ontology \
        --heights 1 2 3 4 \
        --model google/gemma-3-27b-it \
        --base-url https://api.openrouter.ai/v1 \
        --concurrency 4 \
        --model-slug gemma3_27b \
        --output-dir results/full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import HEIGHTS, MODAL_ENDPOINTS, TASK_CODES  # noqa: E402
from src.env_loader import get_openai_inference_credentials, load_env  # noqa: E402
from src.export import write_jsonl  # noqa: E402
from src.inference import load_examples, run_inference  # noqa: E402

load_env()


TASK_SLUG = {"property": "infer_property", "ontology": "infer_subtype"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASK_CODES), choices=list(TASK_CODES))
    p.add_argument("--heights", nargs="+", type=int, default=list(HEIGHTS))
    p.add_argument("--model", required=True)
    _default_base_url, _default_key = get_openai_inference_credentials()
    p.add_argument("--base-url", default=None, help="Override endpoint. If absent, use MODAL_ENDPOINTS[model] or .env OPENAI_BASE_URL.")
    p.add_argument("--api-key", default=_default_key)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--model-slug", required=True, help="Short slug for output filenames, e.g. gemma3_4b")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Cap examples per (task, height)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or MODAL_ENDPOINTS.get(args.model) or _default_base_url
    if base_url is None:
        raise SystemExit(
            "No base URL: pass --base-url, add model to MODAL_ENDPOINTS, "
            "or set OPENAI_BASE_URL."
        )

    for task in args.tasks:
        # Accumulate rows for this task across all heights -> single per-(model, task) JSONL
        all_rows = []
        per_height_summary: dict[str, dict] = {}
        for h in args.heights:
            pkl = args.examples_dir / f"examples_{task}_h{h}.pkl"
            if not pkl.exists():
                raise FileNotFoundError(pkl)
            examples, task_type, height = load_examples(pkl)
            assert task_type == task and height == h
            if args.limit is not None:
                examples = examples[: args.limit]

            start = time.monotonic()
            rows = asyncio.run(
                run_inference(
                    examples,
                    task_type=task,
                    height=h,
                    model_name=args.model,
                    base_url=base_url,
                    api_key=args.api_key,
                    concurrency=args.concurrency,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_attempts=args.max_attempts,
                    example_id_prefix=f"{task}_h{h}",
                )
            )
            elapsed = time.monotonic() - start

            n = len(rows)
            strong = sum(r["is_correct_strong"] for r in rows)
            parse_fail = sum(r["parse_failed"] for r in rows)
            summary = {
                "n": n,
                "strong_accuracy": strong / n if n else None,
                "parse_fail_rate": parse_fail / n if n else None,
                "elapsed_seconds": elapsed,
                "throughput_ips": n / elapsed if elapsed > 0 else None,
            }
            per_height_summary[f"h{h}"] = summary
            print(f"[{args.model_slug}/{task}/h{h}] {json.dumps(summary)}")

            all_rows.extend(rows)

        out_name = f"{args.model_slug}_{TASK_SLUG[task]}.jsonl"
        out_path = args.output_dir / out_name
        write_jsonl(all_rows, out_path)
        print(f"wrote {len(all_rows)} rows -> {out_path}")

        # Side-by-side per-height summary
        meta = {
            "model": args.model,
            "model_slug": args.model_slug,
            "task": task,
            "n_rows": len(all_rows),
            "per_height": per_height_summary,
        }
        with (args.output_dir / f"{args.model_slug}_{TASK_SLUG[task]}_runmeta.json").open("w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
