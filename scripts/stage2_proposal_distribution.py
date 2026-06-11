#!/usr/bin/env python3
"""Best-of-n proposal-distribution test on the recognition-gap rowset.

Samples n free-form answers per row from the Modal endpoint and asks whether
gold ever appears in the model's own proposal distribution. Decomposes the
deployment gap: gold-proposed-but-loses is a ranking failure (selection could
fix it); gold-never-proposed is a generation failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.inference import build_messages  # noqa: E402
from src.stage2_steering import parse_float_list, score_reply  # noqa: E402


async def sample_row(
    client,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    row: dict[str, Any],
    temperature: float,
    samples: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    messages = build_messages(row["system_prompt"], row["prompt_text"], model)
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            n=samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    out = []
    for sample_index, choice in enumerate(response.choices):
        reply = (choice.message.content or "").strip()
        score = score_reply(row, reply)
        out.append(
            {
                "schema_version": 1,
                "source_row_index": int(row["row_index"]),
                "example_id": row.get("example_id"),
                "height": row.get("height"),
                "model": model,
                "temperature": temperature,
                "sample_index": sample_index,
                "method": "proposal_distribution_best_of_n",
                "target_variable": "selected_hypothesis",
                "representation_type": "prompt_text",
                "gold_hypothesis": row["prefix_monitor"]["gold_hypothesis"],
                "model_output": reply,
                **score,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    by_temp: dict[float, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_temp[row["temperature"]][row["source_row_index"]].append(row)
    summary = {}
    for temperature, per_row in sorted(by_temp.items()):
        strong = {idx: [bool(r["is_correct_strong"]) for r in v] for idx, v in per_row.items()}
        weak = {idx: [bool(r["is_correct_weak"]) for r in v] for idx, v in per_row.items()}
        summary[f"temp_{temperature:g}"] = {
            "n_rows": len(per_row),
            "samples_per_row": samples,
            "mean_p_strong": float(np.mean([s for v in strong.values() for s in v])),
            "rows_gold_proposed_strong": int(sum(any(v) for v in strong.values())),
            "rows_gold_proposed_weak": int(sum(any(v) for v in weak.values())),
            "per_row_strong_hits": {str(idx): int(sum(v)) for idx, v in sorted(strong.items())},
        }
    return summary


async def main_async(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    load_env()
    started = time.time()
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=0.0,
    )
    print(f"selected_rows={len(selected_rows)}", flush=True)
    ensure_on_path()
    temps = parse_float_list(args.temperatures)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        sample_row(
            client,
            semaphore,
            model=args.model,
            row=row,
            temperature=temperature,
            samples=args.samples_per_row,
            max_tokens=args.max_tokens,
        )
        for row in selected_rows
        for temperature in temps
    ]
    results = await asyncio.gather(*tasks)
    await client.close()
    rows = [item for batch in results for item in batch]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, args.samples_per_row)
    for key, entry in summary.items():
        print(
            f"{key}: mean_P(strong)={entry['mean_p_strong']:.3f} "
            f"rows_with_gold_in_{args.samples_per_row}_samples="
            f"{entry['rows_gold_proposed_strong']}/{entry['n_rows']}",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "script": "scripts/stage2_proposal_distribution.py",
        "model": args.model,
        "endpoint": "modal_openai_compatible",
        "task": "infer_property",
        "target_variable": "selected_hypothesis",
        "representation_type": "prompt_text",
        "method": "proposal_distribution_best_of_n",
        "selection": selection_summary,
        "generation": {
            "samples_per_row": args.samples_per_row,
            "temperatures": list(temps),
            "max_tokens": args.max_tokens,
        },
        "summary": summary,
        "n": len(rows),
        "causal_abstraction_claim": (
            "Decomposes the recognition-generation gap: rows where gold appears among n free-form "
            "samples are ranking failures (selection-fixable); rows where gold never appears are "
            "proposal failures. Prompt-level evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--prefix-trajectory-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--prefix-checkpoint", default="0")
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--model", default="gemma3-27b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--samples-per-row", type=int, default=16)
    parser.add_argument("--temperatures", default="0.7,1.0")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/proposal_distribution_27b_property_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/proposal_distribution_27b_property_manifest.json"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
