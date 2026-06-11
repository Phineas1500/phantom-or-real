#!/usr/bin/env python3
"""Self-critique loop test on the recognition-gap rowset.

Two-pass prompting via the Modal endpoint: generate, then ask the model to
verify its own hypotheses against the theories/observations and revise. Tests
whether the internal correctness evaluation (causally epiphenomenal in the
residual stream per the erasure result) becomes deployable when routed through
tokens instead of activations.
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
from src.stage2_steering import score_reply  # noqa: E402

CRITIQUE_PROMPT = (
    "Review your answer. Check each hypothesis against the theories and "
    "observations given above: does it follow from them, and is it the best "
    "available explanation? If you find an error, correct it. Reply with only "
    "your final hypothesis or hypotheses, one per line, in the same format as "
    "before."
)


async def run_chain(
    client,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    row: dict[str, Any],
    sample_index: int,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    messages = build_messages(row["system_prompt"], row["prompt_text"], model)
    async with semaphore:
        first = await client.chat.completions.create(
            model=model, messages=messages, n=1, temperature=temperature, max_tokens=max_tokens
        )
    first_reply = (first.choices[0].message.content or "").strip()
    followup = messages + [
        {"role": "assistant", "content": first_reply},
        {"role": "user", "content": CRITIQUE_PROMPT},
    ]
    async with semaphore:
        second = await client.chat.completions.create(
            model=model, messages=followup, n=1, temperature=temperature, max_tokens=max_tokens
        )
    second_reply = (second.choices[0].message.content or "").strip()

    out = []
    for stage, reply in (("baseline", first_reply), ("self_critique", second_reply)):
        score = score_reply(row, reply)
        out.append(
            {
                "schema_version": 1,
                "source_row_index": int(row["row_index"]),
                "example_id": row.get("example_id"),
                "height": row.get("height"),
                "model": model,
                "condition": stage,
                "sample_index": sample_index,
                "method": "self_critique_two_pass",
                "target_variable": "selected_hypothesis",
                "representation_type": "prompt_text",
                "gold_hypothesis": row["prefix_monitor"]["gold_hypothesis"],
                "model_output": reply,
                **score,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by[row["condition"]][row["source_row_index"]].append(bool(row["is_correct_strong"]))
    base = {idx: float(np.mean(v)) for idx, v in by["baseline"].items()}
    summary: dict[str, Any] = {}
    for condition, per_row in sorted(by.items()):
        p = {idx: float(np.mean(v)) for idx, v in per_row.items()}
        flat = [s for v in per_row.values() for s in v]
        entry: dict[str, Any] = {
            "n_rows": len(per_row),
            "n_generations": len(flat),
            "strong_accuracy": float(np.mean(flat)),
            "per_row_p_strong": {str(idx): p[idx] for idx in sorted(p)},
        }
        if condition != "baseline":
            deltas = [p[idx] - base[idx] for idx in p if idx in base]
            entry["mean_delta_p_strong_vs_baseline"] = float(np.mean(deltas))
            entry["rows_improved"] = int(sum(d > 0 for d in deltas))
            entry["rows_degraded"] = int(sum(d < 0 for d in deltas))
            entry["false_to_true_rows"] = int(
                sum(base[idx] == 0.0 and p[idx] > 0.0 for idx in p if idx in base)
            )
        summary[condition] = entry
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

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        run_chain(
            client,
            semaphore,
            model=args.model,
            row=row,
            sample_index=sample_index,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        for row in selected_rows
        for sample_index in range(args.samples_per_row)
    ]
    results = await asyncio.gather(*tasks)
    await client.close()
    rows = [item for batch in results for item in batch]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    for condition, entry in summary.items():
        print(f"{condition}: P(strong)={entry['strong_accuracy']:.3f}", flush=True)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "script": "scripts/stage2_self_critique.py",
        "model": args.model,
        "endpoint": "modal_openai_compatible",
        "task": "infer_property",
        "target_variable": "selected_hypothesis",
        "representation_type": "prompt_text",
        "method": "self_critique_two_pass",
        "critique_prompt": CRITIQUE_PROMPT,
        "selection": selection_summary,
        "generation": {
            "samples_per_row": args.samples_per_row,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "summary": summary,
        "n": len(rows),
        "controls": ["paired_first_pass_baseline"],
        "causal_abstraction_claim": (
            "Tests whether the internal correctness evaluation, causally epiphenomenal in the "
            "residual stream under erasure, becomes behaviorally deployable when routed through "
            "the context window via a self-critique turn. Prompt-level evidence."
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
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/self_critique_27b_property_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/self_critique_27b_property_manifest.json"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
