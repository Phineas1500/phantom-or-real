#!/usr/bin/env python3
"""Hypotheses-in-context behavioral test on the recognition-gap rowset.

Black-box test of the deployment-gap hypothesis via the Modal OpenAI-compatible
endpoint: on rows where free-form generation is strong-wrong but forced-choice
recognition picks gold, does listing candidate hypotheses in the prompt (without
forcing a choice) make free-form generation correct? The foils-only control
separates selection-from-candidates from generic priming. See
docs/causal_handle_directions.md experiment 7 context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
from scripts.stage2_steer_forced_choice_direction import (  # noqa: E402
    GENERATION_REQUEST_RE,
    build_forced_choice_prompt,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.inference import build_messages  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402

GENERATION_REQUEST = "Please come up with hypotheses to explain observations."


def make_user_prompt(row: dict[str, Any], candidates: list[str] | None) -> str:
    if candidates is None:
        return row["prompt_text"]
    base = GENERATION_REQUEST_RE.sub("", row["prompt_text"]).strip()
    listing = "\n".join(f"- {candidate}" for candidate in candidates)
    return (
        f"{base}\n\n"
        f"Candidate hypotheses to consider:\n{listing}\n\n"
        f"{GENERATION_REQUEST}"
    )


def build_conditions(
    row: dict[str, Any], *, order_seed: int
) -> dict[str, list[str] | None]:
    monitor = row["prefix_monitor"]
    gold = monitor["gold_hypothesis"]
    hard_foil = monitor["foil_hypothesis"]
    fc = build_forced_choice_prompt(row, row_index=int(row["row_index"]), option_seed=order_seed)
    polarity_foil = fc.foil_hypothesis
    rng = random.Random(order_seed + int(row["row_index"]) * 7919)
    gold_present = [gold, hard_foil]
    foils_only = [polarity_foil, hard_foil]
    rng.shuffle(gold_present)
    rng.shuffle(foils_only)
    return {
        "baseline": None,
        "candidates_gold_present": gold_present,
        "candidates_foils_only": foils_only,
    }


async def run_condition(
    client,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    row: dict[str, Any],
    condition: str,
    candidates: list[str] | None,
    samples: int,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    user = make_user_prompt(row, candidates)
    messages = build_messages(row["system_prompt"], user, model)
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
                "task": row.get("task"),
                "height": row.get("height"),
                "model": model,
                "condition": condition,
                "sample_index": sample_index,
                "method": "candidates_in_context",
                "target_variable": "selected_hypothesis",
                "representation_type": "prompt_text",
                "candidates": candidates,
                "gold_hypothesis": row["prefix_monitor"]["gold_hypothesis"],
                "hard_foil_hypothesis": row["prefix_monitor"]["foil_hypothesis"],
                "model_output": reply,
                **score,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by[row["condition"]][row["source_row_index"]].append(bool(row["is_correct_strong"]))
    base = {idx: float(np.mean(v)) for idx, v in by.get("baseline", {}).items()}
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
        if condition != "baseline" and base:
            deltas = [p[idx] - base[idx] for idx in p if idx in base]
            entry["mean_delta_p_strong_vs_baseline"] = float(np.mean(deltas))
            entry["rows_improved"] = int(sum(d > 0 for d in deltas))
            entry["rows_degraded"] = int(sum(d < 0 for d in deltas))
            entry["false_to_true_rows"] = int(
                sum(base[idx] == 0.0 and p[idx] > 0.0 for idx in p if idx in base)
            )
        summary[condition] = entry
    if "candidates_gold_present" in summary and "candidates_foils_only" in summary:
        summary["gold_present_minus_foils_only"] = (
            summary["candidates_gold_present"]["strong_accuracy"]
            - summary["candidates_foils_only"]["strong_accuracy"]
        )
    return summary


async def main_async(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    load_env()
    started = time.time()
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    if not base_url:
        raise ValueError("no --base-url and OPENAI_BASE_URL unset")

    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=0.0,
    )
    if not selected_rows:
        raise ValueError(f"no rows selected: {selection_summary}")
    print(f"selected_rows={len(selected_rows)}", flush=True)

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(f"scorer_preflight: strong={preflight['is_correct_strong']}", flush=True)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for row in selected_rows:
        conditions = build_conditions(row, order_seed=args.order_seed)
        for condition, candidates in conditions.items():
            tasks.append(
                run_condition(
                    client,
                    semaphore,
                    model=args.model,
                    row=row,
                    condition=condition,
                    candidates=candidates,
                    samples=args.samples_per_row,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            )
    results = await asyncio.gather(*tasks)
    await client.close()
    rows = [item for batch in results for item in batch]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    for condition, entry in summary.items():
        if isinstance(entry, dict):
            print(
                f"{condition}: P(strong)={entry['strong_accuracy']:.3f} "
                f"rows={entry['n_rows']} gens={entry['n_generations']}",
                flush=True,
            )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "script": "scripts/stage2_candidates_in_context.py",
        "model": args.model,
        "endpoint": "modal_openai_compatible",
        "task": "infer_property",
        "target_variable": "selected_hypothesis",
        "representation_type": "prompt_text",
        "method": "candidates_in_context",
        "jsonl": str(args.jsonl),
        "prefix_trajectory_jsonl": str(args.prefix_trajectory_jsonl),
        "selection": selection_summary,
        "generation": {
            "samples_per_row": args.samples_per_row,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "order_seed": args.order_seed,
        },
        "summary": summary,
        "n": len(rows),
        "controls": ["regenerated_baseline", "foils_only_candidates", "candidate_order_shuffled"],
        "causal_abstraction_claim": (
            "Behavioral deployment-gap test: if free-form generation becomes strong-correct when "
            "gold is among in-context candidates but not when only foils are listed, the "
            "recognition-generation gap is candidate availability at selection time, not missing "
            "knowledge. Prompt-level evidence; not an activation-level intervention."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
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
    parser.add_argument("--order-seed", type=int, default=20260430)
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/candidates_in_context_27b_property_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/candidates_in_context_27b_property_manifest.json"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
