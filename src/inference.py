"""Concurrent async inference driver (Phase 2.3).

Reads pickled examples, calls an OpenAI-compatible chat completions endpoint
with a bounded concurrency semaphore and exponential-backoff retries, scores
each response, computes structural annotations, and streams rows into JSONL.

Any endpoint supporting the OpenAI chat completions schema works: local vLLM,
Together.ai, Fireworks, OpenRouter, Modal. Configure via env vars or CLI:

    OPENAI_BASE_URL     -> base URL, default None (= OpenAI)
    OPENAI_API_KEY      -> API key, default "not-needed" for vLLM

Run example:
    python -m src.inference \
        --examples-pkl data/full/examples_property_h2.pkl \
        --model google/gemma-3-4b-it \
        --base-url http://localhost:8000/v1 \
        --concurrency 8 \
        --output results/gemma3_4b_infer_property_h2.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm_asyncio

from .annotations import compute_structural_annotations
from .bd_path import ensure_on_path
from .config import SYSTEM_PROMPT, make_user_prompt
from .env_loader import load_env
from .export import build_row, write_jsonl
from .messages import build_messages

load_env()

ensure_on_path()

from evaluate import (  # noqa: E402
    compute_quality,
    compute_strong_accuracy,
    compute_weak_accuracy,
    parse_ground_truth,
)

from .gemma3_parse import parse_hypotheses as parse_hypotheses_from_response  # noqa: E402


REFUSAL_HINTS = ("cannot", "unable to", "not able to", "i can't", "i cannot")


@dataclass
class InferenceResult:
    reply: str
    failure_mode: str | None  # None if normal parse succeeded


def classify_failure(reply: str, pred_hyps: list[str]) -> str | None:
    """Return a failure_mode string or None if the example scored normally."""
    if reply is None:
        return "api_error"
    if not reply.strip():
        return "empty_response"
    if pred_hyps:
        return None
    low = reply.lower()
    if any(hint in low for hint in REFUSAL_HINTS):
        return "refusal"
    return "parse_error"


RETRY_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)


async def _call_with_retry(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> str | None:
    """Return the raw reply text or None on hard failure."""
    messages = build_messages(system, user, model)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(min=1, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        reraise=True,
    ):
        with attempt:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content or ""
    return None  # pragma: no cover


async def _run_one(
    client: AsyncOpenAI,
    *,
    sem: asyncio.Semaphore,
    example_id: str,
    task_type: str,
    height: int,
    model_name: str,
    ontology: Any,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> dict[str, Any]:
    user_prompt = make_user_prompt(ontology)

    async with sem:
        try:
            reply = await _call_with_retry(
                client,
                model_name,
                SYSTEM_PROMPT,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                max_attempts=max_attempts,
            )
        except (RetryError, *RETRY_EXCEPTIONS):
            reply = None

    pred_hyps = parse_hypotheses_from_response(reply or "")
    gt_hyps = parse_ground_truth(ontology.hypotheses)

    strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
    weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, ontology.observations, ontology.theories)
    quality = compute_quality(pred_hyps, gt_hyps, ontology.observations, ontology.theories)
    if strong_acc == 1:
        weak_acc = 1
        quality = 1.0

    failure_mode = classify_failure(reply or "", pred_hyps)
    parse_failed = failure_mode is not None

    structural = compute_structural_annotations(ontology, task_type)

    return build_row(
        example_id=example_id,
        task=task_type,
        height=height,
        model=model_name,
        prompt_text=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        ontology=ontology,
        model_output=(reply or ""),
        is_correct_strong=bool(strong_acc),
        is_correct_weak=bool(weak_acc),
        quality_score=float(quality),
        parse_failed=parse_failed,
        failure_mode=failure_mode,
        error_type=None,  # filled by Phase 5 classifier
        structural=structural,
    )


async def run_inference(
    examples: list[Any],
    *,
    task_type: str,
    height: int,
    model_name: str,
    base_url: str | None,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
    example_id_prefix: str,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    coros = [
        _run_one(
            client,
            sem=sem,
            example_id=f"{example_id_prefix}_{i:05d}",
            task_type=task_type,
            height=height,
            model_name=model_name,
            ontology=ex,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        for i, ex in enumerate(examples)
    ]
    rows: list[dict[str, Any]] = await tqdm_asyncio.gather(*coros, desc=f"{task_type} h={height}")
    await client.close()
    return rows


def load_examples(pkl_path: Path) -> tuple[list[Any], str, int]:
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    return payload["examples"], payload["task_type"], payload["height"]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    strong = [r["is_correct_strong"] for r in rows]
    weak = [r["is_correct_weak"] for r in rows]
    parse_fail = sum(1 for r in rows if r["parse_failed"])
    failure_modes: dict[str, int] = {}
    for r in rows:
        fm = r["failure_mode"]
        if fm is None:
            continue
        failure_modes[fm] = failure_modes.get(fm, 0) + 1
    return {
        "n": n,
        "strong_accuracy": float(np.mean(strong)) if n else None,
        "weak_accuracy": float(np.mean(weak)) if n else None,
        "parse_fail_rate": parse_fail / n if n else None,
        "failure_modes": failure_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concurrent inference over an examples pickle")
    parser.add_argument("--examples-pkl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--model", required=True, help="Model name as accepted by the endpoint")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples")
    parser.add_argument("--id-prefix", type=str, default=None)
    args = parser.parse_args()

    examples, task_type, height = load_examples(args.examples_pkl)
    if args.limit is not None:
        examples = examples[: args.limit]

    prefix = args.id_prefix or f"{task_type}_h{height}"

    start = time.monotonic()
    rows = asyncio.run(
        run_inference(
            examples,
            task_type=task_type,
            height=height,
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_attempts=args.max_attempts,
            example_id_prefix=prefix,
        )
    )
    elapsed = time.monotonic() - start

    n = write_jsonl(rows, args.output)
    summary = _summarize(rows)
    summary["elapsed_seconds"] = elapsed
    summary["throughput_ips"] = n / elapsed if elapsed > 0 else None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
