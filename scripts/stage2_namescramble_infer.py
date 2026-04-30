#!/usr/bin/env python3
"""Run inference/scoring on name-scramble JSONL prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import sys

from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_loader import get_openai_inference_credentials, load_env  # noqa: E402
from src.gemma3_parse import parse_hypotheses  # noqa: E402
from src.inference import classify_failure  # noqa: E402
from src.messages import build_messages  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402

ensure_on_path()
from evaluate import compute_quality, compute_strong_accuracy, compute_weak_accuracy, parse_ground_truth  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


async def _chat_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> str:
    messages = build_messages(system, user, model)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(min=1, max=30),
        retry=retry_if_exception_type(Exception),
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
    return ""


async def _run_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    row: dict,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> dict:
    system = row.get("system_prompt", "")
    user = row["prompt_text"]
    async with sem:
        reply = await _chat_with_retry(
            client,
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
        )
    pred_hyps = parse_hypotheses(reply or "")
    gt_hyps = parse_ground_truth(row["ground_truth"])
    strong = compute_strong_accuracy(pred_hyps, gt_hyps)
    weak = compute_weak_accuracy(pred_hyps, gt_hyps, row["ontology_raw"]["observations"], row["ontology_raw"]["theories"])
    quality = compute_quality(pred_hyps, gt_hyps, row["ontology_raw"]["observations"], row["ontology_raw"]["theories"])
    failure_mode = classify_failure(reply or "", pred_hyps)
    out = dict(row)
    out["model_output"] = reply
    out["is_correct_strong"] = bool(strong)
    out["is_correct_weak"] = bool(weak)
    out["quality_score"] = float(quality)
    out["failure_mode"] = failure_mode
    out["parse_failed"] = failure_mode is not None
    return out


async def run_all(
    rows: list[dict],
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> list[dict]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_one(
            client,
            sem,
            row,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        for row in rows
    ]
    out = await asyncio.gather(*tasks)
    await client.close()
    return out


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    default_base, default_key = get_openai_inference_credentials()
    parser.add_argument("--base-url", default=default_base)
    parser.add_argument("--api-key", default=default_key)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    out_rows = asyncio.run(
        run_all(
            rows,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_attempts=args.max_attempts,
        )
    )
    write_jsonl(args.output_jsonl, out_rows)
    print(args.output_jsonl)


if __name__ == "__main__":
    main()
