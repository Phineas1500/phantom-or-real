"""Phase 5: LLM-as-judge error type classification.

Per BEHAVIORAL_DATA_PLAN.md §5, we reproduce the paper's 4-category taxonomy
(Appendix H.1 / Figure 10) using gpt-5.4-nano as primary judge and gpt-5.4-mini
as fallback after an agreement check.

The prompt follows Figure 10 exactly in structure, with a single example per
category adapted from paper Appendix H.1 patterns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm_asyncio

from .env_loader import get_openai_gpt_credentials, load_env
from .export import read_jsonl

load_env()


ERROR_LABELS = {
    "Wrong ontology direction": "wrong_direction",
    "Fall back to trivial hypotheses": "trivial",
    "Ignore the ontology and produce unnecessary hypotheses": "unnecessary",
    "Hallucinated entities": "hallucinated",
}

CLASSIFICATION_PROMPT = """You are an expert evaluator of reasoning quality. Your task is to analyze a model's reasoning process for a question and determine the type of error it contains.

## Task
Given:
* A question
* The model's reasoning process
* The correct answer

Identify which error type best describes the mistake in the reasoning. Choose exactly one error type from the following categories.

## Error Types

**Error Type 1: Wrong ontology direction**
The hypothesis contains the wrong ontology direction. The model produces a subtype / property relation with subject and predicate reversed relative to the correct answer. Example: correct answer is "Every mammal is an animal", model produces "Every animal is a mammal".

**Error Type 2: Fall back to trivial hypotheses**
The hypothesis restates an observation verbatim instead of generalizing. Example: correct answer is "Every dalpist is not muffled"; model produces "Jerry is not muffled" (where "Jerry is not muffled" was already in the observations).

**Error Type 3: Ignore the ontology and produce unnecessary hypotheses**
The hypothesis is technically correct but unnecessary given the ontology. The model produces the correct generalization plus additional hypotheses that do not follow Occam's Razor. Example: correct answer is "Every lerpant is salty"; model produces "Every lerpant is salty" AND "Every yumpus is salty" (where yumpus is already a subtype of lerpant).

**Error Type 4: Hallucinated entities**
The reasoning relies on entities or concepts not present in the input. Example: the input mentions "Jerry" and "lerpant"; model produces "Every gwompant is salty" where "gwompant" does not appear in the input.

## Now evaluate the following case

Question: {question}
Model Reasoning: {model_output}
Correct Answer: {correct_answer}

## Output Format
Respond on a single line in exactly this format:
Error Type: <one of: Wrong ontology direction | Fall back to trivial hypotheses | Ignore the ontology and produce unnecessary hypotheses | Hallucinated entities>
"""


RETRY_EXC = (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError)


def parse_label(reply: str) -> str:
    low = reply.lower()
    for label, code in ERROR_LABELS.items():
        if label.lower() in low:
            return code
    return "unclassified"


def _is_gpt5_plus(model: str) -> bool:
    """GPT-5+ reasoning models require max_completion_tokens, reject temperature=0."""
    low = model.lower()
    return (
        low.startswith("gpt-5")
        or low.startswith("gpt-6")
        or low.startswith("o3")
        or low.startswith("o4")
    )


async def _classify_one(
    client: AsyncOpenAI,
    *,
    model: str,
    sem: asyncio.Semaphore,
    row: dict[str, Any],
) -> str:
    prompt = CLASSIFICATION_PROMPT.format(
        question=row["prompt_text"],
        model_output=row["model_output"],
        correct_answer=row["ground_truth"],
    )
    if _is_gpt5_plus(model):
        kwargs = {"max_completion_tokens": 512}  # gpt-5 nano/mini spend budget on hidden reasoning
    else:
        kwargs = {"temperature": 0, "max_tokens": 64}

    async with sem:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_random_exponential(min=1, max=30),
                retry=retry_if_exception_type(RETRY_EXC),
                reraise=True,
            ):
                with attempt:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs,
                    )
                    return parse_label(completion.choices[0].message.content or "")
        except Exception:
            return "unclassified"
    return "unclassified"  # pragma: no cover


async def classify_failures(
    failed_rows: list[dict[str, Any]],
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    concurrency: int,
) -> list[str]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    coros = [_classify_one(client, model=model, sem=sem, row=r) for r in failed_rows]
    labels = await tqdm_asyncio.gather(*coros, desc=f"classify:{model}")
    await client.close()
    return labels


def write_jsonl_with_error_types(in_path: Path, out_path: Path, labels_by_id: dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row["is_correct_strong"]:
                row["error_type"] = labels_by_id.get(row["example_id"], row.get("error_type"))
            fout.write(json.dumps(row, ensure_ascii=False))
            fout.write("\n")


def classify_file(
    in_path: Path,
    out_path: Path,
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    concurrency: int,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(in_path)
    failed = [r for r in rows if not r["is_correct_strong"]]
    if limit is not None:
        failed = failed[:limit]

    labels = asyncio.run(
        classify_failures(
            failed,
            model=model,
            base_url=base_url,
            api_key=api_key,
            concurrency=concurrency,
        )
    )

    labels_by_id = {row["example_id"]: lbl for row, lbl in zip(failed, labels)}
    write_jsonl_with_error_types(in_path, out_path, labels_by_id)

    dist: dict[str, int] = {}
    for lbl in labels:
        dist[lbl] = dist.get(lbl, 0) + 1
    return {"n_classified": len(labels), "distribution": dist}


def main() -> None:
    default_base_url, default_key = get_openai_gpt_credentials()
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="Judge model. Default is gpt-5.4-mini because our nano-vs-mini agreement "
             "check ran 57.5% (below the plan's 85% threshold); the shipped dataset was "
             "classified with mini. Pass --model gpt-5.4-nano to re-run on the cheaper tier.",
    )
    p.add_argument("--base-url", default=default_base_url)
    p.add_argument("--api-key", default=default_key)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No API key for GPT judge. Set OPENAI_API_KEY_GPT (preferred, keeps it "
            "separate from the inference endpoint key) or OPENAI_API_KEY."
        )

    summary = classify_file(
        args.input,
        args.output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
