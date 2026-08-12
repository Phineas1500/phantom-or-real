#!/usr/bin/env python3
"""Self-judgement census (SJ/OC reconnaissance; future-directions Tier-1 item,
step 1 — registration-free by design; informs the SJ/OC factorization
registration).

For a seeded balanced draw of stage-1 rows, replay each row's original prompt
and the model's own stage-1 answer as an assistant turn, then ask the model
whether its answer was correct. Census the objective-correctness x
self-judgement 2x2. Black-box generation only; venue-agnostic by design
(no activations, no cross-job comparisons; the causal SJ/OC arms are a
separate, Scholar-lane registration).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.messages import build_messages  # noqa: E402

SJ_QUESTION = (
    "Look back at the answer you just gave. Was your final proposed hypothesis "
    "exactly correct? Reply with a single word: yes or no."
)


def select_rows(jsonl: Path, *, per_cell: int, seed: int) -> list[dict]:
    pool: dict[tuple[int, bool], list[dict]] = {}
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            row = json.loads(line)
            if row.get("height") not in (3, 4) or row.get("parse_failed"):
                continue
            row["row_index"] = row_index
            pool.setdefault((row["height"], bool(row["is_correct_strong"])), []).append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for cell in sorted(pool, key=str):
        candidates = pool[cell]
        if len(candidates) < per_cell:
            raise ValueError(f"cell {cell}: only {len(candidates)} rows, need {per_cell}")
        selected.extend(rng.sample(candidates, per_cell))
    return sorted(selected, key=lambda r: r["row_index"])


def parse_sj(text: str) -> str | None:
    word = text.strip().lower().strip(".,!\"' \n")
    if word.startswith("yes"):
        return "yes"
    if word.startswith("no"):
        return "no"
    return None


async def judge_row(client, args, row, sem) -> dict:
    messages = build_messages(row["system_prompt"], row["prompt_text"], args.model)
    messages = messages + [
        {"role": "assistant", "content": row["model_output"]},
        {"role": "user", "content": SJ_QUESTION},
    ]
    samples = []
    async with sem:
        for _ in range(args.samples):
            for attempt in range(4):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model, messages=messages,
                        max_tokens=8, temperature=args.temperature,
                    )
                    samples.append(resp.choices[0].message.content or "")
                    break
                except Exception:
                    if attempt == 3:
                        samples.append("")
                    else:
                        await asyncio.sleep(2 * (attempt + 1))
    parsed = [parse_sj(s) for s in samples]
    votes = Counter(p for p in parsed if p)
    if votes.get("yes", 0) > votes.get("no", 0):
        sj = "yes"
    elif votes.get("no", 0) > votes.get("yes", 0):
        sj = "no"
    else:
        sj = "tie"
    return {
        "source_row_index": row["row_index"],
        "height": row["height"],
        "is_correct_strong": bool(row["is_correct_strong"]),
        "sj_majority": sj,
        "sj_samples": samples,
        "sj_parsed": parsed,
        "n_parsed": sum(1 for p in parsed if p),
        "unanimous": len({p for p in parsed if p}) == 1 and any(parsed),
    }


async def main_async(args) -> int:
    from openai import AsyncOpenAI

    rows = select_rows(args.jsonl, per_cell=args.per_cell, seed=args.seed)
    print(f"selected {len(rows)} rows ({args.per_cell} per height x correctness cell)", flush=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120)
    sem = asyncio.Semaphore(args.concurrency)
    out_rows = []
    tasks = [judge_row(client, args, row, sem) for row in rows]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        out_rows.append(await coro)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)} judged", flush=True)

    out_rows.sort(key=lambda r: r["source_row_index"])
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cells = Counter((r["is_correct_strong"], r["sj_majority"]) for r in out_rows)
    n = len(out_rows)
    census = {f"oc_{oc}__sj_{sj}": cells.get((oc, sj), 0) for oc in (True, False) for sj in ("yes", "no", "tie")}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "n_rows": n,
        "per_cell": args.per_cell,
        "seed": args.seed,
        "samples_per_row": args.samples,
        "temperature": args.temperature,
        "sj_question": SJ_QUESTION,
        "census": census,
        "census_fractions": {k: v / n for k, v in census.items()},
        "conflict_cells": {
            "confident_wrong_oc_false_sj_yes": cells.get((False, "yes"), 0),
            "unconfident_right_oc_true_sj_no": cells.get((True, "no"), 0),
        },
        "sj_yes_rate_overall": sum(1 for r in out_rows if r["sj_majority"] == "yes") / n,
        "unanimous_rate": sum(1 for r in out_rows if r["unanimous"]) / n,
        "parse_ok_rate": sum(r["n_parsed"] for r in out_rows) / (n * args.samples),
        "by_height": {
            str(h): dict(Counter((r["is_correct_strong"], r["sj_majority"]) for r in out_rows if r["height"] == h).most_common())
            for h in (3, 4)
        },
        "out_jsonl": str(args.out_jsonl),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("census", "conflict_cells", "sj_yes_rate_overall", "unanimous_rate", "parse_ok_rate")}, indent=2))
    print(f"wrote {args.out_jsonl} and {args.summary}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default="gemma3-27b")
    parser.add_argument("--per-cell", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/sj_census/sj_census_2k.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("docs/sj_census_2k_summary.json"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
