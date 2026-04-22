"""Phase 5 Stage 1: nano vs mini agreement check on 200 sampled failures.

Per BEHAVIORAL_DATA_PLAN.md §5.3:
  - Agreement >=90% -> proceed with nano for the full run (~$15)
  - 85-90%          -> proceed with nano, flag low-confidence in report
  - <85%            -> step up to mini (~$50)

Samples failures roughly proportionally across (task, height, model) buckets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_loader import get_openai_gpt_credentials, load_env  # noqa: E402
from src.error_classification import classify_failures  # noqa: E402
from src.export import read_jsonl  # noqa: E402

load_env()


def sample_failures(
    paths: list[Path], n: int, seed: int = 20260420
) -> list[dict]:
    rng = random.Random(seed)
    by_bucket: dict[tuple, list[dict]] = defaultdict(list)
    for p in paths:
        for r in read_jsonl(p):
            if not r["is_correct_strong"]:
                by_bucket[(r["model"], r["task"], r["height"])].append(r)

    n_buckets = len(by_bucket)
    per_bucket = max(1, n // n_buckets)
    sampled: list[dict] = []
    for rows in by_bucket.values():
        rng.shuffle(rows)
        sampled.extend(rows[:per_bucket])

    rng.shuffle(sampled)
    return sampled[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", nargs="+", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--nano-model", default="gpt-5.4-nano")
    p.add_argument("--mini-model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("results/error_agreement.json"))
    args = p.parse_args()

    base_url, api_key = get_openai_gpt_credentials()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY_GPT not set.")

    sampled = sample_failures(args.jsonl, args.n)
    print(f"sampled {len(sampled)} failures across {len(set((r['model'], r['task'], r['height']) for r in sampled))} buckets")

    nano = asyncio.run(
        classify_failures(
            sampled, model=args.nano_model, base_url=base_url, api_key=api_key,
            concurrency=args.concurrency,
        )
    )
    mini = asyncio.run(
        classify_failures(
            sampled, model=args.mini_model, base_url=base_url, api_key=api_key,
            concurrency=args.concurrency,
        )
    )

    agree = sum(1 for a, b in zip(nano, mini) if a == b)
    n = len(sampled)
    rate = agree / n if n else 0.0

    disagreements = Counter()
    for a, b in zip(nano, mini):
        if a != b:
            disagreements[f"{a} <> {b}"] += 1

    summary = {
        "n": n,
        "agreement_rate": rate,
        "nano_distribution": dict(Counter(nano)),
        "mini_distribution": dict(Counter(mini)),
        "disagreements": dict(disagreements),
        "recommendation": (
            "proceed_with_nano" if rate >= 0.90
            else "proceed_with_nano_flag_low_conf" if rate >= 0.85
            else "step_up_to_mini"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
