"""Classify incorrect rows that have error_type=null.

After rescoring via scripts/rescore_jsonl.py, some rows flip correct->incorrect
and end up with error_type=null since the Phase 5 classifier was run before.
This script finds those rows per file, classifies them with gpt-5.4-mini,
and rewrites the JSONL in place.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_loader import get_openai_gpt_credentials, load_env  # noqa: E402
from src.error_classification import classify_failures  # noqa: E402

load_env()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--pattern", default="gemma3_*.jsonl")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    base_url, api_key = get_openai_gpt_credentials()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY_GPT not set.")

    summaries = []
    for path in sorted(args.in_dir.glob(args.pattern)):
        with path.open() as f:
            rows = [json.loads(l) for l in f if l.strip()]

        missing = [r for r in rows if not r["is_correct_strong"] and r.get("error_type") is None]
        if not missing:
            summaries.append({"path": str(path), "missing": 0})
            continue

        labels = asyncio.run(
            classify_failures(
                missing, model=args.model, base_url=base_url, api_key=api_key,
                concurrency=args.concurrency,
            )
        )
        id_to_label = {r["example_id"]: lbl for r, lbl in zip(missing, labels)}
        for r in rows:
            if r["example_id"] in id_to_label:
                r["error_type"] = id_to_label[r["example_id"]]

        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")
        tmp.replace(path)
        summaries.append({"path": str(path), "missing": len(missing), "labels": labels})

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
