#!/usr/bin/env python3
"""Validate Qwen Stage 1 JSONLs before downstream Stage 2 extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_OUTPUT_SNIPPETS = (
    "Thinking Process:",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return rows


def validate(path: Path, *, min_rows: int | None) -> dict[str, Any]:
    rows = read_jsonl(path)
    if min_rows is not None and len(rows) < min_rows:
        raise ValueError(f"{path} has {len(rows)} rows, expected at least {min_rows}")

    bad_rows: list[str] = []
    for row in rows:
        output = row.get("model_output") or ""
        if any(snippet in output for snippet in FORBIDDEN_OUTPUT_SNIPPETS):
            bad_rows.append(str(row.get("example_id")))
    if bad_rows:
        preview = ", ".join(bad_rows[:10])
        raise ValueError(f"{path} contains Qwen thinking output in rows: {preview}")

    heights = sorted({row.get("height") for row in rows})
    parse_failed = sum(bool(row.get("parse_failed")) for row in rows)
    strong = sum(bool(row.get("is_correct_strong")) for row in rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "heights": heights,
        "parse_fail_rate": parse_failed / len(rows) if rows else None,
        "strong_accuracy": strong / len(rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--min-rows", type=int, default=None)
    args = parser.parse_args()

    summaries = [validate(path, min_rows=args.min_rows) for path in args.jsonl]
    print(json.dumps({"ok": True, "files": summaries}, indent=2))


if __name__ == "__main__":
    main()
