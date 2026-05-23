#!/usr/bin/env python3
"""Merge Qwen inference shards into the fixed Stage 1 JSONL layout."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


TASK_SLUG = {"property": "infer_property", "ontology": "infer_subtype"}
TASK_FROM_SLUG = {value: key for key, value in TASK_SLUG.items()}
EXAMPLE_ID_RE = re.compile(r"^(property|ontology)_h(\d+)_(\d+)$")
FORBIDDEN_OUTPUT_SNIPPETS = ("Thinking Process:",)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            row = {key: value for key, value in row.items() if key != "_source_file"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expected_count(examples_dir: Path, task: str, height: int) -> int:
    with (examples_dir / f"examples_{task}_h{height}.pkl").open("rb") as f:
        payload = pickle.load(f)
    return len(payload["examples"])


def parse_example_id(row: dict[str, Any]) -> tuple[str, int, int]:
    example_id = row.get("example_id")
    if not isinstance(example_id, str):
        raise ValueError(f"row from {row.get('_source_file')} has no string example_id")
    match = EXAMPLE_ID_RE.match(example_id)
    if match is None:
        raise ValueError(f"bad example_id {example_id!r} from {row.get('_source_file')}")
    task, height, index = match.groups()
    return task, int(height), int(index)


def validate_row(row: dict[str, Any], *, task_slug: str) -> None:
    if row.get("task") != task_slug:
        raise ValueError(f"{row.get('_source_file')} row task {row.get('task')!r} != {task_slug!r}")
    output = row.get("model_output") or ""
    if any(snippet in output for snippet in FORBIDDEN_OUTPUT_SNIPPETS):
        raise ValueError(f"{row.get('_source_file')} row {row.get('example_id')} contains Qwen thinking output")


def merge_task(
    *,
    shard_root: Path,
    examples_dir: Path,
    output_dir: Path,
    model_slug: str,
    task: str,
    heights: list[int],
    allow_incomplete: bool,
) -> dict[str, Any]:
    task_slug = TASK_SLUG[task]
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    pattern = f"*_{task_slug}.jsonl"
    for path in sorted(shard_root.rglob(pattern)):
        for row in read_jsonl(path):
            validate_row(row, task_slug=task_slug)
            parsed_task, height, index = parse_example_id(row)
            if parsed_task != task or height not in heights:
                continue
            key = (height, index)
            if key in rows_by_key:
                duplicates.append(
                    {
                        "example_id": row["example_id"],
                        "first": rows_by_key[key].get("_source_file"),
                        "second": row.get("_source_file"),
                    }
                )
                continue
            rows_by_key[key] = row
    if duplicates:
        raise ValueError(f"duplicate shard rows for {task_slug}: {duplicates[:5]}")

    missing: dict[str, list[int]] = {}
    for height in heights:
        n_expected = expected_count(examples_dir, task, height)
        height_missing = [idx for idx in range(n_expected) if (height, idx) not in rows_by_key]
        if height_missing:
            missing[f"h{height}"] = height_missing
    if missing and not allow_incomplete:
        preview = {height: values[:10] for height, values in missing.items()}
        raise ValueError(f"missing shard rows for {task_slug}: {preview}")

    merged = [rows_by_key[key] for key in sorted(rows_by_key)]
    out_path = output_dir / f"{model_slug}_{task_slug}.jsonl"
    write_jsonl(merged, out_path)
    per_height = {}
    for height in heights:
        height_rows = [row for (row_height, _), row in sorted(rows_by_key.items()) if row_height == height]
        n = len(height_rows)
        strong = sum(bool(row.get("is_correct_strong")) for row in height_rows)
        parse_failed = sum(bool(row.get("parse_failed")) for row in height_rows)
        per_height[f"h{height}"] = {
            "n": n,
            "expected": expected_count(examples_dir, task, height),
            "complete": n == expected_count(examples_dir, task, height),
            "strong_accuracy": strong / n if n else None,
            "parse_fail_rate": parse_failed / n if n else None,
        }
    meta = {
        "model_slug": model_slug,
        "task": task,
        "task_slug": task_slug,
        "n_rows": len(merged),
        "allow_incomplete": allow_incomplete,
        "per_height": per_height,
    }
    meta_path = output_dir / f"{model_slug}_{task_slug}_runmeta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"jsonl": str(out_path), "runmeta": str(meta_path), **meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, default=Path("data/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/full/with_errortype"))
    parser.add_argument("--model-slug", default="qwen35_27b")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_SLUG), default=tuple(TASK_SLUG))
    parser.add_argument("--heights", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--summary", type=Path, default=Path("docs/qwen35_27b_inference_merge_summary.json"))
    args = parser.parse_args()

    reports = [
        merge_task(
            shard_root=args.shard_root,
            examples_dir=args.examples_dir,
            output_dir=args.output_dir,
            model_slug=args.model_slug,
            task=task,
            heights=args.heights,
            allow_incomplete=args.allow_incomplete,
        )
        for task in args.tasks
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"reports": reports}, indent=2))
    print(args.summary)


if __name__ == "__main__":
    main()
