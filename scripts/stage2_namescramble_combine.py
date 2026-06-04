#!/usr/bin/env python3
"""Combine per-height name-scramble inference files into per-task files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/stage2/namescramble_infer"))
    parser.add_argument("--conditions", nargs="+", default=("nonce", "natural"))
    parser.add_argument("--tasks", nargs="+", default=("infer_property", "infer_subtype"))
    parser.add_argument("--heights", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--summary", type=Path, default=Path("docs/namescramble_combine_summary.json"))
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "conditions": list(args.conditions),
        "tasks": list(args.tasks),
        "heights": list(args.heights),
        "outputs": {},
    }

    for condition in args.conditions:
        summary["outputs"][condition] = {}
        for task in args.tasks:
            combined: list[dict[str, Any]] = []
            inputs: list[dict[str, Any]] = []
            for height in args.heights:
                path = args.input_dir / condition / f"{task}_h{height}.jsonl"
                rows = read_jsonl(path)
                combined.extend(rows)
                inputs.append({"height": height, "path": str(path), "rows": len(rows)})
            out_path = args.input_dir / condition / f"{task}.jsonl"
            write_jsonl(out_path, combined)
            summary["outputs"][condition][task] = {
                "path": str(out_path),
                "rows": len(combined),
                "inputs": inputs,
            }
            print(out_path)

    write_json(args.summary, summary)
    print(args.summary)


if __name__ == "__main__":
    main()
