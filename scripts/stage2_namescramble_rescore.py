#!/usr/bin/env python3
"""Rescore completed name-scramble generations against corrected references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bd_path import ensure_on_path  # noqa: E402
from src.gemma3_parse import parse_hypotheses  # noqa: E402
from src.inference import classify_failure  # noqa: E402
from src.stage2_namescramble import repair_name_mapping_references  # noqa: E402

ensure_on_path()
from evaluate import compute_quality, compute_strong_accuracy, compute_weak_accuracy, parse_ground_truth  # noqa: E402


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


def rescore_row(reference: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    reply = generated.get("model_output", "") or ""
    pred_hyps = parse_hypotheses(reply)
    gt_hyps = parse_ground_truth(reference["ground_truth"])
    strong = compute_strong_accuracy(pred_hyps, gt_hyps)
    weak = compute_weak_accuracy(
        pred_hyps,
        gt_hyps,
        reference["ontology_raw"]["observations"],
        reference["ontology_raw"]["theories"],
    )
    quality = compute_quality(
        pred_hyps,
        gt_hyps,
        reference["ontology_raw"]["observations"],
        reference["ontology_raw"]["theories"],
    )
    failure_mode = classify_failure(reply, pred_hyps)

    out = dict(reference)
    out["model_output"] = reply
    out["is_correct_strong"] = bool(strong)
    out["is_correct_weak"] = bool(weak)
    out["quality_score"] = float(quality)
    out["failure_mode"] = failure_mode
    out["parse_failed"] = failure_mode is not None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    generated_rows = read_jsonl(args.generated_jsonl)
    prepared_by_id: dict[str, dict[str, Any]] = {}
    if args.prepared_jsonl is not None:
        prepared_rows = read_jsonl(args.prepared_jsonl)
        prepared_by_id = {str(row["example_id"]): row for row in prepared_rows}

    rescored: list[dict[str, Any]] = []
    missing: list[str] = []
    for generated in generated_rows:
        example_id = str(generated["example_id"])
        if prepared_by_id:
            reference = prepared_by_id.get(example_id)
            if reference is None:
                missing.append(example_id)
                continue
        else:
            reference = repair_name_mapping_references(generated)
        rescored.append(rescore_row(reference, generated))

    if missing:
        raise SystemExit(f"{args.generated_jsonl}: missing {len(missing)} prepared ids, first={missing[:5]}")
    if len(rescored) != len(generated_rows):
        raise SystemExit(f"row count mismatch: generated={len(generated_rows)} rescored={len(rescored)}")

    output = args.output_jsonl or args.generated_jsonl
    write_jsonl(output, rescored)
    strong = sum(bool(row.get("is_correct_strong")) for row in rescored)
    weak = sum(bool(row.get("is_correct_weak")) for row in rescored)
    parse_failed = sum(bool(row.get("parse_failed")) for row in rescored)
    n = len(rescored)
    print(json.dumps({
        "output": str(output),
        "rows": n,
        "strong_accuracy": strong / n if n else None,
        "weak_accuracy": weak / n if n else None,
        "parse_fail_rate": parse_failed / n if n else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
