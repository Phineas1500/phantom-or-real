#!/usr/bin/env python3
"""Stage 2 Phase B.3.b: create 4B name-scramble evaluation sets and manifest."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_namescramble import (  # noqa: E402
    apply_name_mapping,
    build_name_mapping,
    build_natural_pool,
    build_nonce_pool,
    extract_row_symbols,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def model_file_stem(model_name: str, task: str) -> str:
    # model_name: gemma3-4b -> file stem in results/full/with_errortype
    model_slug = model_name.replace("-", "_")
    return f"{model_slug}_{task}"


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    height: int,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool = [row for row in rows if int(row["height"]) == height]
    if len(pool) < n:
        raise ValueError(f"not enough rows at h={height}: have={len(pool)} need={n}")
    rng = random.Random(seed + height * 101)
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    return [pool[idx] for idx in indices[:n]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/full/with_errortype"))
    parser.add_argument("--model-name", default="gemma3-4b")
    parser.add_argument("--tasks", nargs="+", default=("infer_property", "infer_subtype"))
    parser.add_argument("--heights", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--per-height", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--conditions", nargs="+", choices=("nonce", "natural"), default=("nonce", "natural"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/stage2/namescramble"))
    parser.add_argument("--nonce-pool-out", type=Path, default=Path("docs/stage2/nonce_pool.json"))
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=Path("results/stage2/probe_diagnostics_namescramble_4b.json"),
    )
    args = parser.parse_args()

    all_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    all_symbols: set[str] = set()
    for task in args.tasks:
        path = args.input_dir / f"{model_file_stem(args.model_name, task)}.jsonl"
        rows = read_jsonl(path)
        all_rows_by_task[task] = rows
        for row in rows:
            all_symbols.update(extract_row_symbols(row))

    nonce_pool = build_nonce_pool(all_symbols, seed=args.seed, multiplier=8)
    natural_pool = build_natural_pool(all_symbols)
    write_json(args.nonce_pool_out, nonce_pool)

    global_forbidden = set(all_symbols)
    generated_manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "tasks": list(args.tasks),
        "heights": list(args.heights),
        "per_height": args.per_height,
        "conditions": list(args.conditions),
        "seed": args.seed,
        "nonce_pool_path": str(args.nonce_pool_out),
        "natural_pool_size": len(natural_pool),
        "datasets": {},
    }

    for condition in args.conditions:
        condition_out = args.output_dir / condition
        generated_manifest["datasets"][condition] = {}
        for task in args.tasks:
            generated_manifest["datasets"][condition][task] = {}
            rows = all_rows_by_task[task]
            for height in args.heights:
                sampled = sample_rows(rows, height=height, n=args.per_height, seed=args.seed + hash(task) % 10_000)
                out_rows: list[dict[str, Any]] = []
                mapping_preview: list[dict[str, Any]] = []
                for idx, row in enumerate(sampled):
                    mapping = build_name_mapping(
                        row,
                        condition=condition,
                        nonce_pool=nonce_pool,
                        natural_pool=natural_pool,
                        global_forbidden=global_forbidden,
                        seed=args.seed + idx + height * 10_000,
                    )
                    new_row = apply_name_mapping(row, mapping, condition=condition)
                    out_rows.append(new_row)
                    if idx < 3:
                        mapping_preview.append(
                            {
                                "example_id": row["example_id"],
                                "mapping": mapping,
                            }
                        )
                out_path = condition_out / f"{task}_h{height}.jsonl"
                write_jsonl(out_path, out_rows)
                generated_manifest["datasets"][condition][task][f"h{height}"] = {
                    "output_path": str(out_path),
                    "rows": len(out_rows),
                    "mapping_preview": mapping_preview,
                    "status": "generated",
                    "next_step": "run inference + scoring + activation extraction; then evaluate fixed probe",
                }

    diagnostics_payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "status": "prepared_datasets_pending_model_eval",
        "inputs": {
            "input_dir": str(args.input_dir),
            "tasks": list(args.tasks),
            "heights": list(args.heights),
            "per_height": args.per_height,
            "conditions": list(args.conditions),
            "seed": args.seed,
        },
        "resources": {
            "nonce_pool_path": str(args.nonce_pool_out),
            "natural_pool_size": len(natural_pool),
            "dataset_manifest": generated_manifest["datasets"],
        },
        "metrics": {
            "auc_drop_vs_baseline": None,
            "flip_rate_vs_baseline": None,
        },
        "notes": [
            "This file records dataset generation only.",
            "Complete B.3.b requires inference + scoring + activation extraction + fixed-probe evaluation.",
        ],
    }

    write_json(args.diagnostics_out, diagnostics_payload)
    print(args.nonce_pool_out)
    print(args.diagnostics_out)
    print(args.output_dir)


if __name__ == "__main__":
    main()
