#!/usr/bin/env python3
"""Capture unhinted L30 concept-position states for naturally-correct vs
naturally-incorrect property rows (pre-registered item F(i), capture lane).

No hints, no generation: one prompt forward per row; saves per-row state
blocks at gold-concept mention positions plus a manifest with labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_rank_k_guard_v2 import COMPOSITE_MANIFEST_ROWS  # noqa: E402
from scripts.stage2_subtype_discriminator import json_default  # noqa: E402


def select_balanced_rows(
    jsonl: Path,
    *,
    exclude: set[int],
    heights: list[int],
    per_cell: int,
    seed: int,
) -> list[dict]:
    pool: dict[tuple[int, bool], list[dict]] = {}
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            if row_index in exclude or not line.strip():
                continue
            row = json.loads(line)
            height = row.get("height")
            if height not in heights or row.get("parse_failed"):
                continue
            key = (height, bool(row.get("is_correct_strong")))
            row["row_index"] = row_index
            pool.setdefault(key, []).append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for height in heights:
        for label in (True, False):
            candidates = pool.get((height, label), [])
            if len(candidates) < per_cell:
                raise ValueError(f"cell (h{height}, correct={label}): {len(candidates)} < {per_cell}")
            selected.extend(rng.sample(candidates, per_cell))
    return sorted(selected, key=lambda row: row["row_index"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-cell", type=int, default=24)
    parser.add_argument("--selection-seed", type=int, default=20260705)
    parser.add_argument("--guard-report", type=Path, default=Path("docs/rank_k_guard_v2_27b_property_shard0of2.json"))
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/stage2/erasure/natural_state_capture_27b_property_L30.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage2/erasure/natural_state_capture_27b_property_L30.manifest.jsonl"))
    args = parser.parse_args()
    started = time.time()

    heights = [int(part) for part in args.heights.split(",")]
    guard_rows = set(json.load(args.guard_report.open())["selection"]["all_selected_rows"])
    exclude = set(COMPOSITE_MANIFEST_ROWS) | guard_rows
    rows = select_balanced_rows(
        args.jsonl, exclude=exclude, heights=heights, per_cell=args.per_cell, seed=args.selection_seed
    )
    cells = {}
    for row in rows:
        key = f"h{row['height']}_{'correct' if row.get('is_correct_strong') else 'incorrect'}"
        cells[key] = cells.get(key, 0) + 1
    print(f"selected {len(rows)} rows: {cells} (excluded {len(exclude)} basis-provenance rows)", flush=True)
    if args.dry_run:
        print(json.dumps({"rows": [r["row_index"] for r in rows]}, default=json_default))
        return 0

    import torch
    from dotenv import load_dotenv

    from scripts.stage2_decode_time_correction import torch_dtype  # noqa: E402
    from scripts.stage2_hint_delta import concept_positions, prompt_cache  # noqa: E402
    from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
    from src.bd_path import ensure_on_path  # noqa: E402

    load_dotenv()
    torch.set_grad_enabled(False)
    ensure_on_path()
    model = load_tl_model(
        args.model, n_devices=args.n_devices, n_ctx=args.n_ctx,
        dtype=torch_dtype(args.dtype), load_mode=args.load_mode,
    )
    hook_name = validate_hooks(model, [args.layer])[0]
    tokenizer = model.tokenizer
    print(f"using_hook={hook_name}", flush=True)

    arrays: dict[str, np.ndarray] = {}
    kept = 0
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as manifest:
        for i, row in enumerate(rows):
            gold_concept = row["ontology_fol_structured"]["hypothesis"]["subject"]
            text = render_chat_text(
                tokenizer, system=row["system_prompt"], user=row["prompt_text"],
                model_name=args.model, add_generation_prompt=True,
            )
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            positions = concept_positions(tokenizer, text, gold_concept, 0, len(token_ids))
            if not positions:
                print(f"skip row {row['row_index']}: no concept positions", flush=True)
                continue
            cache = prompt_cache(model, token_ids, [hook_name])
            states = cache[hook_name][positions].detach().cpu().numpy().astype(np.float32)
            arrays[f"L{args.layer}_row{row['row_index']}_unhinted_concept_states"] = states
            manifest.write(json.dumps({
                "source_row_index": row["row_index"],
                "example_id": row.get("example_id"),
                "height": row.get("height"),
                "is_correct_strong": bool(row.get("is_correct_strong")),
                "gold_concept": gold_concept,
                "n_concept_positions": len(positions),
                "prompt_token_count": len(token_ids),
            }, default=json_default) + "\n")
            kept += 1
            if (i + 1) % 16 == 0:
                print(f"captured {i + 1}/{len(rows)} rows ({time.time() - started:.0f}s)", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kept_rows": kept,
        "selected_rows": len(rows),
        "elapsed_seconds": time.time() - started,
        "output": str(args.output),
        "manifest": str(args.manifest),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
