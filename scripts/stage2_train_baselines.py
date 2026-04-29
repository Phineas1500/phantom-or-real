#!/usr/bin/env python3
"""Train B0 metadata baselines (Phase 0.4).

Outputs per-(variant, task, split) JSON files under results/stage2/baselines/
and an aggregate b0_summary_<model_slug>.json.

CPU-only; run on a Scholar login node or a short interactive session.
Requires the 4B tokenizer to be cached locally (for prompt_token_count).
Set HF_HOME if your cache is not in the default location.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import VARIANT_FEATURE_NAMES, compute_token_counts, train_and_evaluate  # noqa: E402
from src.env_loader import load_env  # noqa: E402

VARIANTS = ["b0_height", "b0_prompt", "b0_namefreq"]
SPLITS = ["s1", "s2"]
TOKENIZER_MODEL = "google/gemma-3-4b-it"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_splits(splits_path: Path) -> dict[str, dict[str, str]]:
    """Return {split_name: {example_id: partition}} for s1 and s2."""
    s1: dict[str, str] = {}
    s2: dict[str, str] = {}
    with splits_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = row["example_id"]
            s1[eid] = row["s1"]
            s2[eid] = row["s2"]
    return {"s1": s1, "s2": s2}


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=ROOT / "results" / "full" / "with_errortype",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "results" / "stage2" / "splits.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "stage2" / "baselines",
    )
    parser.add_argument(
        "--model-slug",
        default="gemma3_4b",
        help="Must match the prefix used in the JSONL filenames",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help="HF_HOME override for tokenizer loading; defaults to $HF_HOME env var",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip prompt_token_count (only b0_height will work without the tokenizer)",
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    jsonl_paths = sorted(args.jsonl_dir.glob(f"{args.model_slug}_*.jsonl"))
    if not jsonl_paths:
        sys.exit(f"No JSONL files found for slug '{args.model_slug}' under {args.jsonl_dir}")

    print(f"Loading JSONL files:")
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    all_rows: list[dict] = []
    for p in jsonl_paths:
        rows = load_jsonl(p)
        task = rows[0]["task"] if rows else p.stem.replace(f"{args.model_slug}_", "")
        rows_by_task[task].extend(rows)
        all_rows.extend(rows)
        print(f"  {p.name}: {len(rows)} rows (task={task})")

    print(f"Loading splits from {args.splits}")
    splits = load_splits(args.splits)

    # ── Tokenizer (optional) ─────────────────────────────────────────────────
    token_counts: dict[str, int] | None = None
    if not args.no_tokenizer:
        import os
        hf_home = str(args.hf_cache) if args.hf_cache else os.environ.get("HF_HOME")
        print(f"Loading tokenizer {TOKENIZER_MODEL} (HF_HOME={hf_home}) ...")
        from transformers import AutoTokenizer
        tok_kwargs = {"cache_dir": hf_home} if hf_home else {}
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, **tok_kwargs)
        print(f"  Computing prompt token counts for {len(all_rows)} rows ...")
        token_counts = compute_token_counts(all_rows, tokenizer, model_name=TOKENIZER_MODEL)
        print(f"  Done. Example count: {len(token_counts)}")
    else:
        print("WARNING: --no-tokenizer set; only b0_height will be trained.")

    # ── Train and evaluate ───────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}  # (task, split, variant) -> best result

    for task, task_rows in sorted(rows_by_task.items()):
        print(f"\n── Task: {task} ({len(task_rows)} rows) ──")
        for split_name in SPLITS:
            split_assignment = splits[split_name]
            for variant in VARIANTS:
                if variant != "b0_height" and token_counts is None:
                    print(f"  Skipping {variant}/{split_name} (no tokenizer).")
                    continue

                print(f"  {variant} / {split_name} ...", end=" ", flush=True)
                try:
                    result = train_and_evaluate(
                        task_rows,
                        variant,
                        split_assignment,
                        token_counts=token_counts,
                    )
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    continue

                result["model_slug"] = args.model_slug
                result["task"] = task
                result["split"] = split_name
                result["generated_at"] = datetime.now(timezone.utc).isoformat()

                out_name = f"{variant}_{args.model_slug}_{task}_{split_name}.json"
                out_path = args.out_dir / out_name
                out_path.write_text(json.dumps(result, indent=2) + "\n")

                auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else "N/A"
                print(f"AUC={auc_str}, bal_acc={result['balanced_accuracy']:.4f} → {out_path.name}")

                # Track best B0 per (task, split, height) for summary.
                for h, h_auc in result["per_height_auc"].items():
                    key = f"{task}/{split_name}/h{h}"
                    if h_auc is not None:
                        prev = summary.get(key, {}).get("auc_for_height")
                        if prev is None or h_auc > prev:
                            summary[key] = {
                                "task": task,
                                "split": split_name,
                                "height": int(h),
                                "best_variant": variant,
                                "auc_for_height": h_auc,
                            }

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary_path = args.out_dir / f"b0_summary_{args.model_slug}.json"
    summary_doc = {
        "model_slug": args.model_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "strongest pre-output B0 per (task, split, height) for delta-reporting",
        "best_per_task_split_height": list(summary.values()),
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2) + "\n")
    print(f"\nWrote summary → {summary_path}")


if __name__ == "__main__":
    main()