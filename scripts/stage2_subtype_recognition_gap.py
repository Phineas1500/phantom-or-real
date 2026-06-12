#!/usr/bin/env python3
"""Subtype behavioral prerequisites on the Modal endpoint.

Identifies Gemma subtype recognition-gap rows (strong-wrong free-form,
hard-foil forced choice picks gold), then validates the localization donors
on them: hint-first concept hint and gold-among-candidates. Output manifest
makes the subtype concept-position patch job queue-ready. Task 10 /
claims-table row 12.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_steer_answer_property_margins import build_margin_prompt, emitted_wrong_foil  # noqa: E402
from scripts.stage2_steer_forced_choice_direction import GENERATION_REQUEST_RE, parse_choice  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.inference import build_messages  # noqa: E402
from src.stage2_probes import read_split_assignments  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402

GENERATION_REQUEST = "Please come up with hypotheses to explain observations."


def select_candidate_rows(args) -> list[dict[str, Any]]:
    assignments = read_split_assignments(args.splits)
    rows = []
    with args.jsonl.open() as f:
        for row_index, line in enumerate(f):
            row = json.loads(line)
            if row.get("height") not in (3, 4) or row.get("parse_failed") or row.get("is_correct_strong"):
                continue
            assignment = assignments.get((str(args.jsonl), row_index))
            if assignment is None or assignment.get("s1_split") != "test":
                continue
            if emitted_wrong_foil(row) is None:
                continue
            row["row_index"] = row_index
            rows.append(row)
            if len(rows) >= args.max_scan_rows:
                break
    return rows


def hint_user(row: dict[str, Any], kind: str, fc) -> str:
    concept = row["ontology_fol_structured"]["hypothesis"]["subject"]
    base = GENERATION_REQUEST_RE.sub("", row["prompt_text"]).strip()
    if kind == "baseline":
        return row["prompt_text"]
    if kind == "hint_concept_first":
        return f"Hint: the hypothesis should be about {concept}.\n\n{base}\n\n{GENERATION_REQUEST}"
    if kind == "candidates_gold_present":
        listing = f"- {fc.gold_hypothesis}\n- {fc.foil_hypothesis}"
        return f"{base}\n\nCandidate hypotheses to consider:\n{listing}\n\n{GENERATION_REQUEST}"
    raise ValueError(kind)


async def main_async(args) -> int:
    from openai import AsyncOpenAI

    load_env()
    started = time.time()
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    ensure_on_path()
    candidates = select_candidate_rows(args)
    print(f"scan_rows={len(candidates)}", flush=True)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def recognition(row):
        fc = build_margin_prompt(row, row_index=int(row["row_index"]), option_seed=args.option_seed, foil_source="stage1_model_output")
        messages = build_messages(fc.system, fc.user, args.model)
        async with semaphore:
            response = await client.chat.completions.create(model=args.model, messages=messages, n=1, temperature=0.0, max_tokens=8)
        choice = parse_choice((response.choices[0].message.content or "").strip())
        return row, fc, choice, choice == fc.gold_choice

    recog = await asyncio.gather(*(recognition(row) for row in candidates))
    gap = [(row, fc) for row, fc, choice, ok in recog if ok][: args.max_gap_rows]
    print(f"recognition_correct={sum(ok for *_, ok in recog)}/{len(recog)}; using gap rows={len(gap)}", flush=True)

    async def arm_run(row, fc, kind):
        messages = build_messages(row["system_prompt"], hint_user(row, kind, fc), args.model)
        async with semaphore:
            response = await client.chat.completions.create(model=args.model, messages=messages, n=args.samples_per_row, temperature=args.temperature, max_tokens=args.max_tokens)
        out = []
        for sample_index, choice in enumerate(response.choices):
            reply = (choice.message.content or "").strip()
            score = score_reply(row, reply)
            out.append({
                "source_row_index": int(row["row_index"]),
                "height": row.get("height"),
                "condition": kind,
                "sample_index": sample_index,
                "gold_hypothesis": fc.gold_hypothesis,
                "hard_foil_hypothesis": fc.foil_hypothesis,
                "model_output": reply,
                **score,
            })
        return out

    kinds = ("baseline", "hint_concept_first", "candidates_gold_present")
    batches = await asyncio.gather(*(arm_run(row, fc, kind) for row, fc in gap for kind in kinds))
    await client.close()
    rows_out = [item for batch in batches for item in batch]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = defaultdict(lambda: defaultdict(list))
    for r in rows_out:
        by[r["condition"]][r["source_row_index"]].append(bool(r["is_correct_strong"]))
    summary = {}
    for cond in sorted(by):
        flat = [s for v in by[cond].values() for s in v]
        summary[cond] = {"n_rows": len(by[cond]), "strong_accuracy": float(np.mean(flat))}
        print(f"{cond}: P(strong)={summary[cond]['strong_accuracy']:.3f}", flush=True)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "script": "scripts/stage2_subtype_recognition_gap.py",
        "model": args.model,
        "endpoint": "modal_openai_compatible",
        "task": "infer_subtype",
        "target_variable": "target_concept",
        "method": "subtype_recognition_gap_prerequisites",
        "option_seed": args.option_seed,
        "scan": {
            "scanned": len(recog),
            "recognition_correct": int(sum(ok for *_, ok in recog)),
        },
        "gap_row_indices": [int(row["row_index"]) for row, _ in gap],
        "summary": summary,
        "n": len(rows_out),
        "causal_abstraction_claim": (
            "Subtype prerequisites for the localization replication: recognition-gap rowset plus "
            "behavioral validation of the hint-first donor and candidates effect. Prompt-level "
            "evidence; the concept-position patch job consumes gap_row_indices."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_subtype.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--model", default="gemma3-27b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-scan-rows", type=int, default=48)
    parser.add_argument("--max-gap-rows", type=int, default=16)
    parser.add_argument("--option-seed", type=int, default=20260430)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/subtype_recognition_gap_27b.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/subtype_recognition_gap_27b_manifest.json"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
