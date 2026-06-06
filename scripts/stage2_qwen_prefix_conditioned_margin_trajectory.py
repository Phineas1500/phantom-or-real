#!/usr/bin/env python3
"""Score Qwen gold/hard-foil margins after original-output prefixes.

This is the Qwen analogue of ``stage2_prefix_conditioned_margin_trajectory.py``.
Qwen does not have a regenerated decode-trace artifact in the current row-set
manifest, so this script uses the existing hard-foil forced-choice artifact and
the original Stage 1 free-form output as the prefix trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_qwen_hard_foil_forced_choice import (  # noqa: E402
    ScoredSequence,
    json_default,
    load_hf_model,
    score_sequence_logprob,
    torch_dtype,
)
from src.activations import render_chat_text  # noqa: E402
from src.env_loader import load_env  # noqa: E402


@dataclass(frozen=True)
class SelectedRows:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_source_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open() as f:
        for idx, line in enumerate(f):
            if line.strip():
                rows[idx] = json.loads(line)
    return rows


def parse_checkpoints(value: str) -> list[int | str]:
    checkpoints: list[int | str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "final":
            checkpoints.append("final")
        else:
            checkpoints.append(int(item))
    if not checkpoints:
        raise ValueError("expected at least one checkpoint")
    return checkpoints


def expand_checkpoints(checkpoints: list[int | str], generated_token_count: int) -> list[dict[str, Any]]:
    expanded = []
    seen = set()
    for checkpoint in checkpoints:
        if checkpoint == "final":
            token_count = generated_token_count
            label = "final"
        else:
            token_count = min(max(int(checkpoint), 0), generated_token_count)
            label = str(checkpoint)
        key = (label, token_count)
        if key in seen:
            continue
        seen.add(key)
        expanded.append({"checkpoint": label, "prefix_token_count": token_count})
    return expanded


def candidate_to_dict(candidate: ScoredSequence | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "text": candidate.text,
        "token_count": len(candidate.token_ids),
        "logprob": candidate.logprob,
        "mean_logprob": candidate.mean_logprob,
    }


def select_recognition_gap_rows(
    *,
    rows: list[dict[str, Any]],
    model_key: str,
    task: str,
    height: int | None,
    limit: int | None,
    require_recognition_gap: bool,
) -> SelectedRows:
    candidates = []
    skipped_model = 0
    skipped_task = 0
    skipped_height = 0
    skipped_not_gap = 0
    skipped_parse = 0
    skipped_missing = 0

    for row in rows:
        if row.get("model_key") != model_key:
            skipped_model += 1
            continue
        if row.get("task") != task:
            skipped_task += 1
            continue
        if height is not None and int(row.get("height", -1)) != height:
            skipped_height += 1
            continue
        if row.get("mcq_choice_parse_failed"):
            skipped_parse += 1
            continue
        if not row.get("gold_hypothesis") or not row.get("foil_hypothesis"):
            skipped_missing += 1
            continue
        if require_recognition_gap and (
            bool(row.get("original_is_correct_strong")) or not bool(row.get("mcq_is_correct_choice"))
        ):
            skipped_not_gap += 1
            continue
        candidates.append(row)

    selected = sorted(candidates, key=lambda row: int(row["source_row_index"]))
    if limit is not None and limit >= 0:
        selected = selected[:limit]

    by_height: dict[str, int] = defaultdict(int)
    for row in candidates:
        by_height[f"h{row.get('height')}"] += 1

    return SelectedRows(
        rows=selected,
        summary={
            "selection_mode": "qwen_hard_foil_recognition_generation_gap"
            if require_recognition_gap
            else "qwen_hard_foil_rows",
            "model_key": model_key,
            "task": task,
            "height": height,
            "limit": limit,
            "available_total": len(candidates),
            "available_by_height": dict(sorted(by_height.items())),
            "selected_rows": len(selected),
            "skipped_model": skipped_model,
            "skipped_task": skipped_task,
            "skipped_height": skipped_height,
            "skipped_not_gap": skipped_not_gap,
            "skipped_parse": skipped_parse,
            "skipped_missing": skipped_missing,
        },
    )


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "fraction_below_0": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "fraction_below_0": float(np.mean(arr < 0.0)),
    }


def summarize_trajectory_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[str(row["checkpoint"])].append(row)

    checkpoint_summary = {}
    for checkpoint, subset in sorted(by_checkpoint.items(), key=lambda item: (item[1][0]["prefix_token_count"], item[0])):
        selected_vs_gold = [
            float(row["selected_vs_gold_logprob_margin"])
            for row in subset
            if row.get("selected_vs_gold_logprob_margin") is not None
        ]
        selected_vs_foil = [
            float(row["selected_vs_foil_logprob_margin"])
            for row in subset
            if row.get("selected_vs_foil_logprob_margin") is not None
        ]
        gold_vs_foil = [
            float(row["gold_vs_foil_logprob_margin"])
            for row in subset
            if row.get("gold_vs_foil_logprob_margin") is not None
        ]
        prefix_token_counts = [int(row["prefix_token_count"]) for row in subset]
        checkpoint_summary[checkpoint] = {
            "prefix_token_count": subset[0]["prefix_token_count"],
            "prefix_token_count_summary": {
                "min": min(prefix_token_counts),
                "max": max(prefix_token_counts),
                "mean": float(np.mean(prefix_token_counts)),
            },
            "n": len(subset),
            "selected_available": len(selected_vs_gold),
            "gold_vs_foil_logprob_margin": summarize_values(gold_vs_foil),
            "selected_vs_gold_logprob_margin": summarize_values(selected_vs_gold),
            "selected_vs_foil_logprob_margin": summarize_values(selected_vs_foil),
            "selected_vs_gold_nonnegative": sum(value >= 0.0 for value in selected_vs_gold),
            "gold_vs_foil_nonnegative": sum(value >= 0.0 for value in gold_vs_foil),
        }

    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_row[int(row["source_row_index"])].append(row)

    row_commitments = []
    for source_row_index, subset in sorted(by_row.items()):
        ordered = sorted(subset, key=lambda row: row["prefix_token_count"])
        first_selected = next(
            (
                row
                for row in ordered
                if row.get("selected_vs_gold_logprob_margin") is not None
                and row["selected_vs_gold_logprob_margin"] >= 0.0
            ),
            None,
        )
        first_gold = next(
            (
                row
                for row in ordered
                if row.get("gold_vs_foil_logprob_margin") is not None
                and row["gold_vs_foil_logprob_margin"] >= 0.0
            ),
            None,
        )
        row0 = ordered[0]
        row_commitments.append(
            {
                "source_row_index": source_row_index,
                "example_id": row0.get("example_id"),
                "height": row0.get("height"),
                "original_is_correct_strong": row0.get("original_is_correct_strong"),
                "original_parse_failed": row0.get("original_parse_failed"),
                "mcq_is_correct_choice": row0.get("mcq_is_correct_choice"),
                "first_selected_vs_gold_nonnegative_checkpoint": (
                    first_selected["checkpoint"] if first_selected else None
                ),
                "first_selected_vs_gold_nonnegative_prefix_tokens": (
                    first_selected["prefix_token_count"] if first_selected else None
                ),
                "first_gold_vs_foil_nonnegative_checkpoint": first_gold["checkpoint"] if first_gold else None,
                "first_gold_vs_foil_nonnegative_prefix_tokens": (
                    first_gold["prefix_token_count"] if first_gold else None
                ),
            }
        )

    return {
        "by_checkpoint": checkpoint_summary,
        "row_commitments": row_commitments,
        "rows_with_selected_available": sum(
            any(row.get("selected_vs_gold_logprob_margin") is not None for row in subset)
            for subset in by_row.values()
        ),
        "rows_selected_vs_gold_nonnegative_at_checkpoint_0": sum(
            any(
                row["prefix_token_count"] == 0
                and row.get("selected_vs_gold_logprob_margin") is not None
                and row["selected_vs_gold_logprob_margin"] >= 0.0
                for row in subset
            )
            for subset in by_row.values()
        ),
        "rows_gold_vs_foil_nonnegative_at_checkpoint_0": sum(
            any(
                row["prefix_token_count"] == 0
                and row.get("gold_vs_foil_logprob_margin") is not None
                and row["gold_vs_foil_logprob_margin"] >= 0.0
                for row in subset
            )
            for subset in by_row.values()
        ),
    }


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Qwen Prefix-Conditioned Margin Trajectory",
        "",
        f"Generated: `{report['created_at_utc']}`",
        "",
        "Purpose: score gold and the Stage 1 hard-foil hypothesis after original Qwen output-prefix checkpoints on recognition-gap rows.",
        "",
        "## Summary",
        "",
        f"- Rows: `{report['n_rows']}`; trajectory rows: `{report['n']}`.",
        f"- Available recognition-gap rows before limit: `{report['selection']['available_total']}`.",
        f"- Selected hypothesis source: `{report['selected_hypothesis_source']}`.",
        f"- Selected-vs-gold nonnegative at checkpoint 0 on `{summary['rows_selected_vs_gold_nonnegative_at_checkpoint_0']}/{report['n_rows']}` rows.",
        f"- Gold-vs-foil nonnegative at checkpoint 0 on `{summary['rows_gold_vs_foil_nonnegative_at_checkpoint_0']}/{report['n_rows']}` rows.",
        "",
        "## Interpretation Note",
        "",
        report["interpretation_note"],
        "",
        "## Checkpoint Summary",
        "",
        "| checkpoint | prefix tokens | n | selected avail. | selected-vs-gold mean | selected>=gold | gold-vs-foil mean | gold>=foil |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for checkpoint, row in summary["by_checkpoint"].items():
        token_summary = row.get("prefix_token_count_summary") or {}
        if token_summary.get("min") == token_summary.get("max"):
            token_display = str(token_summary.get("min"))
        else:
            token_display = f"{token_summary.get('min')}-{token_summary.get('max')}"
        lines.append(
            f"| {checkpoint} | {token_display} | {row['n']} | {row['selected_available']} | {fmt(row['selected_vs_gold_logprob_margin']['mean'])} | {row['selected_vs_gold_nonnegative']}/{row['selected_available']} | {fmt(row['gold_vs_foil_logprob_margin']['mean'])} | {row['gold_vs_foil_nonnegative']}/{row['n']} |"
        )

    lines += [
        "",
        "## Row Commitment Checkpoints",
        "",
        "| row | example | h | original strong | MCQ correct | first selected>=gold | first gold>=foil |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["row_commitments"]:
        lines.append(
            f"| {row['source_row_index']} | {row['example_id']} | {row['height']} | {row['original_is_correct_strong']} | {row['mcq_is_correct_choice']} | {row['first_selected_vs_gold_nonnegative_checkpoint']} | {row['first_gold_vs_foil_nonnegative_checkpoint']} |"
        )

    lines += [
        "",
        "## Causal-Abstraction Claim",
        "",
        report["causal_abstraction_claim"],
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recognition-jsonl",
        type=Path,
        default=Path("results/stage2/qwen_causal/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.jsonl"),
    )
    parser.add_argument("--source-jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_subtype.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model-key", default="qwen35_27b")
    parser.add_argument("--task", default="infer_subtype")
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--require-recognition-gap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoints", default="0,1,4,8,16,32,64,final")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/decode_time/qwen_prefix_conditioned_margin_trajectory_h4_subset.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.json"),
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.md"),
    )
    return parser


def main() -> int:
    load_env()
    args = build_arg_parser().parse_args()
    torch.set_grad_enabled(False)
    started = time.time()

    recognition_rows = read_jsonl(args.recognition_jsonl)
    source_rows = read_source_rows(args.source_jsonl)
    checkpoints = parse_checkpoints(args.checkpoints)
    selected = select_recognition_gap_rows(
        rows=recognition_rows,
        model_key=args.model_key,
        task=args.task,
        height=args.height,
        limit=args.limit,
        require_recognition_gap=args.require_recognition_gap,
    )
    if not selected.rows:
        raise ValueError(f"no rows selected: {selected.summary}")

    print("Qwen prefix-conditioned margin trajectory", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"recognition_rows={len(recognition_rows)} selected={len(selected.rows)}", flush=True)
    print(f"selection={selected.summary}", flush=True)
    print(f"checkpoints={checkpoints}", flush=True)
    print(f"transformers={package_version('transformers')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    if args.dry_run:
        first = selected.rows[0]
        print(
            json.dumps(
                {
                    "selection": selected.summary,
                    "first_row": {
                        "source_row_index": first.get("source_row_index"),
                        "example_id": first.get("example_id"),
                        "height": first.get("height"),
                        "gold_hypothesis": first.get("gold_hypothesis"),
                        "foil_hypothesis": first.get("foil_hypothesis"),
                        "original_margin_gold_minus_foil": first.get("original_margin_gold_minus_foil"),
                    },
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            )
        )
        return 0

    dtype = torch_dtype(args.dtype)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        device=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else None

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    trajectory_rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, row in enumerate(selected.rows, start=1):
            source_row_index = int(row["source_row_index"])
            source_row = source_rows.get(source_row_index)
            if source_row is None:
                raise ValueError(f"missing source row {source_row_index} in {args.source_jsonl}")
            prompt_text = render_chat_text(
                tokenizer,
                system=source_row["system_prompt"],
                user=source_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            generated_ids = tokenizer(source_row.get("model_output") or "", add_special_tokens=False)["input_ids"]
            if len(prompt_ids) + len(generated_ids) > args.n_ctx:
                raise ValueError(
                    f"row {source_row_index} exceeds n_ctx={args.n_ctx}: "
                    f"prompt={len(prompt_ids)} generated={len(generated_ids)}"
                )

            selected_hypothesis = row["foil_hypothesis"]
            expanded = expand_checkpoints(checkpoints, len(generated_ids))
            print(
                f"row {row_idx}/{len(selected.rows)} source_row={source_row_index} "
                f"h={row.get('height')} generated_tokens={len(generated_ids)} "
                f"checkpoints={len(expanded)}",
                flush=True,
            )
            for checkpoint in expanded:
                prefix_count = int(checkpoint["prefix_token_count"])
                prefix_ids = generated_ids[:prefix_count]
                context_ids = prompt_ids + prefix_ids
                prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                gold = score_sequence_logprob(
                    model=model,
                    prompt_token_ids=context_ids,
                    candidate_text=row["gold_hypothesis"],
                )
                foil = score_sequence_logprob(
                    model=model,
                    prompt_token_ids=context_ids,
                    candidate_text=row["foil_hypothesis"],
                )
                selected_score = foil
                output_row = {
                    "schema_version": 1,
                    "source_file": row.get("source_file"),
                    "source_row_index": source_row_index,
                    "example_id": row.get("example_id"),
                    "task": row.get("task"),
                    "height": row.get("height"),
                    "model": args.model,
                    "model_key": args.model_key,
                    "target_variable": "commitment_state",
                    "method": "qwen_prefix_conditioned_margin_trajectory",
                    "representation_type": "prompt_margin",
                    "checkpoint": checkpoint["checkpoint"],
                    "prefix_token_count": prefix_count,
                    "prefix_text": prefix_text,
                    "prompt_token_count": len(prompt_ids),
                    "generated_token_count": len(generated_ids),
                    "original_is_correct_strong": bool(row.get("original_is_correct_strong")),
                    "original_is_correct_weak": bool(row.get("original_is_correct_weak")),
                    "original_parse_failed": bool(row.get("original_parse_failed")),
                    "mcq_is_correct_choice": bool(row.get("mcq_is_correct_choice")),
                    "mcq_choice_parse_failed": bool(row.get("mcq_choice_parse_failed")),
                    "original_margin_gold_minus_foil": row.get("original_margin_gold_minus_foil"),
                    "mcq_margin_gold_minus_foil": row.get("mcq_margin_gold_minus_foil"),
                    "selected_hypothesis_source": "stage1_hard_foil",
                    "gold_hypothesis": row.get("gold_hypothesis"),
                    "foil_hypothesis": row.get("foil_hypothesis"),
                    "selected_hypothesis": selected_hypothesis,
                    "gold": candidate_to_dict(gold),
                    "foil": candidate_to_dict(foil),
                    "selected": candidate_to_dict(selected_score),
                    "gold_vs_foil_logprob_margin": gold.logprob - foil.logprob,
                    "gold_vs_foil_mean_logprob_margin": gold.mean_logprob - foil.mean_logprob,
                    "selected_vs_gold_logprob_margin": selected_score.logprob - gold.logprob,
                    "selected_vs_foil_logprob_margin": selected_score.logprob - foil.logprob,
                }
                trajectory_rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                fout.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_trajectory_rows(trajectory_rows)
    original_strong = [bool(row.get("original_is_correct_strong")) for row in selected.rows]
    original_parse = [bool(row.get("original_parse_failed")) for row in selected.rows]
    mcq_correct = [bool(row.get("mcq_is_correct_choice")) for row in selected.rows]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_qwen_prefix_conditioned_margin_trajectory.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "commitment_state",
        "method": "qwen_prefix_conditioned_margin_trajectory",
        "representation_type": "prompt_margin",
        "controls": ["historical_stage1_generation_prefix", "hard_foil_selected_hypothesis"],
        "recognition_jsonl": str(args.recognition_jsonl),
        "source_jsonl": str(args.source_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "selected_hypothesis_source": "stage1_hard_foil",
        "checkpoints": checkpoints,
        "selection": selected.summary,
        "n_rows": len(selected.rows),
        "n": len(trajectory_rows),
        "baseline_metrics": {
            "original_strong_accuracy": float(np.mean(original_strong)) if original_strong else None,
            "original_parse_fail_rate": float(np.mean(original_parse)) if original_parse else None,
            "mcq_choice_accuracy": float(np.mean(mcq_correct)) if mcq_correct else None,
        },
        "intervention_metrics": {},
        "paired_flips": {},
        "parse_fail_rate": {
            "original_generation": float(np.mean(original_parse)) if original_parse else None,
        },
        "matched_noise_summary": {},
        "summary": summary,
        "interpretation_note": (
            "This is not a causal result. It uses Qwen's original Stage 1 free-form output as "
            "the prefix trajectory and treats the hard foil as the emitted selected hypothesis. "
            "It is comparable to the Gemma prefix-conditioned diagnostic only as cross-model "
            "recognition-gap trajectory evidence; it is not a matched regenerated decode-trace replication."
        ),
        "causal_abstraction_claim": (
            "Predictive trajectory diagnostic only. It tests whether the Stage 1 hard-foil "
            "hypothesis is already more likely than gold under prompt and generated-prefix "
            "contexts; it does not perform a causal intervention."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")
    args.md_output.write_text(render_md(report) + "\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.md_output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
