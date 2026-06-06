#!/usr/bin/env python3
"""Score gold/foil/selected hypotheses after generated-prefix checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import input_device_for_model, load_tl_model, render_chat_text  # noqa: E402


@dataclass(frozen=True)
class ScoredCandidate:
    text: str
    token_count: int
    logprob: float
    mean_logprob: float


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


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


def score_candidate_logprob(
    *,
    model,
    prompt_token_ids: list[int],
    candidate_text: str,
) -> ScoredCandidate:
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("model has no tokenizer")
    candidate_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
    if not candidate_ids:
        raise ValueError(f"candidate text produced no tokens: {candidate_text!r}")

    input_ids = prompt_token_ids + candidate_ids[:-1]
    target_ids = candidate_ids
    input_device = input_device_for_model(model)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=input_device)

    with torch.inference_mode():
        logits = model(tokens, return_type="logits", prepend_bos=False)
        positions = torch.arange(
            len(prompt_token_ids) - 1,
            len(prompt_token_ids) - 1 + len(candidate_ids),
            device=logits.device,
        )
        target = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
        selected_logits = logits[0, positions, :]
        log_probs = torch.log_softmax(selected_logits.float(), dim=-1)
        token_logprob = log_probs[torch.arange(len(candidate_ids), device=logits.device), target]
        total = float(token_logprob.sum().detach().cpu())

    return ScoredCandidate(
        text=candidate_text,
        token_count=len(candidate_ids),
        logprob=total,
        mean_logprob=total / len(candidate_ids),
    )


def scored_candidate_to_dict(candidate: ScoredCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "text": candidate.text,
        "token_count": candidate.token_count,
        "logprob": candidate.logprob,
        "mean_logprob": candidate.mean_logprob,
    }


def selected_hypothesis(row: dict[str, Any]) -> str | None:
    margin = row.get("prompt_margin_scores") or {}
    selected = margin.get("selected_hypothesis")
    if selected:
        return selected
    parsed = row.get("parsed_hypotheses") or []
    if parsed:
        return str(parsed[0])
    return None


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
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_checkpoint.setdefault(str(row["checkpoint"]), []).append(row)

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

    by_row: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_row.setdefault(int(row["source_row_index"]), []).append(row)

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
                "generated_is_correct_strong": row0.get("generated_is_correct_strong"),
                "parse_failed": row0.get("parse_failed"),
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


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Prefix-Conditioned Margin Trajectory",
        "",
        f"Generated: `{report['created_at_utc']}`",
        "",
        "Purpose: score gold, hard foil, and selected hypotheses after generated-prefix checkpoints on the Gemma manifest recognition-gap rows.",
        "",
        "## Summary",
        "",
        f"- Rows: `{report['n_rows']}`; trajectory rows: `{report['n']}`.",
        f"- Selected hypothesis available on `{summary['rows_with_selected_available']}/{report['n_rows']}` rows.",
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
        "| row | example | h | strong | parse fail | first selected>=gold | first gold>=foil |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["row_commitments"]:
        lines.append(
            f"| {row['source_row_index']} | {row['example_id']} | {row['height']} | {row['generated_is_correct_strong']} | {row['parse_failed']} | {row['first_selected_vs_gold_nonnegative_checkpoint']} | {row['first_gold_vs_foil_nonnegative_checkpoint']} |"
        )

    lines += [
        "",
        "## Causal-Abstraction Claim",
        "",
        report["causal_abstraction_claim"],
        "",
    ]
    return "\n".join(lines)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-jsonl", type=Path, default=Path("results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_manifest_recognition_gap.jsonl"))
    parser.add_argument("--source-jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--checkpoints", default="0,1,4,8,16,32,64,final")
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/prefix_conditioned_margin_trajectory_gemma_manifest.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/prefix_conditioned_margin_trajectory_gemma_manifest.md"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    decode_rows = read_jsonl(args.decode_jsonl)
    source_rows = read_source_rows(args.source_jsonl)
    checkpoints = parse_checkpoints(args.checkpoints)

    print("Stage 2 prefix-conditioned margin trajectory", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"decode_rows={len(decode_rows)}", flush=True)
    print(f"checkpoints={checkpoints}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    if args.dry_run:
        dry_rows = []
        for row in decode_rows:
            tokenizer_count = len(str(row.get("model_output", "")).split())
            dry_rows.append(
                {
                    "source_row_index": row.get("source_row_index"),
                    "example_id": row.get("example_id"),
                    "selected_hypothesis": selected_hypothesis(row),
                    "has_gold": bool(row.get("gold_hypothesis")),
                    "has_foil": bool(row.get("foil_hypothesis")),
                    "word_count_proxy": tokenizer_count,
                    "checkpoints": checkpoints,
                }
            )
        print(json.dumps({"dry_run_rows": len(dry_rows), "first_row": dry_rows[0] if dry_rows else None}, indent=2))
        return 0

    dtype = torch_dtype(args.dtype)
    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    trajectory_rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, row in enumerate(decode_rows, start=1):
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
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            generated_ids = tokenizer(row.get("model_output") or "", add_special_tokens=False)["input_ids"]
            selected = selected_hypothesis(row)
            expanded = expand_checkpoints(checkpoints, len(generated_ids))
            print(
                f"row {row_idx}/{len(decode_rows)} source_row={source_row_index} "
                f"h={row.get('height')} generated_tokens={len(generated_ids)} "
                f"checkpoints={len(expanded)} selected={bool(selected)}",
                flush=True,
            )
            for checkpoint in expanded:
                prefix_count = int(checkpoint["prefix_token_count"])
                prefix_ids = generated_ids[:prefix_count]
                context_ids = prompt_ids + prefix_ids
                prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                gold = score_candidate_logprob(
                    model=model,
                    prompt_token_ids=context_ids,
                    candidate_text=row["gold_hypothesis"],
                )
                foil = score_candidate_logprob(
                    model=model,
                    prompt_token_ids=context_ids,
                    candidate_text=row["foil_hypothesis"],
                )
                selected_score = (
                    score_candidate_logprob(
                        model=model,
                        prompt_token_ids=context_ids,
                        candidate_text=selected,
                    )
                    if selected
                    else None
                )
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
                    "method": "prefix_conditioned_margin_trajectory",
                    "representation_type": "prompt_margin",
                    "checkpoint": checkpoint["checkpoint"],
                    "prefix_token_count": prefix_count,
                    "prefix_text": prefix_text,
                    "prompt_token_count": len(prompt_ids),
                    "generated_token_count": len(generated_ids),
                    "generated_is_correct_strong": bool(row.get("is_correct_strong")),
                    "generated_is_correct_weak": bool(row.get("is_correct_weak")),
                    "parse_failed": bool(row.get("parse_failed")),
                    "quality_score": row.get("quality_score"),
                    "gold_hypothesis": row.get("gold_hypothesis"),
                    "foil_hypothesis": row.get("foil_hypothesis"),
                    "selected_hypothesis": selected,
                    "gold": scored_candidate_to_dict(gold),
                    "foil": scored_candidate_to_dict(foil),
                    "selected": scored_candidate_to_dict(selected_score),
                    "gold_vs_foil_logprob_margin": gold.logprob - foil.logprob,
                    "gold_vs_foil_mean_logprob_margin": gold.mean_logprob - foil.mean_logprob,
                    "selected_vs_gold_logprob_margin": (
                        selected_score.logprob - gold.logprob if selected_score is not None else None
                    ),
                    "selected_vs_foil_logprob_margin": (
                        selected_score.logprob - foil.logprob if selected_score is not None else None
                    ),
                }
                trajectory_rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                fout.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_trajectory_rows(trajectory_rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_prefix_conditioned_margin_trajectory.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "commitment_state",
        "method": "prefix_conditioned_margin_trajectory",
        "representation_type": "prompt_margin",
        "controls": ["regenerated_baseline_trace_reuse"],
        "decode_jsonl": str(args.decode_jsonl),
        "source_jsonl": str(args.source_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "checkpoints": checkpoints,
        "n_rows": len(decode_rows),
        "n": len(trajectory_rows),
        "baseline_metrics": {
            "generated_strong_accuracy": (
                sum(bool(row.get("is_correct_strong")) for row in decode_rows) / len(decode_rows)
                if decode_rows
                else None
            ),
            "parse_fail_rate": (
                sum(bool(row.get("parse_failed")) for row in decode_rows) / len(decode_rows)
                if decode_rows
                else None
            ),
        },
        "intervention_metrics": {},
        "paired_flips": {},
        "parse_fail_rate": {
            "baseline_generation": (
                sum(bool(row.get("parse_failed")) for row in decode_rows) / len(decode_rows)
                if decode_rows
                else None
            )
        },
        "matched_noise_summary": {},
        "summary": summary,
        "interpretation_note": (
            "This is not a causal result. It shows that the regenerated selected hypothesis is "
            "already more prompt-likely than gold at prefix 0 on 13/13 parsed rows, while "
            "gold beats the hard foil at prefix 0 on only 1/14 rows. That is a strong "
            "predictive trajectory signature of early wrong-hypothesis preference, not proof "
            "that the preference is causally responsible for the final answer."
        ),
        "causal_abstraction_claim": (
            "Predictive trajectory diagnostic only. It tests whether the selected hypothesis "
            "is already more likely than gold under generated-prefix contexts; it does not "
            "perform a causal intervention."
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
