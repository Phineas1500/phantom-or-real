#!/usr/bin/env python3
"""Run Qwen hard-foil forced-choice checks on Stage 1 failures.

This is the Qwen/Hugging Face counterpart to the Gemma hard-foil branch in
``stage2_steer_answer_property_margins.py``, but without steering. It asks
whether Qwen recognizes the gold hypothesis when its free-form Stage 1 answer
was wrong and the foil is the model's own emitted wrong hypothesis.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_steer_answer_property_margins import (  # noqa: E402
    build_margin_prompt,
    emitted_wrong_foil,
)
from scripts.stage2_steer_forced_choice_direction import parse_choice  # noqa: E402
from src.activations import render_chat_text  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.stage2_probes import read_jsonl, read_split_assignments  # noqa: E402
from src.stage2_steering import parse_int_list  # noqa: E402


@dataclass(frozen=True)
class ScoredSequence:
    text: str
    token_ids: list[int]
    logprob: float
    mean_logprob: float


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
    return str(value)


def first_parameter_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def input_device_for_model(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        return first_parameter_device(model.get_input_embeddings())
    return first_parameter_device(model)


def load_hf_model(
    model_name: str,
    *,
    dtype: torch.dtype,
    device_map: str,
    device: str,
    attn_implementation: str | None,
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map != "none":
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device_map == "none":
        model.to(device)
    model.eval()
    model.tokenizer = tokenizer
    return model, tokenizer


def score_sequence_logprob(
    *,
    model,
    prompt_token_ids: list[int],
    candidate_text: str,
) -> ScoredSequence:
    tokenizer = model.tokenizer
    candidate_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
    if not candidate_ids:
        raise ValueError(f"candidate text produced no tokens: {candidate_text!r}")

    input_ids = prompt_token_ids + candidate_ids[:-1]
    target_ids = candidate_ids
    device = input_device_for_model(model)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=tokens, use_cache=False)
        logits = outputs.logits
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

    return ScoredSequence(
        text=candidate_text,
        token_ids=list(candidate_ids),
        logprob=total,
        mean_logprob=total / len(candidate_ids),
    )


def generate_one_hf(
    *,
    model,
    token_ids: list[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    stop_at_eos: bool,
) -> tuple[list[int], str]:
    tokenizer = model.tokenizer
    device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    if stop_at_eos and tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(**kwargs)
    new_ids = output[0, tokens.shape[1] :].detach().cpu().tolist()
    return new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def select_rows(
    *,
    rows: list[dict[str, Any]],
    jsonl_path: Path,
    splits_path: Path,
    split_family: str,
    split: str,
    heights: list[int],
    limit: int,
    seed: int,
    baseline_incorrect_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments = read_split_assignments(splits_path)
    source_file = str(jsonl_path)
    height_set = set(heights)
    candidates: list[dict[str, Any]] = []
    available_by_height: dict[str, int] = defaultdict(int)
    skipped_parse_failed = 0
    skipped_correct = 0
    skipped_no_foil = 0
    skipped_split = 0

    for row_index, row in enumerate(rows):
        if int(row.get("height", -1)) not in height_set:
            continue
        assignment = assignments.get((source_file, row_index))
        if assignment is None or assignment.get(f"{split_family}_split") != split:
            skipped_split += 1
            continue
        if row.get("parse_failed"):
            skipped_parse_failed += 1
            continue
        if baseline_incorrect_only and bool(row.get("is_correct_strong")):
            skipped_correct += 1
            continue
        foil = emitted_wrong_foil(row)
        if foil is None:
            skipped_no_foil += 1
            continue
        selected = dict(row)
        selected["row_index"] = row_index
        selected["margin_foil_hypothesis"] = foil
        candidates.append(selected)
        available_by_height[f"h{row.get('height')}"] += 1

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected_rows = sorted(candidates[:limit], key=lambda row: int(row["row_index"]))
    summary = {
        "source_file": source_file,
        "split_family": split_family,
        "split": split,
        "heights": heights,
        "limit": limit,
        "seed": seed,
        "selection_mode": "incorrect_rows_with_stage1_model_foil"
        if baseline_incorrect_only
        else "rows_with_stage1_model_foil",
        "available_by_height": dict(sorted(available_by_height.items())),
        "available_total": len(candidates),
        "selected_rows": len(selected_rows),
        "baseline_incorrect_only": baseline_incorrect_only,
        "skipped_split": skipped_split,
        "skipped_parse_failed": skipped_parse_failed,
        "skipped_correct": skipped_correct,
        "skipped_no_foil": skipped_no_foil,
    }
    return selected_rows, summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_height: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_height[f"h{row.get('height')}"].append(row)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "mcq_choice_accuracy": sum(bool(row["mcq_is_correct_choice"]) for row in group) / n,
            "mcq_parse_fail_rate": sum(bool(row["mcq_choice_parse_failed"]) for row in group) / n,
            "mean_original_margin": sum(float(row["original_margin_gold_minus_foil"]) for row in group) / n,
            "mean_mcq_margin": sum(float(row["mcq_margin_gold_minus_foil"]) for row in group) / n,
            "mcq_parsed_choice_counts": {
                choice: sum(row.get("mcq_parsed_choice") == choice for row in group)
                for choice in ("A", "B")
            },
        }

    return {
        "overall": summarize_group(rows),
        "by_height": {height: summarize_group(group) for height, group in sorted(by_height.items())},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_subtype.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model-key", default="qwen35_27b")
    parser.add_argument("--task", default="infer_subtype")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--heights", default="4")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=20260523)
    parser.add_argument("--option-seed", type=int, default=20260430)
    parser.add_argument("--baseline-incorrect-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
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
        default=Path("results/stage2/qwen_causal/qwen35_27b_subtype_h4_hardfoil_forced_choice.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/qwen35_27b_subtype_h4_hardfoil_forced_choice.json"),
    )
    return parser


def main() -> int:
    load_env()
    args = build_arg_parser().parse_args()
    started = time.time()
    torch.set_grad_enabled(False)
    rows = read_jsonl(args.jsonl)
    heights = parse_int_list(args.heights)
    selected_rows, selection_summary = select_rows(
        rows=rows,
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        split_family=args.split_family,
        split=args.split,
        heights=heights,
        limit=args.limit,
        seed=args.selection_seed,
        baseline_incorrect_only=args.baseline_incorrect_only,
    )
    if not selected_rows:
        raise ValueError(f"no rows selected: {selection_summary}")

    first_prompt = build_margin_prompt(
        selected_rows[0],
        row_index=int(selected_rows[0]["row_index"]),
        option_seed=args.option_seed,
        foil_source="stage1_model_output",
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selection": selection_summary,
                    "first_selected": {
                        "row_index": selected_rows[0]["row_index"],
                        "example_id": selected_rows[0].get("example_id"),
                        "height": selected_rows[0].get("height"),
                        "gold_hypothesis": first_prompt.gold_hypothesis,
                        "foil_hypothesis": first_prompt.foil_hypothesis,
                        "mcq_user": first_prompt.user,
                    },
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            )
        )
        return 0

    dtype = torch_dtype(args.dtype)
    print("Qwen hard-foil forced-choice check", flush=True)
    print(f"model={args.model} task={args.task} rows={len(selected_rows)} selection={selection_summary}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}", flush=True)
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
    output_rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_number, row in enumerate(selected_rows, start=1):
            source_row_index = int(row["row_index"])
            forced = build_margin_prompt(
                row,
                row_index=source_row_index,
                option_seed=args.option_seed,
                foil_source="stage1_model_output",
            )
            original_prompt = render_chat_text(
                tokenizer,
                system=row["system_prompt"],
                user=row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )
            mcq_prompt = render_chat_text(
                tokenizer,
                system=forced.system,
                user=forced.user,
                model_name=args.model,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )
            original_tokens = tokenizer(original_prompt, add_special_tokens=False)["input_ids"]
            mcq_tokens = tokenizer(mcq_prompt, add_special_tokens=False)["input_ids"]
            if len(original_tokens) > args.n_ctx or len(mcq_tokens) > args.n_ctx:
                raise ValueError(
                    f"row {source_row_index} exceeds n_ctx={args.n_ctx}: "
                    f"original={len(original_tokens)} mcq={len(mcq_tokens)}"
                )

            original_gold = score_sequence_logprob(
                model=model,
                prompt_token_ids=original_tokens,
                candidate_text=forced.gold_hypothesis,
            )
            original_foil = score_sequence_logprob(
                model=model,
                prompt_token_ids=original_tokens,
                candidate_text=forced.foil_hypothesis,
            )
            mcq_gold_text = f"({forced.gold_choice})"
            mcq_foil_text = "(B)" if forced.gold_choice == "A" else "(A)"
            mcq_gold = score_sequence_logprob(
                model=model,
                prompt_token_ids=mcq_tokens,
                candidate_text=mcq_gold_text,
            )
            mcq_foil = score_sequence_logprob(
                model=model,
                prompt_token_ids=mcq_tokens,
                candidate_text=mcq_foil_text,
            )
            new_ids, reply = generate_one_hf(
                model=model,
                token_ids=mcq_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                stop_at_eos=args.stop_at_eos,
            )
            parsed_choice = parse_choice(reply)
            output = {
                "schema_version": 1,
                "source_file": str(args.jsonl),
                "source_row_index": source_row_index,
                "example_id": row.get("example_id"),
                "task": row.get("task"),
                "height": row.get("height"),
                "model": args.model,
                "model_key": args.model_key,
                "original_is_correct_strong": bool(row.get("is_correct_strong")),
                "original_is_correct_weak": bool(row.get("is_correct_weak")),
                "original_parse_failed": bool(row.get("parse_failed")),
                "gold_hypothesis": forced.gold_hypothesis,
                "foil_hypothesis": forced.foil_hypothesis,
                "foil_source": "stage1_model_output",
                "mcq_gold_choice": forced.gold_choice,
                "mcq_option_a": forced.option_a,
                "mcq_option_b": forced.option_b,
                "original_prompt_token_count": len(original_tokens),
                "mcq_prompt_token_count": len(mcq_tokens),
                "original_gold_logprob": original_gold.logprob,
                "original_foil_logprob": original_foil.logprob,
                "original_margin_gold_minus_foil": original_gold.logprob - original_foil.logprob,
                "mcq_gold_logprob": mcq_gold.logprob,
                "mcq_foil_logprob": mcq_foil.logprob,
                "mcq_margin_gold_minus_foil": mcq_gold.logprob - mcq_foil.logprob,
                "mcq_generated_token_count": len(new_ids),
                "mcq_model_output": reply,
                "mcq_parsed_choice": parsed_choice,
                "mcq_choice_parse_failed": parsed_choice is None,
                "mcq_is_correct_choice": parsed_choice == forced.gold_choice,
            }
            output_rows.append(output)
            fout.write(json.dumps(output, ensure_ascii=False, default=json_default) + "\n")
            fout.flush()
            print(
                f"row {row_number}/{len(selected_rows)} source={source_row_index} "
                f"h={row.get('height')} orig_margin={output['original_margin_gold_minus_foil']:.3f} "
                f"mcq_margin={output['mcq_margin_gold_minus_foil']:.3f} "
                f"choice={parsed_choice} correct={output['mcq_is_correct_choice']}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_qwen_hard_foil_forced_choice.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "selection": selection_summary,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "device_map": args.device_map,
            "chat_template_kwargs": chat_template_kwargs or {},
        },
        "summary": summarize_rows(output_rows),
        "out_jsonl": str(args.out_jsonl),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
