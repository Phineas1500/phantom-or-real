#!/usr/bin/env python3
"""Steer Qwen HF generations with a raw answer-property direction.

This mirrors ``stage2_steer_answer_property_direction.py`` for Qwen by using
Hugging Face hooks at ``model.model.layers[L]`` instead of TransformerLens
hooks. The answer-label, row-selection, and summary logic are intentionally
shared with the Gemma script so the reports stay comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_qwen_patch_hf import (  # noqa: E402
    load_hf_model,
    render_tokens,
    torch_dtype,
    validate_hf_layers,
)
from scripts.stage2_qwen_steer_raw_direction import (  # noqa: E402
    generate_one,
    make_hf_steering_hook,
)
from scripts.stage2_steer_answer_property_direction import (  # noqa: E402
    AnswerContent,
    add_answer_metrics,
    answer_sign,
    gold_answer_content,
    load_resume_rows,
    make_answer_condition_plan,
    parse_condition_kinds,
    save_direction_artifact,
    select_answer_rows,
    serializable_direction_summary,
    summarize_answer_rows,
    train_answer_probe_direction,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.stage2_probes import read_jsonl  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    make_orthogonal_unit_direction,
    parse_float_list,
    parse_int_list,
    score_reply,
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def output_row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["source_row_index"]), str(row["condition"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model-key", default="qwen35_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-prefix", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--answer-label-source", choices=("gold", "stage1_model_output"), default="gold")
    parser.add_argument(
        "--answer-target",
        choices=("polarity", "predicate_pair", "predicate_one_vs_rest"),
        default="polarity",
    )
    parser.add_argument("--positive-predicate", default=None)
    parser.add_argument("--negative-predicate", default=None)
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260523)
    parser.add_argument("--orthogonal-seed", type=int, default=20260545)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--conditions", default="baseline,toward_gold,away_gold,orthogonal")
    parser.add_argument("--strengths", default="0.5,1")
    parser.add_argument(
        "--intervention-scope",
        choices=("prompt_only", "last_token_each_forward"),
        default="last_token_each_forward",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
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
        default=Path("results/stage2/steering/qwen35_answer_property_l45_polarity_smoke.jsonl"),
    )
    parser.add_argument(
        "--direction-output",
        type=Path,
        default=Path("results/stage2/steering/qwen35_answer_property_l45_polarity_direction.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/qwen35_answer_property_steering_27b_l45_polarity_smoke.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed (source_row_index, condition) rows already present in --out-jsonl.",
    )
    return parser


def main() -> int:
    load_env()
    args = build_arg_parser().parse_args()
    torch.set_grad_enabled(False)

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)

    heights = parse_int_list(args.heights)
    strengths = parse_float_list(args.strengths)
    condition_plan = make_answer_condition_plan(
        condition_kinds=parse_condition_kinds(args.conditions),
        strengths=strengths,
    )
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)
    activation_prefix = (
        args.activation_prefix
        if args.activation_prefix is not None
        else args.activation_dir / f"{args.model_key}_{args.task}_L{args.layer}"
    )
    activation_path = activation_prefix.with_suffix(".safetensors")
    sidecar_path = activation_prefix.with_suffix(".example_ids.jsonl")
    source_rows = read_jsonl(args.jsonl)
    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else None

    print("Qwen HF answer/property steering", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"answer_label_source={args.answer_label_source}", flush=True)
    print(f"answer_target={args.answer_target}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"intervention_scope={args.intervention_scope}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    started = time.time()
    direction = train_answer_probe_direction(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        source_rows=source_rows,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        label_source=args.answer_label_source,
        answer_target=args.answer_target,
        positive_predicate=args.positive_predicate,
        negative_predicate=args.negative_predicate,
        c_values=parse_float_list(args.c_values),
        max_iter=args.max_iter,
        solver=args.solver,
    )
    orthogonal_direction = make_orthogonal_unit_direction(
        direction["unit_direction"],
        seed=args.orthogonal_seed,
    )
    save_direction_artifact(
        path=args.direction_output,
        direction=direction,
        orthogonal_direction=orthogonal_direction,
    )
    print(
        "direction: "
        f"best_c={direction['best_c']} val_auc={direction['val_auc']:.4f} "
        f"test_auc={direction['test_auc']:.4f} proj_std={direction['train_projection_std']:.4f}",
        flush=True,
    )

    selected_rows, selection_summary = select_answer_rows(
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=heights,
        per_height_label=args.per_height_label,
        seed=args.selection_seed,
        label_source=args.answer_label_source,
        answer_target=args.answer_target,
        positive_predicate=args.positive_predicate,
        negative_predicate=args.negative_predicate,
        drop_parse_failed=True,
    )
    print(
        f"selected_rows={len(selected_rows)} "
        f"available_counts={selection_summary['available_counts']}",
        flush=True,
    )
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(
        "scorer_preflight: "
        f"strong={scorer_preflight['is_correct_strong']} "
        f"parse_failed={scorer_preflight['parse_failed']}",
        flush=True,
    )

    if args.dry_run:
        payload = {
            "direction": serializable_direction_summary(direction),
            "selection": selection_summary,
            "conditions": [condition.__dict__ for condition in condition_plan],
            "chat_template_kwargs": chat_template_kwargs or {},
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
        return 0

    model, tokenizer = load_hf_model(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        device=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    validate_hf_layers(model, [args.layer])
    print(f"using_hook=model.model.layers.{args.layer}.output", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    expected_keys = {
        (int(row["row_index"]), condition.label)
        for row in selected_rows
        for condition in condition_plan
    }
    existing_rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    resume_malformed_rows = 0
    resume_ignored_rows = 0
    if args.resume:
        existing_rows_by_key, resume_malformed_rows, resume_ignored_rows = load_resume_rows(
            args.out_jsonl,
            expected_keys,
        )
        print(
            "resume: "
            f"loaded={len(existing_rows_by_key)} malformed={resume_malformed_rows} "
            f"ignored={resume_ignored_rows} expected={len(expected_keys)}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    baseline_content_by_source: dict[int, AnswerContent] = {}
    if existing_rows_by_key:
        for stage1_row in selected_rows:
            source_row_index = int(stage1_row["row_index"])
            for condition in condition_plan:
                existing_row = existing_rows_by_key.get((source_row_index, condition.label))
                if existing_row is None:
                    continue
                rows.append(existing_row)
                if condition.direction_kind is None:
                    baseline_content_by_source[source_row_index] = AnswerContent(
                        predicate=existing_row.get("parsed_predicate"),
                        negated=existing_row.get("parsed_negated"),
                    )

        with args.out_jsonl.open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    vector_by_kind = {
        "answer": direction["unit_direction"],
        "orthogonal": orthogonal_direction,
    }
    projection_std = float(direction["train_projection_std"])
    output_mode = "a" if args.resume else "w"
    with args.out_jsonl.open(output_mode) as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            source_row_index = int(stage1_row["row_index"])
            token_ids = render_tokens(
                tokenizer=tokenizer,
                row=stage1_row,
                model_name=args.model,
                chat_template_kwargs=chat_template_kwargs,
            )
            if len(token_ids) > args.n_ctx:
                raise ValueError(f"row {source_row_index} exceeds n_ctx={args.n_ctx}: {len(token_ids)}")
            gold = gold_answer_content(stage1_row)
            print(
                f"row {row_idx}/{len(selected_rows)} "
                f"source_row={source_row_index} h={stage1_row['height']} "
                f"original_correct={stage1_row['is_correct_strong']} "
                f"gold={gold.predicate}/negated={gold.negated} prompt_tokens={len(token_ids)}",
                flush=True,
            )
            for condition in condition_plan:
                row_key = (source_row_index, condition.label)
                if row_key in existing_rows_by_key:
                    existing_row = existing_rows_by_key[row_key]
                    print(
                        f"  {condition.label}: resume_skip "
                        f"strong={existing_row['is_correct_strong']} "
                        f"polarity={existing_row['parsed_negated']} "
                        f"pred={existing_row['parsed_predicate']} "
                        f"parse_failed={existing_row['parse_failed']}",
                        flush=True,
                    )
                    continue

                hook_state = {"calls": 0, "applications": 0}
                handle = None
                signed_delta = 0.0
                try:
                    if condition.direction_kind is not None:
                        sign_to_gold = answer_sign(int(stage1_row["answer_label"]))
                        if condition.direction_kind == "toward_gold":
                            signed_delta = sign_to_gold * condition.strength_sd * projection_std
                            vector = vector_by_kind["answer"]
                        elif condition.direction_kind == "away_gold":
                            signed_delta = -sign_to_gold * condition.strength_sd * projection_std
                            vector = vector_by_kind["answer"]
                        elif condition.direction_kind == "orthogonal":
                            signed_delta = sign_to_gold * condition.strength_sd * projection_std
                            vector = vector_by_kind["orthogonal"]
                        else:
                            raise ValueError(f"unknown direction kind {condition.direction_kind!r}")
                        hook_fn, hook_state = make_hf_steering_hook(
                            vector=vector,
                            delta=signed_delta,
                            scope=args.intervention_scope,
                        )
                        handle = model.model.layers[args.layer].register_forward_hook(hook_fn)
                    new_ids, reply = generate_one(
                        model=model,
                        token_ids=token_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        stop_at_eos=args.stop_at_eos,
                    )
                finally:
                    if handle is not None:
                        handle.remove()

                score = score_reply(stage1_row, reply)
                baseline_content = baseline_content_by_source.get(source_row_index)
                answer_metrics = add_answer_metrics(
                    stage1_row=stage1_row,
                    score=score,
                    baseline_content=baseline_content,
                )
                if condition.direction_kind is None:
                    baseline_content_by_source[source_row_index] = AnswerContent(
                        predicate=answer_metrics["parsed_predicate"],
                        negated=answer_metrics["parsed_negated"],
                    )
                    answer_metrics["baseline_predicate"] = answer_metrics["parsed_predicate"]
                    answer_metrics["baseline_negated"] = answer_metrics["parsed_negated"]
                    answer_metrics["answer_content_changed_vs_baseline"] = False

                output_row = {
                    "schema_version": 1,
                    "source_file": source_file,
                    "source_row_index": source_row_index,
                    "example_id": stage1_row.get("example_id"),
                    "task": stage1_row.get("task"),
                    "height": stage1_row.get("height"),
                    "model": args.model,
                    "original_model": stage1_row.get("model"),
                    "original_is_correct_strong": bool(stage1_row.get("is_correct_strong")),
                    "original_is_correct_weak": bool(stage1_row.get("is_correct_weak")),
                    "original_parse_failed": bool(stage1_row.get("parse_failed")),
                    "answer_label": int(stage1_row["answer_label"]),
                    "condition": condition.label,
                    "direction_kind": condition.direction_kind,
                    "strength_sd": condition.strength_sd,
                    "signed_strength_sd": signed_delta / projection_std if projection_std else None,
                    "intervention_delta_l2": abs(signed_delta),
                    "intervention_scope": args.intervention_scope,
                    "hook_calls": int(hook_state["calls"]),
                    "hook_applications": int(hook_state["applications"]),
                    "prompt_token_count": len(token_ids),
                    "generated_token_count": len(new_ids),
                    "model_output": reply,
                    **score,
                    **answer_metrics,
                }
                rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"  {condition.label}: strong={output_row['is_correct_strong']} "
                    f"polarity={output_row['parsed_negated']} pred={output_row['parsed_predicate']} "
                    f"parse_failed={output_row['parse_failed']} new_tokens={len(new_ids)} "
                    f"hooks={hook_state['applications']}/{hook_state['calls']}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_qwen_steer_answer_property_direction.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "layer": args.layer,
        "hook_name": f"model.model.layers.{args.layer}.output",
        "activation_path": str(activation_path),
        "sidecar_path": str(sidecar_path),
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "direction_output": str(args.direction_output),
        "out_jsonl": str(args.out_jsonl),
        "probe_direction": serializable_direction_summary(direction),
        "selection": selection_summary,
        "generation": {
            "conditions": [condition.__dict__ for condition in condition_plan],
            "strengths_sd": list(strengths),
            "intervention_scope": args.intervention_scope,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "device_map": args.device_map,
            "chat_template_kwargs": chat_template_kwargs or {},
            "resume": args.resume,
            "resume_existing_rows": len(existing_rows_by_key),
            "resume_malformed_rows": resume_malformed_rows,
            "resume_ignored_rows": resume_ignored_rows,
        },
        "summary": summarize_answer_rows(rows),
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
