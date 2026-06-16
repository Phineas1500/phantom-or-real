#!/usr/bin/env python3
"""Qwen HF multi-layer correctness-subspace erasure necessity test.

Qwen is not run through TransformerLens in this repo, so this mirrors
``stage2_subspace_erasure.py`` while using native Hugging Face hooks at
``model.model.layers[L]``. It mean-ablates the per-layer raw correctness probe
direction at every residual position present in each forward pass, including
prompt processing and decode-cache steps.
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
    hidden_from_output,
    input_device_for_model,
    load_hf_model,
    render_tokens,
    replace_hidden_in_output,
    torch_dtype,
    validate_hf_layers,
)
from scripts.stage2_subspace_erasure import (  # noqa: E402
    build_layer_directions,
    make_condition_plan,
    parse_condition_kinds,
    save_layer_direction_artifact,
    summarize_erasure_rows,
    summarize_hook_states,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.stage2_steering import parse_int_list, score_reply, select_balanced_stage1_rows  # noqa: E402


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def serializable_direction_summary(direction: dict[str, Any]) -> dict[str, Any]:
    skip = {"unit_direction", "raw_coef", "coef_std", "scaler_mean", "scaler_scale"}
    return {key: value for key, value in direction.items() if key not in skip}


def make_hf_erasure_hook(
    *,
    vector: np.ndarray,
    projection_mean: float,
    projection_std: float,
) -> tuple[Any, dict[str, Any]]:
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    state: dict[str, Any] = {"calls": 0, "positions": 0, "abs_delta_sd_sum": 0.0}
    denom = max(float(projection_std), 1e-8)

    def hook_fn(_module, _inputs, output):
        hidden = hidden_from_output(output)
        key = (str(hidden.device), hidden.dtype)
        unit = cache.get(key)
        if unit is None:
            unit = torch.as_tensor(vector, device=hidden.device, dtype=torch.float32)
            cache[key] = unit
        projection = hidden.float() @ unit
        delta = projection - float(projection_mean)
        patched = hidden - (delta.unsqueeze(-1) * unit).to(hidden.dtype)
        state["calls"] += 1
        state["positions"] += int(delta.numel())
        state["abs_delta_sd_sum"] += float(delta.abs().sum().item()) / denom
        return replace_hidden_in_output(output, patched)

    return hook_fn, state


def generate_one(
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
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    if stop_at_eos and tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        output_tokens = model.generate(**kwargs)
    output_ids = output_tokens[0].detach().cpu().tolist()
    new_ids = output_ids[len(token_ids) :]
    reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return new_ids, reply


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model-key", default="qwen35_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="16,31,40,45,53")
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=4)
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument(
        "--row-shard",
        default=None,
        help="Optional 'i/n' slice of selected rows so sampled runs fit the wall limit.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260621)
    parser.add_argument("--probe-seed", type=int, default=20260622)
    parser.add_argument("--orthogonal-seed", type=int, default=20260623)
    parser.add_argument("--gaussian-seed", type=int, default=20260624)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--conditions", default="baseline,erase_raw,erase_orthogonal,erase_gaussian")
    parser.add_argument("--max-new-tokens", type=int, default=96)
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
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/qwen35_subspace_erasure_27b_property.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/erasure/qwen35_subspace_erasure_27b_property_directions.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/qwen35_subspace_erasure_27b_property.json"))
    return parser


def main() -> int:
    load_env()
    args = build_arg_parser().parse_args()
    torch.set_grad_enabled(False)
    started = time.time()

    if args.samples_per_row < 1:
        raise ValueError("--samples-per-row must be >= 1")
    if args.samples_per_row > 1 and not args.do_sample:
        raise ValueError("--samples-per-row > 1 requires --do-sample")
    layers = parse_int_list(args.layers)
    condition_plan = make_condition_plan(parse_condition_kinds(args.conditions))
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)

    print("Qwen HF multi-layer subspace erasure", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layers={layers}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"samples_per_row={args.samples_per_row}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}", flush=True)

    selected_rows, selection_summary = select_balanced_stage1_rows(
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=parse_int_list(args.heights),
        per_height_label=args.per_height_label,
        seed=args.selection_seed,
        drop_parse_failed=True,
        target_split="test",
    )
    if args.row_shard:
        shard_index, shard_count = (int(part) for part in args.row_shard.split("/"))
        if not (0 <= shard_index < shard_count):
            raise ValueError(f"invalid --row-shard {args.row_shard!r}")
        selected_rows = selected_rows[shard_index::shard_count]
        selection_summary["row_shard"] = args.row_shard
        selection_summary["shard_rows"] = len(selected_rows)
    total_generations = len(selected_rows) * len(condition_plan) * args.samples_per_row
    print(
        f"selected_rows={len(selected_rows)} total_generations={total_generations} "
        f"selection={selection_summary}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selection": selection_summary,
                    "conditions": [condition.__dict__ for condition in condition_plan],
                    "layers": layers,
                    "total_generations": total_generations,
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            ),
            flush=True,
        )
        return 0

    by_layer = build_layer_directions(
        layers=layers,
        activation_dir=args.activation_dir,
        model_key=args.model_key,
        task=args.task,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        probe_seed=args.probe_seed,
        orthogonal_seed=args.orthogonal_seed,
        gaussian_seed=args.gaussian_seed,
        c_values=tuple(float(part) for part in args.c_values.split(",")),
        max_iter=args.max_iter,
        solver=args.solver,
    )
    save_layer_direction_artifact(args.direction_output, by_layer)

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(
        f"scorer_preflight: strong={scorer_preflight['is_correct_strong']} "
        f"parse_failed={scorer_preflight['parse_failed']}",
        flush=True,
    )

    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else None
    model, tokenizer = load_hf_model(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        device=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    validate_hf_layers(model, layers)
    print(f"using_hooks={{{', '.join(f'{layer}: model.model.layers.{layer}.output' for layer in layers)}}}", flush=True)
    print(f"chat_template_kwargs={chat_template_kwargs or {}}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            token_ids = render_tokens(
                tokenizer=tokenizer,
                row=stage1_row,
                model_name=args.model,
                chat_template_kwargs=chat_template_kwargs,
            )
            if len(token_ids) > args.n_ctx:
                raise ValueError(f"row {stage1_row['row_index']} exceeds n_ctx={args.n_ctx}: {len(token_ids)}")
            print(
                f"row {row_idx}/{len(selected_rows)} source_row={stage1_row['row_index']} "
                f"h={stage1_row.get('height')} original_strong={stage1_row.get('is_correct_strong')} "
                f"prompt_tokens={len(token_ids)}",
                flush=True,
            )
            for condition_index, condition in enumerate(condition_plan):
                for sample_index in range(args.samples_per_row):
                    if args.do_sample:
                        torch.manual_seed(
                            args.selection_seed
                            + int(stage1_row["row_index"]) * 10007
                            + condition_index * 101
                            + sample_index
                        )
                    hook_states: dict[int, dict[str, Any]] = {}
                    handles = []
                    try:
                        if condition.vector_kind is not None:
                            for layer in layers:
                                entry = by_layer[layer]
                                stats = entry["stats"][condition.vector_kind]
                                hook_fn, hook_state = make_hf_erasure_hook(
                                    vector=entry["vectors"][condition.vector_kind],
                                    projection_mean=stats["projection_mean"],
                                    projection_std=stats["projection_std"],
                                )
                                hook_states[layer] = hook_state
                                handles.append(model.model.layers[layer].register_forward_hook(hook_fn))
                        new_ids, reply = generate_one(
                            model=model,
                            token_ids=token_ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos,
                        )
                    finally:
                        for handle in handles:
                            handle.remove()
                    score = score_reply(stage1_row, reply)
                    output_row = {
                        "schema_version": 1,
                        "source_file": source_file,
                        "source_row_index": int(stage1_row["row_index"]),
                        "example_id": stage1_row.get("example_id"),
                        "task": stage1_row.get("task"),
                        "height": stage1_row.get("height"),
                        "model": args.model,
                        "original_model": stage1_row.get("model"),
                        "original_is_correct_strong": bool(stage1_row.get("is_correct_strong")),
                        "original_is_correct_weak": bool(stage1_row.get("is_correct_weak")),
                        "original_parse_failed": bool(stage1_row.get("parse_failed")),
                        "condition": condition.label,
                        "sample_index": sample_index,
                        "method": "qwen_hf_multi_layer_subspace_erasure",
                        "target_variable": "free_form_correctness",
                        "representation_type": "raw_direction",
                        "erasure_kind": condition.vector_kind,
                        "erasure_layers": layers,
                        "prompt_token_count": len(token_ids),
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        "hook_summary": summarize_hook_states(hook_states),
                        **score,
                    }
                    rows.append(output_row)
                    fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                    print(
                        f"  {condition.label}[{sample_index}]: strong={output_row['is_correct_strong']} "
                        f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                        f"new_tokens={len(new_ids)}",
                        flush=True,
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    erasure_summary = summarize_erasure_rows(rows)
    hook_names = {layer: f"model.model.layers.{layer}.output" for layer in layers}
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_qwen_subspace_erasure.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "free_form_correctness",
        "split": args.split_family,
        "site_or_layer": ",".join(f"L{layer}" for layer in layers),
        "method": "qwen_hf_multi_layer_subspace_erasure",
        "representation_type": "raw_direction",
        "layers": layers,
        "hook_names": hook_names,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "direction_output": str(args.direction_output),
        "out_jsonl": str(args.out_jsonl),
        "probe_directions": {
            f"L{layer}": serializable_direction_summary(entry["direction"])
            for layer, entry in by_layer.items()
        },
        "control_projection_stats": {
            f"L{layer}": {kind: stats for kind, stats in entry["stats"].items() if kind != "raw"}
            for layer, entry in by_layer.items()
        },
        "selection": selection_summary,
        "generation": {
            "conditions": [condition.__dict__ for condition in condition_plan],
            "samples_per_row": args.samples_per_row,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "device_map": args.device_map,
            "chat_template_kwargs": chat_template_kwargs or {},
        },
        "summary": erasure_summary,
        "baseline_metrics": erasure_summary["by_condition"].get("baseline"),
        "intervention_metrics": {
            condition: metrics
            for condition, metrics in erasure_summary["by_condition"].items()
            if condition != "baseline"
        },
        "paired_flips": erasure_summary["paired_vs_baseline"],
        "n": len(rows),
        "controls": [
            "regenerated_baseline",
            "orthogonal_direction",
            "matched_gaussian_noise",
            "qwen_hf_native_layer_hooks",
            "disable_thinking_chat_template",
        ],
        "causal_abstraction_claim": (
            "Qwen cross-model necessity test: mean-ablates the per-layer raw correctness "
            "direction at every available residual position during prompt processing and "
            "decode. A clean cross-model epiphenomenality result requires raw erasure to "
            "remain flat while orthogonal/Gaussian controls verify perturbation sensitivity."
        ),
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
