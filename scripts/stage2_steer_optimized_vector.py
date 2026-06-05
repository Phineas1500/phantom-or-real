#!/usr/bin/env python3
"""Train and test a one-shot optimized residual steering vector for Stage 2."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import (  # noqa: E402
    input_device_for_model,
    load_tl_model,
    render_chat_text,
    validate_hooks,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    load_probe_dataset,
    read_split_assignments,
    split_indices_from_assignments,
)
from src.stage2_steering import (  # noqa: E402
    make_condition_plan,
    make_gaussian_unit_direction,
    make_orthogonal_unit_direction,
    parse_condition_kinds,
    parse_float_list,
    parse_int_list,
    raw_probe_projection_sidecar,
    score_reply,
    select_balanced_stage1_rows,
    summarize_steering_rows,
)


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


def make_steering_hook(*, vector: np.ndarray, delta: float, scope: str) -> tuple[Any, dict[str, int]]:
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    state = {"calls": 0, "applications": 0}

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        apply = scope == "last_token_each_forward" or state["calls"] == 0
        state["calls"] += 1
        if not apply:
            return act
        key = (str(act.device), act.dtype)
        direction_tensor = cache.get(key)
        if direction_tensor is None:
            direction_tensor = torch.as_tensor(vector, device=act.device, dtype=act.dtype)
            cache[key] = direction_tensor
        act[:, -1, :] = act[:, -1, :] + float(delta) * direction_tensor
        state["applications"] += 1
        return act

    return hook_fn, state


def generate_one(
    *,
    model,
    token_ids: list[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    stop_at_eos: bool,
    cache_dtype: torch.dtype,
) -> tuple[list[int], str]:
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("model has no tokenizer")
    input_device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    previous_default_dtype = torch.get_default_dtype()
    with torch.inference_mode():
        try:
            torch.set_default_dtype(cache_dtype)
            output_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                stop_at_eos=stop_at_eos,
                do_sample=do_sample,
                temperature=temperature,
                prepend_bos=False,
                return_type="tokens",
                verbose=False,
                use_past_kv_cache=True,
            )
        finally:
            torch.set_default_dtype(previous_default_dtype)
    output_ids = output_tokens[0].detach().cpu().tolist()
    new_ids = output_ids[len(token_ids) :]
    reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return new_ids, reply


def hook_device_for_layer(model, layer: int) -> torch.device:
    block = model.blocks[layer]
    for parameter in block.parameters(recurse=True):
        return parameter.device
    return input_device_for_model(model)


def target_text_for_row(row: dict[str, Any]) -> str:
    target = str(row.get("ground_truth") or "").strip()
    if not target:
        raise ValueError(f"row {row.get('row_index')} has empty ground_truth")
    return target


def teacher_forcing_tokens(
    *,
    tokenizer,
    model_name: str,
    row: dict[str, Any],
    max_target_tokens: int,
    input_device: torch.device,
) -> dict[str, Any]:
    prompt_text = render_chat_text(
        tokenizer,
        system=row["system_prompt"],
        user=row["prompt_text"],
        model_name=model_name,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text_for_row(row), add_special_tokens=False)["input_ids"][:max_target_tokens]
    if not target_ids:
        raise ValueError(f"row {row.get('row_index')} has empty target tokenization")
    tokens = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=input_device)
    return {
        "row_index": int(row["row_index"]),
        "height": row.get("height"),
        "original_is_correct_strong": bool(row.get("is_correct_strong")),
        "prompt_len": len(prompt_ids),
        "target_len": len(target_ids),
        "target_text": target_text_for_row(row),
        "tokens": tokens,
    }


def loss_with_vector(
    *,
    model,
    hook_name: str,
    vector_param: torch.nn.Parameter,
    example: dict[str, Any],
) -> torch.Tensor:
    prompt_len = int(example["prompt_len"])
    target_len = int(example["target_len"])
    start = max(prompt_len - 1, 0)
    end = prompt_len + target_len - 1

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        local_end = min(end, act.shape[1])
        if start >= local_end:
            return act
        vector = vector_param.to(device=act.device, dtype=act.dtype)
        delta = torch.zeros_like(act)
        delta[:, start:local_end, :] = vector
        return act + delta

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        logits = model(example["tokens"], return_type="logits", prepend_bos=False)
    pred = logits[0, start:end, :].float()
    target = example["tokens"][0, prompt_len : prompt_len + target_len].to(pred.device)
    return F.cross_entropy(pred, target)


def optimize_gold_continuation_vector(
    *,
    model,
    hook_name: str,
    layer: int,
    train_rows: list[dict[str, Any]],
    model_name: str,
    max_target_tokens: int,
    steps: int,
    lr: float,
    weight_decay: float,
    max_vector_norm: float,
    seed: int,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("model has no tokenizer")
    input_device = input_device_for_model(model)
    vector_device = hook_device_for_layer(model, layer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vector = torch.nn.Parameter(torch.zeros(model.cfg.d_model, dtype=torch.float32, device=vector_device))
    optimizer = torch.optim.AdamW([vector], lr=lr, weight_decay=weight_decay)
    examples = [
        teacher_forcing_tokens(
            tokenizer=tokenizer,
            model_name=model_name,
            row=row,
            max_target_tokens=max_target_tokens,
            input_device=input_device,
        )
        for row in train_rows
    ]
    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    previous_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            order = list(range(len(examples)))
            rng.shuffle(order)
            losses: list[float] = []
            for idx in order:
                loss = loss_with_vector(
                    model=model,
                    hook_name=hook_name,
                    vector_param=vector,
                    example=examples[idx],
                )
                (loss / len(order)).backward()
                losses.append(float(loss.detach().cpu().item()))
                del loss
            grad_norm = float(vector.grad.detach().float().norm().cpu().item()) if vector.grad is not None else None
            optimizer.step()
            with torch.no_grad():
                norm = vector.detach().float().norm()
                if max_vector_norm > 0 and norm > max_vector_norm:
                    vector.mul_(float(max_vector_norm) / float(norm.cpu().item()))
            record = {
                "step": step,
                "mean_loss": float(sum(losses) / len(losses)),
                "min_loss": float(min(losses)),
                "max_loss": float(max(losses)),
                "grad_norm": grad_norm,
                "vector_norm": float(vector.detach().float().norm().cpu().item()),
            }
            history.append(record)
            print(
                f"opt step {step}/{steps} mean_loss={record['mean_loss']:.4f} "
                f"grad_norm={record['grad_norm']} vector_norm={record['vector_norm']:.4f}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        torch.set_grad_enabled(previous_grad_state)
    optimized = vector.detach().float().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(optimized))
    if norm == 0.0:
        raise ValueError("optimized vector has zero norm")
    unit = (optimized / norm).astype(np.float32)
    return {
        "unit_direction": unit,
        "raw_optimized_vector": optimized,
        "raw_optimized_norm": norm,
        "loss_history": history,
        "training_examples": [
            {
                "source_row_index": example["row_index"],
                "height": example["height"],
                "original_is_correct_strong": example["original_is_correct_strong"],
                "prompt_len": example["prompt_len"],
                "target_len": example["target_len"],
                "target_text": example["target_text"],
            }
            for example in examples
        ],
        "objective": "gold_continuation_teacher_forced_ce",
        "max_target_tokens": max_target_tokens,
        "optimizer": "AdamW",
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_vector_norm": max_vector_norm,
    }


def add_projection_stats(
    *,
    direction: dict[str, Any],
    activation_path: Path,
    sidecar_path: Path,
    splits_path: Path,
    source_file: str,
    split_family: str,
    projection_split: str,
) -> dict[str, Any]:
    dataset = load_probe_dataset(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        drop_parse_failed=True,
    )
    splits = split_indices_from_assignments(
        dataset["sidecar"],
        assignments=read_split_assignments(splits_path),
        source_file=source_file,
        split_field=f"{split_family}_split",
    )
    indices = splits[projection_split]
    unit = np.asarray(direction["unit_direction"], dtype=np.float32)
    projections = np.asarray(dataset["x"][indices] @ unit, dtype=np.float64)
    std = float(projections.std(ddof=0))
    if std == 0.0:
        raise ValueError("optimized direction projection has zero train split standard deviation")
    direction.update(
        {
            "input_rows": dataset["input_rows"],
            "kept_rows": dataset["kept_rows"],
            "d_model": dataset["d_model"],
            "projection_split": projection_split,
            "train_projection_mean": float(projections.mean()),
            "train_projection_std": std,
            "train_projection_min": float(projections.min()),
            "train_projection_max": float(projections.max()),
        }
    )
    return direction


def save_direction_artifact(
    *,
    path: Path,
    direction: dict[str, Any],
    orthogonal_direction: np.ndarray,
    gaussian_direction: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_direction=direction["unit_direction"],
        raw_optimized_vector=direction["raw_optimized_vector"],
        orthogonal_direction=orthogonal_direction.astype(np.float32),
        gaussian_direction=gaussian_direction.astype(np.float32),
        train_projection_std=np.array(direction["train_projection_std"], dtype=np.float32),
        train_projection_mean=np.array(direction["train_projection_mean"], dtype=np.float32),
    )


def direction_summary(direction: dict[str, Any]) -> dict[str, Any]:
    skip = {"unit_direction", "raw_optimized_vector"}
    return {key: value for key, value in direction.items() if key not in skip}


def prefix_group_summary(flips: dict[str, Any], prefix: str) -> dict[str, Any]:
    rows = [value for label, value in flips.items() if label.startswith(prefix) and isinstance(value, dict)]
    if not rows:
        return {"conditions": [], "max_false_to_true": None, "max_changed": None, "max_true_to_false": None}
    return {
        "conditions": sorted(label for label in flips if label.startswith(prefix)),
        "max_false_to_true": max(int(row.get("false_to_true", 0)) for row in rows),
        "max_changed": max(int(row.get("changed", 0)) for row in rows),
        "max_true_to_false": max(int(row.get("true_to_false", 0)) for row in rows),
        "max_abs_accuracy_delta": max(abs(float(row.get("net_accuracy_delta", 0.0))) for row in rows),
    }


def matched_noise_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flips = summary.get("flips_vs_baseline", {})
    optimized = prefix_group_summary(flips, "optimized_")
    orthogonal = prefix_group_summary(flips, "orthogonal_")
    gaussian = prefix_group_summary(flips, "gaussian_")
    controls = [value for value in [orthogonal["max_false_to_true"], gaussian["max_false_to_true"]] if value is not None]
    control_changed = [value for value in [orthogonal["max_changed"], gaussian["max_changed"]] if value is not None]
    control_max_false_to_true = max(controls) if controls else None
    control_max_changed = max(control_changed) if control_changed else None
    return {
        "status": "available_no_sigma_estimate" if gaussian["conditions"] else "missing_gaussian_control",
        "has_matched_gaussian_control": bool(gaussian["conditions"]),
        "optimized_max_false_to_true": optimized["max_false_to_true"],
        "optimized_max_changed": optimized["max_changed"],
        "optimized_max_true_to_false": optimized["max_true_to_false"],
        "orthogonal_max_false_to_true": orthogonal["max_false_to_true"],
        "orthogonal_max_changed": orthogonal["max_changed"],
        "gaussian_max_false_to_true": gaussian["max_false_to_true"],
        "gaussian_max_changed": gaussian["max_changed"],
        "control_max_false_to_true": control_max_false_to_true,
        "control_max_changed": control_max_changed,
        "repair_exceeds_controls_by_count": (
            optimized["max_false_to_true"] is not None
            and control_max_false_to_true is not None
            and optimized["max_false_to_true"] > control_max_false_to_true
        ),
        "sigma_test_status": "not_estimated_from_single_matched_control_family",
    }


def positive_control_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_configured"}
    if not path.exists():
        return {"status": "missing", "report_path": str(path)}
    with path.open() as f:
        report = json.load(f)
    summary = report.get("summary", {}).get("matched_noise_summary", {})
    return {
        "status": "passed" if summary.get("passed_positive_control_gate") is True else "failed_or_unclear",
        "report_path": str(path),
        "target_variable": report.get("target_variable"),
        "passed_positive_control_gate": summary.get("passed_positive_control_gate"),
        "best_effect_over_control_sigma": summary.get("best_toward_upper", {}).get("effect_over_control_sigma"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-prefix", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--opt-train-per-height-label", type=int, default=1)
    parser.add_argument("--eval-per-height-label", type=int, default=2)
    parser.add_argument("--train-selection-seed", type=int, default=20260605)
    parser.add_argument("--eval-selection-seed", type=int, default=20260427)
    parser.add_argument("--orthogonal-seed", type=int, default=20260615)
    parser.add_argument("--gaussian-seed", type=int, default=20260616)
    parser.add_argument("--opt-steps", type=int, default=8)
    parser.add_argument("--opt-lr", type=float, default=0.25)
    parser.add_argument("--opt-weight-decay", type=float, default=0.0)
    parser.add_argument("--opt-max-vector-norm", type=float, default=10.0)
    parser.add_argument("--target-max-tokens", type=int, default=16)
    parser.add_argument("--conditions", default="baseline,optimized,orthogonal,gaussian")
    parser.add_argument("--strengths", default="-1,1")
    parser.add_argument(
        "--intervention-scope",
        choices=("prompt_only", "last_token_each_forward"),
        default="last_token_each_forward",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--positive-control-report", type=Path, default=Path("docs/positive_control_format_gemma3_27b_l45.json"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/steering/optimized_gold_l45_property_decode_sweep.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/steering/optimized_gold_l45_property_decode_sweep_direction.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/optimized_gold_steering_27b_l45_property_decode_sweep.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()

    heights = parse_int_list(args.heights)
    strengths = parse_float_list(args.strengths)
    condition_kinds = parse_condition_kinds(args.conditions)
    condition_plan = make_condition_plan(condition_kinds=condition_kinds, strengths=strengths)
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)
    activation_prefix = (
        args.activation_prefix
        if args.activation_prefix is not None
        else args.activation_dir / f"{args.model_key}_{args.task}_L{args.layer}"
    )
    activation_path = activation_prefix.with_suffix(".safetensors")
    sidecar_path = activation_prefix.with_suffix(".example_ids.jsonl")

    print("Stage 2 optimized-vector steering pilot", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    started = time.time()
    train_rows, train_selection = select_balanced_stage1_rows(
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=heights,
        per_height_label=args.opt_train_per_height_label,
        seed=args.train_selection_seed,
        drop_parse_failed=True,
        target_split=args.train_split,
    )
    eval_rows, eval_selection = select_balanced_stage1_rows(
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=heights,
        per_height_label=args.eval_per_height_label,
        seed=args.eval_selection_seed,
        drop_parse_failed=True,
        target_split=args.eval_split,
    )
    print(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}", flush=True)

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(eval_rows[0], eval_rows[0]["ground_truth"])
    print(
        "scorer_preflight: "
        f"strong={scorer_preflight['is_correct_strong']} "
        f"parse_failed={scorer_preflight['parse_failed']}",
        flush=True,
    )

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    hook_name = validate_hooks(model, [args.layer])[0]
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    print(f"using_hook={hook_name}", flush=True)

    direction = optimize_gold_continuation_vector(
        model=model,
        hook_name=hook_name,
        layer=args.layer,
        train_rows=train_rows,
        model_name=args.model,
        max_target_tokens=args.target_max_tokens,
        steps=args.opt_steps,
        lr=args.opt_lr,
        weight_decay=args.opt_weight_decay,
        max_vector_norm=args.opt_max_vector_norm,
        seed=args.train_selection_seed,
    )
    direction = add_projection_stats(
        direction=direction,
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        projection_split=args.train_split,
    )
    orthogonal_direction = make_orthogonal_unit_direction(direction["unit_direction"], seed=args.orthogonal_seed)
    gaussian_direction = make_gaussian_unit_direction(direction["unit_direction"], seed=args.gaussian_seed)
    save_direction_artifact(
        path=args.direction_output,
        direction=direction,
        orthogonal_direction=orthogonal_direction,
        gaussian_direction=gaussian_direction,
    )
    print(
        "optimized_direction: "
        f"objective={direction['objective']} final_loss={direction['loss_history'][-1]['mean_loss']:.4f} "
        f"raw_norm={direction['raw_optimized_norm']:.4f} proj_std={direction['train_projection_std']:.4f}",
        flush=True,
    )

    projection_sidecar = raw_probe_projection_sidecar(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        direction=direction,
        higher_is="more_along_optimized_gold_continuation_direction",
    )
    projection_by_row_index = projection_sidecar["by_row_index"]
    missing_projection_rows = [
        int(row["row_index"])
        for row in eval_rows
        if int(row["row_index"]) not in projection_by_row_index
    ]
    print(
        "optimized_projection_sidecar: "
        f"covered={len(eval_rows) - len(missing_projection_rows)}/{len(eval_rows)} "
        f"missing={missing_projection_rows[:5]}",
        flush=True,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    vector_by_kind = {
        "optimized": direction["unit_direction"],
        "orthogonal": orthogonal_direction,
        "gaussian": gaussian_direction,
    }
    projection_std = float(direction["train_projection_std"])
    torch.set_grad_enabled(False)
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(eval_rows, start=1):
            prompt_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            print(
                f"row {row_idx}/{len(eval_rows)} "
                f"source_row={stage1_row['row_index']} h={stage1_row['height']} "
                f"original_correct={stage1_row['is_correct_strong']} prompt_tokens={len(token_ids)}",
                flush=True,
            )
            projection_fields = projection_by_row_index.get(int(stage1_row["row_index"]), {})
            for condition in condition_plan:
                hook_state = {"calls": 0, "applications": 0}
                if condition.direction_kind is None:
                    new_ids, reply = generate_one(
                        model=model,
                        token_ids=token_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        stop_at_eos=args.stop_at_eos,
                        cache_dtype=dtype,
                    )
                else:
                    delta = condition.strength_sd * projection_std
                    hook_fn, hook_state = make_steering_hook(
                        vector=vector_by_kind[condition.direction_kind],
                        delta=delta,
                        scope=args.intervention_scope,
                    )
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        new_ids, reply = generate_one(
                            model=model,
                            token_ids=token_ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype,
                        )
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
                    "direction_kind": condition.direction_kind,
                    "strength_sd": condition.strength_sd,
                    "intervention_delta_l2": abs(condition.strength_sd * projection_std),
                    "intervention_scope": args.intervention_scope,
                    "direction_projection": projection_fields.get("direction_projection"),
                    "direction_projection_z": projection_fields.get("direction_projection_z"),
                    "direction_projection_higher_is": projection_fields.get("direction_projection_higher_is"),
                    "hook_calls": int(hook_state["calls"]),
                    "hook_applications": int(hook_state["applications"]),
                    "prompt_token_count": len(token_ids),
                    "generated_token_count": len(new_ids),
                    "model_output": reply,
                    **score,
                }
                rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"  {condition.label}: strong={output_row['is_correct_strong']} "
                    f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                    f"new_tokens={len(new_ids)} hooks={hook_state['applications']}/{hook_state['calls']}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    steering_summary = summarize_steering_rows(rows)
    noise_summary = matched_noise_summary(steering_summary)
    positive_control = positive_control_status(args.positive_control_report)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_steer_optimized_vector.py",
        "method": "optimized_gold_continuation_vector",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "free_form_correctness",
        "layer": args.layer,
        "hook_name": hook_name,
        "activation_path": str(activation_path),
        "sidecar_path": str(sidecar_path),
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "direction_output": str(args.direction_output),
        "out_jsonl": str(args.out_jsonl),
        "optimized_direction": direction_summary(direction),
        "train_selection": train_selection,
        "selection": eval_selection,
        "optimized_projection_sidecar": projection_sidecar["summary"],
        "positive_control_gate": positive_control,
        "controls": ["regenerated_baseline", "orthogonal_direction", "matched_gaussian_noise", "positive_control"],
        "generation": {
            "conditions": [condition.__dict__ for condition in condition_plan],
            "strengths_sd": list(strengths),
            "intervention_scope": args.intervention_scope,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "matched_noise_summary": noise_summary,
        "summary": steering_summary,
        "causal_abstraction_claim": (
            "Optimized gold-continuation vector test: teacher-forced gold-answer objective trains a single L45 residual direction; "
            "free-form repair must exceed orthogonal/Gaussian controls before any causal-repair claim."
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
