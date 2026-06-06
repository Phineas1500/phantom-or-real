#!/usr/bin/env python3
"""Trace raw-probe projections during baseline decoding without intervention."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import (  # noqa: E402
    input_device_for_model,
    load_tl_model,
    render_chat_text,
    validate_hooks,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    parse_float_list,
    parse_int_list,
    raw_probe_projection_sidecar,
    score_reply,
    select_balanced_stage1_rows,
    train_raw_probe_direction,
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


def serializable_direction_summary(direction: dict[str, Any]) -> dict[str, Any]:
    skip = {"unit_direction", "raw_coef", "coef_std", "scaler_mean", "scaler_scale"}
    return {key: value for key, value in direction.items() if key not in skip}


def save_direction_artifact(*, path: Path, directions: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for layer, direction in directions.items():
        prefix = f"L{layer}"
        payload[f"{prefix}_unit_direction"] = direction["unit_direction"]
        payload[f"{prefix}_raw_coef"] = direction["raw_coef"]
        payload[f"{prefix}_coef_std"] = direction["coef_std"]
        payload[f"{prefix}_scaler_mean"] = direction["scaler_mean"]
        payload[f"{prefix}_scaler_scale"] = direction["scaler_scale"]
        payload[f"{prefix}_train_projection_std"] = np.array(direction["train_projection_std"], dtype=np.float32)
        payload[f"{prefix}_train_projection_mean"] = np.array(direction["train_projection_mean"], dtype=np.float32)
        payload[f"{prefix}_best_c"] = np.array(direction["best_c"], dtype=np.float32)
    np.savez_compressed(path, **payload)


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


def load_jsonl_with_indices(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["row_index"] = idx
            rows[idx] = row
    return rows


def select_manifest_stage1_rows(
    *,
    jsonl_path: Path,
    manifest_path: Path,
    model_key: str,
    task: str,
    require_no_decode_trace: bool,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_file = str(jsonl_path)
    source_rows = load_jsonl_with_indices(jsonl_path)
    with manifest_path.open() as f:
        manifest = json.load(f)

    selected: list[dict[str, Any]] = []
    skipped: dict[str, int] = {
        "model_task": 0,
        "source_file": 0,
        "not_recognition_gap": 0,
        "already_decode_traced": 0,
        "missing_source_row": 0,
    }
    for manifest_row in manifest.get("rows", []):
        if manifest_row.get("model") != model_key or manifest_row.get("task") != task:
            skipped["model_task"] += 1
            continue
        if manifest_row.get("source_file") != source_file:
            skipped["source_file"] += 1
            continue
        if "recognition_generation_gap" not in manifest_row.get("recommended_use", []):
            skipped["not_recognition_gap"] += 1
            continue
        if require_no_decode_trace and manifest_row.get("coverage_flags", {}).get("has_decode_trace"):
            skipped["already_decode_traced"] += 1
            continue
        row_index = int(manifest_row["source_row_index"])
        source_row = source_rows.get(row_index)
        if source_row is None:
            skipped["missing_source_row"] += 1
            continue
        row = dict(source_row)
        row["manifest_entry"] = manifest_row
        selected.append(row)

    selected.sort(key=lambda row: (int(row.get("height", -1)), int(row["row_index"])))
    if limit > 0:
        selected = selected[:limit]

    counts: dict[str, int] = {}
    for row in selected:
        height = row.get("height")
        key = f"h{height}"
        counts[key] = counts.get(key, 0) + 1

    return selected, {
        "source_file": source_file,
        "selection_mode": "manifest_recognition_gap",
        "manifest": str(manifest_path),
        "model_key": model_key,
        "task": task,
        "require_no_decode_trace": require_no_decode_trace,
        "limit": limit,
        "selected_rows": len(selected),
        "selected_counts": counts,
        "skipped_counts": skipped,
        "manifest_report_n": manifest.get("n"),
    }


def score_candidate_logprob(
    *,
    model,
    prompt_token_ids: list[int],
    candidate_text: str,
) -> dict[str, Any]:
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
    return {
        "text": candidate_text,
        "token_count": len(candidate_ids),
        "logprob": total,
        "mean_logprob": total / len(candidate_ids),
    }


def score_prompt_margins(
    *,
    model,
    prompt_token_ids: list[int],
    gold_hypothesis: str | None,
    foil_hypothesis: str | None,
    parsed_hypotheses: list[Any],
) -> dict[str, Any]:
    selected_hypothesis = str(parsed_hypotheses[0]) if parsed_hypotheses else None
    if not gold_hypothesis or not foil_hypothesis:
        return {
            "available": False,
            "selected_hypothesis": selected_hypothesis,
            "selected_hypothesis_count": len(parsed_hypotheses),
        }

    gold = score_candidate_logprob(
        model=model,
        prompt_token_ids=prompt_token_ids,
        candidate_text=gold_hypothesis,
    )
    foil = score_candidate_logprob(
        model=model,
        prompt_token_ids=prompt_token_ids,
        candidate_text=foil_hypothesis,
    )
    selected = None
    if selected_hypothesis:
        selected = score_candidate_logprob(
            model=model,
            prompt_token_ids=prompt_token_ids,
            candidate_text=selected_hypothesis,
        )

    out = {
        "available": True,
        "gold_hypothesis": gold_hypothesis,
        "foil_hypothesis": foil_hypothesis,
        "selected_hypothesis": selected_hypothesis,
        "selected_hypothesis_count": len(parsed_hypotheses),
        "gold": gold,
        "foil": foil,
        "gold_vs_foil_logprob_margin": gold["logprob"] - foil["logprob"],
        "gold_vs_foil_mean_logprob_margin": gold["mean_logprob"] - foil["mean_logprob"],
    }
    if selected is not None:
        out["selected"] = selected
        out["selected_vs_gold_logprob_margin"] = selected["logprob"] - gold["logprob"]
        out["selected_vs_foil_logprob_margin"] = selected["logprob"] - foil["logprob"]
        out["gold_vs_selected_logprob_margin"] = gold["logprob"] - selected["logprob"]
    return out


def make_projection_trace_hook(
    *,
    layer: int,
    direction: dict[str, Any],
    trace_state: dict[int, list[dict[str, Any]]],
) -> Any:
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    unit = np.asarray(direction["unit_direction"], dtype=np.float32)
    mean = float(direction["train_projection_mean"])
    std = float(direction["train_projection_std"])
    if std == 0.0:
        raise ValueError(f"L{layer} train projection std is zero")

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        key = (str(act.device), act.dtype)
        direction_tensor = cache.get(key)
        if direction_tensor is None:
            direction_tensor = torch.as_tensor(unit, device=act.device, dtype=act.dtype)
            cache[key] = direction_tensor
        last = act[:, -1, :]
        projection = torch.sum(last.float() * direction_tensor.float(), dim=-1)
        z = (projection - mean) / std
        call_index = len(trace_state[layer])
        trace_state[layer].append(
            {
                "call_index": call_index,
                "phase": "prefill" if call_index == 0 else "decode",
                "seq_len": int(act.shape[1]),
                "projection": float(projection.detach().float().cpu()[0].item()),
                "projection_z": float(z.detach().float().cpu()[0].item()),
            }
        )
        return act

    return hook_fn


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "fraction_below_0": None,
            "fraction_below_neg1": None,
            "fraction_below_neg2": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
        "fraction_below_0": float(np.mean(arr < 0.0)),
        "fraction_below_neg1": float(np.mean(arr < -1.0)),
        "fraction_below_neg2": float(np.mean(arr < -2.0)),
    }


def summarize_trace_rows(rows: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for layer in layers:
        layer_key = f"L{layer}"
        layer_rows = []
        for row in rows:
            trace = row["projection_traces"].get(layer_key, [])
            prefill = [point["projection_z"] for point in trace if point["phase"] == "prefill"]
            decode = [point["projection_z"] for point in trace if point["phase"] == "decode"]
            layer_rows.append(
                {
                    "baseline_correct": bool(row["is_correct_strong"]),
                    "original_correct": bool(row["original_is_correct_strong"]),
                    "height": row.get("height"),
                    "prefill": prefill,
                    "decode": decode,
                }
            )
        by_baseline_correct = {}
        for label in (False, True):
            subset = [row for row in layer_rows if row["baseline_correct"] is label]
            decode_values = [value for row in subset for value in row["decode"]]
            prefill_values = [value for row in subset for value in row["prefill"]]
            first_decode = [row["decode"][0] for row in subset if row["decode"]]
            last_decode = [row["decode"][-1] for row in subset if row["decode"]]
            by_baseline_correct[str(label).lower()] = {
                "rows": len(subset),
                "prefill_z": summarize_values(prefill_values),
                "decode_z": summarize_values(decode_values),
                "first_decode_z": summarize_values(first_decode),
                "last_decode_z": summarize_values(last_decode),
            }
        by_height = {}
        for height in sorted({row.get("height") for row in rows}):
            subset = [row for row in layer_rows if row["height"] == height]
            by_height[f"h{height}"] = {
                "rows": len(subset),
                "decode_z": summarize_values([value for row in subset for value in row["decode"]]),
            }
        summary[layer_key] = {
            "rows": len(layer_rows),
            "baseline_strong_accuracy": (
                sum(row["baseline_correct"] for row in layer_rows) / len(layer_rows)
                if layer_rows
                else None
            ),
            "all_prefill_z": summarize_values([value for row in layer_rows for value in row["prefill"]]),
            "all_decode_z": summarize_values([value for row in layer_rows for value in row["decode"]]),
            "by_baseline_is_correct_strong": by_baseline_correct,
            "by_height": by_height,
        }
    return summary


def summarize_prompt_margins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("prompt_margin_scores", {}).get("available")]

    def collect(key: str, subset: list[dict[str, Any]]) -> list[float]:
        values = []
        for row in subset:
            value = row.get("prompt_margin_scores", {}).get(key)
            if value is not None:
                values.append(float(value))
        return values

    by_generated_correct = {}
    for label in (False, True):
        subset = [row for row in available if bool(row.get("is_correct_strong")) is label]
        by_generated_correct[str(label).lower()] = {
            "rows": len(subset),
            "gold_vs_foil_logprob_margin": summarize_values(collect("gold_vs_foil_logprob_margin", subset)),
            "selected_vs_gold_logprob_margin": summarize_values(collect("selected_vs_gold_logprob_margin", subset)),
            "selected_vs_foil_logprob_margin": summarize_values(collect("selected_vs_foil_logprob_margin", subset)),
        }

    return {
        "rows": len(rows),
        "available_rows": len(available),
        "gold_vs_foil_logprob_margin": summarize_values(collect("gold_vs_foil_logprob_margin", available)),
        "selected_vs_gold_logprob_margin": summarize_values(collect("selected_vs_gold_logprob_margin", available)),
        "selected_vs_foil_logprob_margin": summarize_values(collect("selected_vs_foil_logprob_margin", available)),
        "by_generated_is_correct_strong": by_generated_correct,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="45,53")
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
    parser.add_argument("--selection-mode", choices=("balanced", "manifest_recognition_gap"), default="balanced")
    parser.add_argument("--manifest", type=Path, default=Path("docs/commitment_rowset_manifest.json"))
    parser.add_argument("--manifest-limit", type=int, default=0)
    parser.add_argument("--manifest-require-no-decode-trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selection-seed", type=int, default=20260427)
    parser.add_argument("--probe-seed", type=int, default=20260472)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_pilot.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_direction.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/decode_projection_trace_27b_l45_l53_property_pilot.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)

    layers = parse_int_list(args.layers)
    heights = parse_int_list(args.heights)
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)

    print("Stage 2 decode projection trace pilot", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layers={layers}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    started = time.time()
    directions: dict[int, dict[str, Any]] = {}
    projection_sidecars: dict[int, dict[str, Any]] = {}
    for layer in layers:
        activation_prefix = args.activation_dir / f"{args.model_key}_{args.task}_L{layer}"
        activation_path = activation_prefix.with_suffix(".safetensors")
        sidecar_path = activation_prefix.with_suffix(".example_ids.jsonl")
        if not activation_path.exists() or not sidecar_path.exists():
            raise FileNotFoundError(
                f"missing activation artifact for L{layer}: {activation_path} / {sidecar_path}"
            )
        direction = train_raw_probe_direction(
            activation_path=activation_path,
            sidecar_path=sidecar_path,
            splits_path=args.splits,
            source_file=source_file,
            split_family=args.split_family,
            seed=args.probe_seed,
            c_values=parse_float_list(args.c_values),
            max_iter=args.max_iter,
            solver=args.solver,
        )
        directions[layer] = direction
        projection_sidecars[layer] = raw_probe_projection_sidecar(
            activation_path=activation_path,
            sidecar_path=sidecar_path,
            direction=direction,
        )
        print(
            f"L{layer} direction: best_c={direction['best_c']} "
            f"val_auc={direction['val_auc']:.4f} test_auc={direction['test_auc']:.4f} "
            f"proj_std={direction['train_projection_std']:.4f}",
            flush=True,
        )
    save_direction_artifact(path=args.direction_output, directions=directions)

    if args.selection_mode == "balanced":
        selected_rows, selection_summary = select_balanced_stage1_rows(
            jsonl_path=args.jsonl,
            splits_path=args.splits,
            source_file=source_file,
            split_family=args.split_family,
            heights=heights,
            per_height_label=args.per_height_label,
            seed=args.selection_seed,
            drop_parse_failed=True,
        )
    else:
        selected_rows, selection_summary = select_manifest_stage1_rows(
            jsonl_path=args.jsonl,
            manifest_path=args.manifest,
            model_key=args.model_key,
            task=args.task,
            require_no_decode_trace=args.manifest_require_no_decode_trace,
            limit=args.manifest_limit,
        )
    if not selected_rows:
        raise ValueError(f"selection produced no rows: {selection_summary}")
    print(
        f"selected_rows={len(selected_rows)} "
        f"selection_mode={selection_summary.get('selection_mode', 'balanced')} "
        f"selected_counts={selection_summary.get('selected_counts')} "
        f"available_counts={selection_summary.get('available_counts')}",
        flush=True,
    )
    for layer in layers:
        by_row_index = projection_sidecars[layer]["by_row_index"]
        missing = [int(row["row_index"]) for row in selected_rows if int(row["row_index"]) not in by_row_index]
        print(
            f"L{layer} projection_sidecar: covered={len(selected_rows) - len(missing)}/{len(selected_rows)} "
            f"missing={missing[:5]}",
            flush=True,
        )

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
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
    hook_names = validate_hooks(model, layers)
    hook_by_layer = dict(zip(layers, hook_names, strict=True))
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    print(f"using_hooks={hook_by_layer}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            prompt_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            trace_state: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
            fwd_hooks = [
                (
                    hook_by_layer[layer],
                    make_projection_trace_hook(
                        layer=layer,
                        direction=directions[layer],
                        trace_state=trace_state,
                    ),
                )
                for layer in layers
            ]
            print(
                f"row {row_idx}/{len(selected_rows)} "
                f"source_row={stage1_row['row_index']} h={stage1_row['height']} "
                f"original_correct={stage1_row['is_correct_strong']} prompt_tokens={len(token_ids)}",
                flush=True,
            )
            with model.hooks(fwd_hooks=fwd_hooks):
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
            manifest_entry = stage1_row.get("manifest_entry")
            recognition = manifest_entry.get("recognition") if manifest_entry else None
            margin_scores = score_prompt_margins(
                model=model,
                prompt_token_ids=token_ids,
                gold_hypothesis=recognition.get("gold_hypothesis") if recognition else None,
                foil_hypothesis=recognition.get("foil_hypothesis") if recognition else None,
                parsed_hypotheses=score.get("parsed_hypotheses") or [],
            )
            projection_fields = {
                f"L{layer}": projection_sidecars[layer]["by_row_index"].get(int(stage1_row["row_index"]), {})
                for layer in layers
            }
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
                "condition": "baseline_trace",
                "method": "baseline_decode_projection_trace",
                "target_variable": "commitment_state",
                "representation_type": "raw_direction",
                "selection_mode": args.selection_mode,
                "manifest_id": manifest_entry.get("manifest_id") if manifest_entry else None,
                "manifest_recommended_use": manifest_entry.get("recommended_use") if manifest_entry else [],
                "recognition": recognition,
                "gold_hypothesis": recognition.get("gold_hypothesis") if recognition else None,
                "foil_hypothesis": recognition.get("foil_hypothesis") if recognition else None,
                "layers": layers,
                "hook_names": hook_by_layer,
                "prompt_token_count": len(token_ids),
                "generated_token_count": len(new_ids),
                "model_output": reply,
                "prompt_projection_sidecar": projection_fields,
                "projection_traces": {f"L{layer}": trace_state[layer] for layer in layers},
                "prompt_margin_scores": margin_scores,
                **score,
            }
            rows.append(output_row)
            fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            fout.flush()
            layer_trace_counts = {f"L{layer}": len(trace_state[layer]) for layer in layers}
            print(
                f"  baseline_trace: strong={output_row['is_correct_strong']} "
                f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                f"new_tokens={len(new_ids)} traces={layer_trace_counts}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    trace_summary = summarize_trace_rows(rows, layers)
    parse_fail_rate = sum(bool(row["parse_failed"]) for row in rows) / len(rows) if rows else None
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_decode_projection_trace.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "commitment_state",
        "split": args.split_family,
        "site_or_layer": ",".join(f"L{layer}:{hook_by_layer[layer]}" for layer in layers),
        "method": "baseline_decode_projection_trace",
        "representation_type": "raw_direction",
        "layers": layers,
        "hook_names": hook_by_layer,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "selection_mode": args.selection_mode,
        "manifest": str(args.manifest) if args.selection_mode != "balanced" else None,
        "direction_output": str(args.direction_output),
        "out_jsonl": str(args.out_jsonl),
        "probe_directions": {f"L{layer}": serializable_direction_summary(directions[layer]) for layer in layers},
        "selection": selection_summary,
        "probe_projection_sidecars": {f"L{layer}": projection_sidecars[layer]["summary"] for layer in layers},
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "n": len(rows),
        "baseline_metrics": {
            "n": len(rows),
            "strong_accuracy": sum(bool(row["is_correct_strong"]) for row in rows) / len(rows) if rows else None,
            "weak_accuracy": sum(bool(row["is_correct_weak"]) for row in rows) / len(rows) if rows else None,
            "parse_fail_rate": parse_fail_rate,
        },
        "intervention_metrics": {},
        "paired_flips": {},
        "parse_fail_rate": {"baseline_trace": parse_fail_rate},
        "matched_noise_summary": {},
        "controls": ["regenerated_baseline"],
        "causal_abstraction_claim": (
            "Predictive calibration diagnostic only: traces whether raw correctness "
            "directions and prompt-continuation margins separate regenerated correct "
            "and incorrect decode trajectories. No causal repair claim is made."
        ),
        "summary": trace_summary,
        "prompt_margin_summary": summarize_prompt_margins(rows),
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
