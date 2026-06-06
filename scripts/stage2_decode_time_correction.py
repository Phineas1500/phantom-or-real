#!/usr/bin/env python3
"""Run a small conditional decode-time correction pilot for Stage 2."""

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

from src.activations import (  # noqa: E402
    input_device_for_model,
    load_tl_model,
    render_chat_text,
    validate_hooks,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    make_gaussian_unit_direction,
    make_orthogonal_unit_direction,
    parse_float_list,
    parse_int_list,
    raw_probe_projection_sidecar,
    score_reply,
    select_balanced_stage1_rows,
    summarize_steering_rows,
    train_raw_probe_direction,
)


@dataclass(frozen=True)
class DecodeCondition:
    label: str
    injection_kind: str | None
    strength_sd: float
    gate: str


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
        orthogonal_direction=orthogonal_direction.astype(np.float32),
        gaussian_direction=gaussian_direction.astype(np.float32),
        raw_coef=direction["raw_coef"],
        coef_std=direction["coef_std"],
        scaler_mean=direction["scaler_mean"],
        scaler_scale=direction["scaler_scale"],
        train_projection_std=np.array(direction["train_projection_std"], dtype=np.float32),
        train_projection_mean=np.array(direction["train_projection_mean"], dtype=np.float32),
        best_c=np.array(direction["best_c"], dtype=np.float32),
    )


def strength_label(strength: float) -> str:
    sign = "pos" if strength >= 0 else "neg"
    magnitude = f"{abs(strength):g}".replace(".", "p")
    return f"{sign}{magnitude}sd"


def threshold_label(threshold_z: float) -> str:
    label = f"{threshold_z:g}".replace("-", "neg").replace(".", "p")
    return f"zlt{label}"


def parse_decode_condition_kinds(value: str) -> list[str]:
    allowed = {
        "baseline",
        "conditional_raw",
        "conditional_orthogonal",
        "conditional_gaussian",
        "always_raw",
    }
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition kind")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    return parsed


def make_decode_condition_plan(
    *,
    condition_kinds: list[str],
    strengths: tuple[float, ...],
    gate_threshold_z: float,
) -> list[DecodeCondition]:
    plan: list[DecodeCondition] = []
    if "baseline" in condition_kinds:
        plan.append(DecodeCondition("baseline", None, 0.0, "none"))
    condition_specs = [
        ("conditional_raw", "raw", "raw_below_threshold"),
        ("conditional_orthogonal", "orthogonal", "raw_below_threshold"),
        ("conditional_gaussian", "gaussian", "raw_below_threshold"),
        ("always_raw", "raw", "always"),
    ]
    z_label = threshold_label(gate_threshold_z)
    for condition_kind, injection_kind, gate in condition_specs:
        if condition_kind not in condition_kinds:
            continue
        for strength in strengths:
            if strength == 0:
                continue
            if gate == "always":
                label = f"{condition_kind}_{strength_label(strength)}"
            else:
                label = f"{condition_kind}_{strength_label(strength)}_{z_label}"
            plan.append(
                DecodeCondition(
                    label=label,
                    injection_kind=injection_kind,
                    strength_sd=float(strength),
                    gate=gate,
                )
            )
    if not plan:
        raise ValueError("condition plan is empty")
    return plan


def _cached_tensor(
    cache: dict[tuple[str, torch.dtype, str], torch.Tensor],
    *,
    key_name: str,
    vector: np.ndarray,
    act: torch.Tensor,
) -> torch.Tensor:
    key = (str(act.device), act.dtype, key_name)
    tensor = cache.get(key)
    if tensor is None:
        tensor = torch.as_tensor(vector, device=act.device, dtype=act.dtype)
        cache[key] = tensor
    return tensor


def make_decode_correction_hook(
    *,
    monitor_vector: np.ndarray,
    injection_vector: np.ndarray,
    delta: float,
    projection_mean: float,
    projection_std: float,
    gate_threshold_z: float,
    gate: str,
) -> tuple[Any, dict[str, Any]]:
    cache: dict[tuple[str, torch.dtype, str], torch.Tensor] = {}
    state: dict[str, Any] = {
        "calls": 0,
        "application_calls": 0,
        "application_batches": 0,
        "projection_z_values": [],
        "applied_projection_z_values": [],
    }

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        state["calls"] += 1
        monitor = _cached_tensor(cache, key_name="monitor", vector=monitor_vector, act=act)
        last = act[:, -1, :]
        projection = torch.sum(last.float() * monitor.float(), dim=-1)
        z = (projection - float(projection_mean)) / float(projection_std)
        z_values = z.detach().float().cpu().tolist()
        state["projection_z_values"].extend(float(value) for value in z_values)

        if gate == "always":
            mask = torch.ones_like(z, dtype=torch.bool)
        elif gate == "raw_below_threshold":
            mask = z < float(gate_threshold_z)
        else:
            raise ValueError(f"unknown gate {gate!r}")

        if not bool(mask.any().item()):
            return act

        injection = _cached_tensor(cache, key_name="injection", vector=injection_vector, act=act)
        mask_f = mask.to(dtype=act.dtype, device=act.device).unsqueeze(-1)
        act[:, -1, :] = act[:, -1, :] + mask_f * float(delta) * injection
        applied_z = z.detach().float().cpu()[mask.detach().cpu()].tolist()
        state["applied_projection_z_values"].extend(float(value) for value in applied_z)
        state["application_calls"] += 1
        state["application_batches"] += int(mask.sum().item())
        return act

    return hook_fn, state


def summarize_hook_state(state: dict[str, Any]) -> dict[str, Any]:
    z_values = [float(value) for value in state.get("projection_z_values", [])]
    applied = [float(value) for value in state.get("applied_projection_z_values", [])]
    return {
        "calls": int(state.get("calls", 0)),
        "application_calls": int(state.get("application_calls", 0)),
        "application_batches": int(state.get("application_batches", 0)),
        "projection_z_count": len(z_values),
        "projection_z_min": min(z_values) if z_values else None,
        "projection_z_max": max(z_values) if z_values else None,
        "projection_z_mean": float(sum(z_values) / len(z_values)) if z_values else None,
        "projection_z_first": z_values[0] if z_values else None,
        "projection_z_last": z_values[-1] if z_values else None,
        "applied_projection_z_count": len(applied),
        "applied_projection_z_min": min(applied) if applied else None,
        "applied_projection_z_max": max(applied) if applied else None,
        "applied_projection_z_mean": float(sum(applied) / len(applied)) if applied else None,
    }


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


def build_report_interface_fields(
    *,
    rows: list[dict[str, Any]],
    steering_summary: dict[str, Any],
) -> dict[str, Any]:
    by_condition = steering_summary.get("by_condition", {})
    flips = steering_summary.get("flips_vs_baseline", {})
    parse_fail_rate = {
        condition: metrics.get("parse_fail_rate")
        for condition, metrics in by_condition.items()
    }
    matched_noise = {
        condition: {
            "metrics": by_condition.get(condition),
            "paired_flips": flips.get(condition),
        }
        for condition in by_condition
        if "gaussian" in condition
    }
    return {
        "baseline_metrics": by_condition.get("baseline"),
        "intervention_metrics": {
            condition: metrics
            for condition, metrics in by_condition.items()
            if condition != "baseline"
        },
        "paired_flips": flips,
        "parse_fail_rate": parse_fail_rate,
        "matched_noise_summary": matched_noise,
        "n": len(rows),
        "controls": [
            "regenerated_baseline",
            "orthogonal_direction",
            "matched_gaussian_noise",
            "positive_control",
        ],
        "causal_abstraction_claim": (
            "Tests whether the raw free_form_correctness direction can be used as a "
            "decode-time correction state. A causal repair claim requires conditional "
            "raw false-to-true repairs above orthogonal and matched-Gaussian controls."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument(
        "--activation-prefix",
        type=Path,
        default=None,
        help=(
            "Optional activation artifact prefix. When supplied, use "
            "<prefix>.safetensors and <prefix>.example_ids.jsonl instead of "
            "constructing the prefix from --activation-dir/--model-key/--task/--layer."
        ),
    )
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260427)
    parser.add_argument("--probe-seed", type=int, default=20260472)
    parser.add_argument("--orthogonal-seed", type=int, default=20260545)
    parser.add_argument("--gaussian-seed", type=int, default=20260604)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument(
        "--conditions",
        default="baseline,conditional_raw,conditional_orthogonal,conditional_gaussian",
    )
    parser.add_argument("--strengths", default="1")
    parser.add_argument(
        "--gate-threshold-z",
        type=float,
        default=0.0,
        help="Apply conditional correction when the raw correctness projection z-score is below this value.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--load-mode",
        choices=("no-processing", "default"),
        default="no-processing",
    )
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/decode_time/decode_time_correction_l45_property_pilot.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/decode_time/decode_time_correction_l45_property_direction.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/decode_time_correction_27b_l45_property_pilot.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)

    heights = parse_int_list(args.heights)
    strengths = parse_float_list(args.strengths)
    condition_kinds = parse_decode_condition_kinds(args.conditions)
    condition_plan = make_decode_condition_plan(
        condition_kinds=condition_kinds,
        strengths=strengths,
        gate_threshold_z=args.gate_threshold_z,
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

    print("Stage 2 decode-time correction pilot", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"gate_threshold_z={args.gate_threshold_z}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    started = time.time()
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
    orthogonal_direction = make_orthogonal_unit_direction(
        direction["unit_direction"],
        seed=args.orthogonal_seed,
    )
    gaussian_direction = make_gaussian_unit_direction(
        direction["unit_direction"],
        seed=args.gaussian_seed,
    )
    save_direction_artifact(
        path=args.direction_output,
        direction=direction,
        orthogonal_direction=orthogonal_direction,
        gaussian_direction=gaussian_direction,
    )
    print(
        "direction: "
        f"best_c={direction['best_c']} val_auc={direction['val_auc']:.4f} "
        f"test_auc={direction['test_auc']:.4f} proj_std={direction['train_projection_std']:.4f}",
        flush=True,
    )

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
    print(
        f"selected_rows={len(selected_rows)} "
        f"available_counts={selection_summary['available_counts']}",
        flush=True,
    )
    projection_sidecar = raw_probe_projection_sidecar(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        direction=direction,
    )
    projection_by_row_index = projection_sidecar["by_row_index"]
    missing_projection_rows = [
        int(row["row_index"])
        for row in selected_rows
        if int(row["row_index"]) not in projection_by_row_index
    ]
    print(
        "probe_projection_sidecar: "
        f"covered={len(selected_rows) - len(missing_projection_rows)}/{len(selected_rows)} "
        f"missing={missing_projection_rows[:5]}",
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
    hook_name = validate_hooks(model, [args.layer])[0]
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    print(f"using_hook={hook_name}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    vector_by_kind = {
        "raw": direction["unit_direction"],
        "orthogonal": orthogonal_direction,
        "gaussian": gaussian_direction,
    }
    projection_mean = float(direction["train_projection_mean"])
    projection_std = float(direction["train_projection_std"])
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
            print(
                f"row {row_idx}/{len(selected_rows)} "
                f"source_row={stage1_row['row_index']} h={stage1_row['height']} "
                f"original_correct={stage1_row['is_correct_strong']} prompt_tokens={len(token_ids)}",
                flush=True,
            )
            projection_fields = projection_by_row_index.get(int(stage1_row["row_index"]), {})
            for condition in condition_plan:
                hook_summary = {
                    "calls": 0,
                    "application_calls": 0,
                    "application_batches": 0,
                    "projection_z_count": 0,
                    "projection_z_min": None,
                    "projection_z_max": None,
                    "projection_z_mean": None,
                    "projection_z_first": None,
                    "projection_z_last": None,
                    "applied_projection_z_count": 0,
                    "applied_projection_z_min": None,
                    "applied_projection_z_max": None,
                    "applied_projection_z_mean": None,
                }
                if condition.injection_kind is None:
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
                    hook_fn, hook_state = make_decode_correction_hook(
                        monitor_vector=direction["unit_direction"],
                        injection_vector=vector_by_kind[condition.injection_kind],
                        delta=delta,
                        projection_mean=projection_mean,
                        projection_std=projection_std,
                        gate_threshold_z=args.gate_threshold_z,
                        gate=condition.gate,
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
                    hook_summary = summarize_hook_state(hook_state)
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
                    "method": "decode_time_raw_projection_gate",
                    "target_variable": "free_form_correctness",
                    "representation_type": "decode_time_correction_state",
                    "injection_kind": condition.injection_kind,
                    "gate": condition.gate,
                    "gate_threshold_z": args.gate_threshold_z,
                    "strength_sd": condition.strength_sd,
                    "intervention_delta_l2": abs(condition.strength_sd * projection_std),
                    "monitor_direction_kind": "raw",
                    "direction_projection": projection_fields.get("direction_projection"),
                    "direction_projection_z": projection_fields.get("direction_projection_z"),
                    "direction_projection_higher_is": projection_fields.get("direction_projection_higher_is"),
                    "prompt_token_count": len(token_ids),
                    "generated_token_count": len(new_ids),
                    "model_output": reply,
                    **hook_summary,
                    **score,
                }
                rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"  {condition.label}: strong={output_row['is_correct_strong']} "
                    f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                    f"new_tokens={len(new_ids)} gate={hook_summary['application_calls']}/{hook_summary['calls']} "
                    f"z_mean={hook_summary['projection_z_mean']}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    steering_summary = summarize_steering_rows(rows)
    interface_fields = build_report_interface_fields(rows=rows, steering_summary=steering_summary)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_decode_time_correction.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "free_form_correctness",
        "split": args.split_family,
        "site_or_layer": f"L{args.layer}:{hook_name}",
        "method": "conditional_decode_time_raw_projection_injection",
        "representation_type": "decode_time_correction_state",
        "layer": args.layer,
        "hook_name": hook_name,
        "activation_path": str(activation_path),
        "sidecar_path": str(sidecar_path),
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "direction_output": str(args.direction_output),
        "out_jsonl": str(args.out_jsonl),
        "probe_direction": serializable_direction_summary(direction),
        "selection": selection_summary,
        "probe_projection_sidecar": projection_sidecar["summary"],
        "generation": {
            "conditions": [condition.__dict__ for condition in condition_plan],
            "strengths_sd": list(strengths),
            "gate_threshold_z": args.gate_threshold_z,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "summary": steering_summary,
        **interface_fields,
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
