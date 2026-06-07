#!/usr/bin/env python3
"""Run prompt-margin gated decode-time correction on the commitment rowset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import (  # noqa: E402
    build_report_interface_fields,
    generate_one,
    json_default,
    make_decode_correction_hook,
    package_version,
    save_direction_artifact,
    serializable_direction_summary,
    summarize_hook_state,
    torch_dtype,
)
from src.activations import input_device_for_model, load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    make_gaussian_unit_direction,
    make_orthogonal_unit_direction,
    parse_float_list,
    raw_probe_projection_sidecar,
    score_reply,
    summarize_steering_rows,
    train_raw_probe_direction,
)


@dataclass(frozen=True)
class DecodeCondition:
    label: str
    injection_kind: str | None
    strength_sd: float
    gate: str


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
                row = json.loads(line)
                row["row_index"] = idx
                rows[idx] = row
    return rows


def strength_label(strength: float) -> str:
    sign = "pos" if strength >= 0 else "neg"
    magnitude = f"{abs(strength):g}".replace(".", "p")
    return f"{sign}{magnitude}sd"


def threshold_label(threshold: float) -> str:
    return f"gvflt{threshold:g}".replace("-", "neg").replace(".", "p")


def parse_condition_kinds(value: str) -> list[str]:
    allowed = {
        "baseline",
        "prompt_margin_raw",
        "prompt_margin_orthogonal",
        "prompt_margin_gaussian",
    }
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    return parsed


def make_condition_plan(
    *,
    condition_kinds: list[str],
    strengths: tuple[float, ...],
    prompt_gold_vs_foil_threshold: float,
) -> list[DecodeCondition]:
    plan: list[DecodeCondition] = []
    if "baseline" in condition_kinds:
        plan.append(DecodeCondition("baseline", None, 0.0, "none"))
    threshold = threshold_label(prompt_gold_vs_foil_threshold)
    specs = [
        ("prompt_margin_raw", "raw"),
        ("prompt_margin_orthogonal", "orthogonal"),
        ("prompt_margin_gaussian", "gaussian"),
    ]
    for kind, injection_kind in specs:
        if kind not in condition_kinds:
            continue
        for strength in strengths:
            if strength == 0:
                continue
            plan.append(
                DecodeCondition(
                    label=f"{kind}_{strength_label(strength)}_{threshold}",
                    injection_kind=injection_kind,
                    strength_sd=float(strength),
                    gate="prompt_gold_vs_foil_below_threshold",
                )
            )
    if not plan:
        raise ValueError("condition plan is empty")
    return plan


def zero_hook_summary() -> dict[str, Any]:
    return {
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


def select_prefix_rows(
    *,
    prefix_jsonl: Path,
    source_jsonl: Path,
    checkpoint: str,
    limit: int | None,
    prompt_gold_vs_foil_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = read_source_rows(source_jsonl)
    prefix_rows = [row for row in read_jsonl(prefix_jsonl) if str(row.get("checkpoint")) == str(checkpoint)]
    prefix_rows = sorted(prefix_rows, key=lambda row: int(row["source_row_index"]))
    if limit is not None and limit >= 0:
        prefix_rows = prefix_rows[:limit]
    selected = []
    skipped_missing_source = 0
    for prefix_row in prefix_rows:
        source_row_index = int(prefix_row["source_row_index"])
        source_row = source_rows.get(source_row_index)
        if source_row is None:
            skipped_missing_source += 1
            continue
        row = dict(source_row)
        row["row_index"] = source_row_index
        row["prefix_checkpoint"] = prefix_row.get("checkpoint")
        row["prefix_monitor"] = {
            "source_row_index": source_row_index,
            "example_id": prefix_row.get("example_id"),
            "height": prefix_row.get("height"),
            "parse_failed": bool(prefix_row.get("parse_failed")),
            "generated_is_correct_strong": bool(prefix_row.get("generated_is_correct_strong")),
            "generated_is_correct_weak": bool(prefix_row.get("generated_is_correct_weak")),
            "quality_score": prefix_row.get("quality_score"),
            "gold_hypothesis": prefix_row.get("gold_hypothesis"),
            "foil_hypothesis": prefix_row.get("foil_hypothesis"),
            "selected_hypothesis": prefix_row.get("selected_hypothesis"),
            "gold_vs_foil_logprob_margin": prefix_row.get("gold_vs_foil_logprob_margin"),
            "gold_vs_foil_mean_logprob_margin": prefix_row.get("gold_vs_foil_mean_logprob_margin"),
            "selected_vs_gold_logprob_margin": prefix_row.get("selected_vs_gold_logprob_margin"),
            "selected_vs_foil_logprob_margin": prefix_row.get("selected_vs_foil_logprob_margin"),
        }
        selected.append(row)
    triggered = [
        row for row in selected
        if float(row["prefix_monitor"]["gold_vs_foil_logprob_margin"]) < prompt_gold_vs_foil_threshold
    ]
    generated_wrong = [row for row in selected if not bool(row["prefix_monitor"].get("generated_is_correct_strong"))]
    generated_correct = [row for row in selected if bool(row["prefix_monitor"].get("generated_is_correct_strong"))]
    by_height: dict[str, int] = defaultdict(int)
    for row in selected:
        by_height[f"h{row.get('height')}"] += 1
    summary = {
        "selection_mode": "prefix_conditioned_margin_manifest_checkpoint",
        "prefix_jsonl": str(prefix_jsonl),
        "source_jsonl": str(source_jsonl),
        "checkpoint": checkpoint,
        "limit": limit,
        "selected_rows": len(selected),
        "skipped_missing_source": skipped_missing_source,
        "heights": dict(sorted(by_height.items())),
        "prompt_gold_vs_foil_threshold": prompt_gold_vs_foil_threshold,
        "prompt_margin_gate_triggered_rows": len(triggered),
        "prompt_margin_gate_wrong_triggered": sum(not bool(row["prefix_monitor"].get("generated_is_correct_strong")) for row in triggered),
        "prompt_margin_gate_correct_triggered": sum(bool(row["prefix_monitor"].get("generated_is_correct_strong")) for row in triggered),
        "calibration_generated_wrong_rows": len(generated_wrong),
        "calibration_generated_correct_rows": len(generated_correct),
        "calibration_parse_fail_rows": sum(bool(row["prefix_monitor"].get("parse_failed")) for row in selected),
        "triggered_source_row_indices": [int(row["row_index"]) for row in triggered],
    }
    return selected, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--prefix-trajectory-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--prefix-checkpoint", default="0")
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-prefix", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--prompt-gold-vs-foil-threshold", type=float, default=-15.0)
    parser.add_argument("--probe-seed", type=int, default=20260472)
    parser.add_argument("--orthogonal-seed", type=int, default=20260545)
    parser.add_argument("--gaussian-seed", type=int, default=20260604)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--conditions", default="baseline,prompt_margin_raw,prompt_margin_orthogonal,prompt_margin_gaussian")
    parser.add_argument("--strengths", default="1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/decode_time/prompt_margin_gated_decode_correction_27b_l45_property_manifest.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/decode_time/prompt_margin_gated_decode_correction_27b_l45_property_manifest_direction.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/prompt_margin_gated_decode_correction_27b_l45_property_manifest.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    strengths = parse_float_list(args.strengths)
    condition_plan = make_condition_plan(
        condition_kinds=parse_condition_kinds(args.conditions),
        strengths=strengths,
        prompt_gold_vs_foil_threshold=args.prompt_gold_vs_foil_threshold,
    )
    if not any(condition.label == "baseline" for condition in condition_plan):
        raise ValueError("baseline condition is required for paired causal interpretation")
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)
    activation_prefix = (
        args.activation_prefix
        if args.activation_prefix is not None
        else args.activation_dir / f"{args.model_key}_{args.task}_L{args.layer}"
    )
    activation_path = activation_prefix.with_suffix(".safetensors")
    sidecar_path = activation_prefix.with_suffix(".example_ids.jsonl")

    print("Prompt-margin gated decode-time correction", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"prompt_gold_vs_foil_threshold={args.prompt_gold_vs_foil_threshold}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=args.prompt_gold_vs_foil_threshold,
    )
    if not selected_rows:
        raise ValueError(f"no rows selected: {selection_summary}")
    print(f"selected_rows={len(selected_rows)} selection={selection_summary}", flush=True)
    if args.dry_run:
        first = selected_rows[0]
        print(
            json.dumps(
                {
                    "selection": selection_summary,
                    "conditions": [condition.__dict__ for condition in condition_plan],
                    "first_row": {
                        "source_row_index": first.get("row_index"),
                        "example_id": first.get("example_id"),
                        "height": first.get("height"),
                        "prefix_monitor": first.get("prefix_monitor"),
                    },
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            ),
            flush=True,
        )
        return 0

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
    orthogonal_direction = make_orthogonal_unit_direction(direction["unit_direction"], seed=args.orthogonal_seed)
    gaussian_direction = make_gaussian_unit_direction(direction["unit_direction"], seed=args.gaussian_seed)
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
    projection_sidecar = raw_probe_projection_sidecar(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        direction=direction,
    )
    projection_by_row_index = projection_sidecar["by_row_index"]

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(
        "scorer_preflight: "
        f"strong={scorer_preflight['is_correct_strong']} parse_failed={scorer_preflight['parse_failed']}",
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

    vector_by_kind = {
        "raw": direction["unit_direction"],
        "orthogonal": orthogonal_direction,
        "gaussian": gaussian_direction,
    }
    projection_mean = float(direction["train_projection_mean"])
    projection_std = float(direction["train_projection_std"])

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            prefix_monitor = stage1_row["prefix_monitor"]
            prompt_gate_active = float(prefix_monitor["gold_vs_foil_logprob_margin"]) < args.prompt_gold_vs_foil_threshold
            prompt_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if len(token_ids) > args.n_ctx:
                raise ValueError(f"row {stage1_row['row_index']} prompt exceeds n_ctx={args.n_ctx}: {len(token_ids)}")
            print(
                f"row {row_idx}/{len(selected_rows)} source_row={stage1_row['row_index']} "
                f"h={stage1_row.get('height')} prompt_gate={prompt_gate_active} "
                f"g-v-f={prefix_monitor['gold_vs_foil_logprob_margin']:.3f} "
                f"calib_correct={prefix_monitor['generated_is_correct_strong']} prompt_tokens={len(token_ids)}",
                flush=True,
            )
            projection_fields = projection_by_row_index.get(int(stage1_row["row_index"]), {})
            baseline_cache: dict[str, Any] | None = None
            for condition in condition_plan:
                hook_summary = zero_hook_summary()
                reused_baseline = False
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
                    score = score_reply(stage1_row, reply)
                    baseline_cache = {"new_ids": new_ids, "reply": reply, "score": score}
                elif not prompt_gate_active and baseline_cache is not None:
                    new_ids = baseline_cache["new_ids"]
                    reply = baseline_cache["reply"]
                    score = baseline_cache["score"]
                    reused_baseline = True
                else:
                    delta = condition.strength_sd * projection_std
                    hook_fn, hook_state = make_decode_correction_hook(
                        monitor_vector=direction["unit_direction"],
                        injection_vector=vector_by_kind[condition.injection_kind],
                        delta=delta,
                        projection_mean=projection_mean,
                        projection_std=projection_std,
                        gate_threshold_z=0.0,
                        gate="always" if prompt_gate_active else "raw_below_threshold",
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
                    "calibration_generated_is_correct_strong": prefix_monitor.get("generated_is_correct_strong"),
                    "calibration_generated_is_correct_weak": prefix_monitor.get("generated_is_correct_weak"),
                    "calibration_parse_failed": prefix_monitor.get("parse_failed"),
                    "condition": condition.label,
                    "method": "prompt_margin_gated_decode_correction",
                    "target_variable": "commitment_state",
                    "representation_type": "decode_time_correction_state",
                    "injection_kind": condition.injection_kind,
                    "gate": condition.gate,
                    "prompt_margin_gate_active": prompt_gate_active,
                    "prompt_gold_vs_foil_threshold": args.prompt_gold_vs_foil_threshold,
                    "prompt_gold_vs_foil_logprob_margin": prefix_monitor.get("gold_vs_foil_logprob_margin"),
                    "prompt_gold_vs_foil_mean_logprob_margin": prefix_monitor.get("gold_vs_foil_mean_logprob_margin"),
                    "prompt_selected_vs_gold_logprob_margin": prefix_monitor.get("selected_vs_gold_logprob_margin"),
                    "prompt_selected_vs_foil_logprob_margin": prefix_monitor.get("selected_vs_foil_logprob_margin"),
                    "gold_hypothesis": prefix_monitor.get("gold_hypothesis"),
                    "foil_hypothesis": prefix_monitor.get("foil_hypothesis"),
                    "selected_hypothesis": prefix_monitor.get("selected_hypothesis"),
                    "strength_sd": condition.strength_sd,
                    "intervention_delta_l2": abs(condition.strength_sd * projection_std) if condition.injection_kind else 0.0,
                    "monitor_direction_kind": "raw_probe_projection_for_hook_logging",
                    "direction_projection": projection_fields.get("direction_projection"),
                    "direction_projection_z": projection_fields.get("direction_projection_z"),
                    "direction_projection_higher_is": projection_fields.get("direction_projection_higher_is"),
                    "prompt_token_count": len(token_ids),
                    "generated_token_count": len(new_ids),
                    "reused_baseline_for_inactive_gate": reused_baseline,
                    "model_output": reply,
                    **hook_summary,
                    **score,
                }
                rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                fout.flush()
                print(
                    f"  {condition.label}: strong={output_row['is_correct_strong']} "
                    f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                    f"new_tokens={len(new_ids)} gate_active={prompt_gate_active} "
                    f"hook={hook_summary['application_calls']}/{hook_summary['calls']} reused={reused_baseline}",
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
        "script": "scripts/stage2_prompt_margin_gated_decode_correction.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "commitment_state",
        "split": args.split_family,
        "site_or_layer": f"L{args.layer}:{hook_name}",
        "method": "prompt_margin_gated_decode_correction",
        "representation_type": "decode_time_correction_state",
        "layer": args.layer,
        "hook_name": hook_name,
        "activation_path": str(activation_path),
        "sidecar_path": str(sidecar_path),
        "jsonl": str(args.jsonl),
        "prefix_trajectory_jsonl": str(args.prefix_trajectory_jsonl),
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
            "prompt_gold_vs_foil_threshold": args.prompt_gold_vs_foil_threshold,
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
    report["controls"] = [
        "regenerated_baseline",
        "orthogonal_direction",
        "matched_gaussian_noise",
        "positive_control",
        "prompt_margin_inactive_gate_reuses_baseline",
    ]
    report["causal_abstraction_claim"] = (
        "Tests whether a conservative prompt-margin `gold_vs_foil_margin` gate can make a raw residual "
        "direction act as a decode-time correction state for `commitment_state` / free-form correctness. "
        "A repair claim requires false-to-true repairs above orthogonal and matched-Gaussian controls."
    )
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
