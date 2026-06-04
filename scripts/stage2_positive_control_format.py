#!/usr/bin/env python3
"""Run a Gemma 3 casing/output-format steering positive control."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_positive_control_verbosity import (  # noqa: E402
    EVAL_QUESTIONS,
    TRAIN_QUESTIONS,
    activation_for_prompt,
    generate_one,
    json_default,
    make_direction,
    make_steering_hook,
    package_version,
    prompt_for,
    save_direction,
    torch_dtype,
)
from src.activations import (  # noqa: E402
    input_device_for_model,
    load_tl_model,
    render_chat_text,
    validate_hooks,
)
from src.stage2_causal_abstraction import make_experiment_report  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    make_gaussian_unit_direction,
    make_orthogonal_unit_direction,
    parse_float_list,
    strength_label,
)


SYSTEM_PROMPT = "You are a helpful assistant."
UPPER_INSTRUCTION = "Answer in ALL CAPS. Do not use lowercase letters."
LOWER_INSTRUCTION = "Answer in lowercase only. Do not use uppercase letters."
BASELINE_INSTRUCTION = "Answer naturally in one concise sentence."


@dataclass(frozen=True)
class FormatCondition:
    label: str
    direction_kind: str | None
    strength_sd: float
    signed_strength: float


def alpha_case_metrics(text: str, token_count: int) -> dict[str, Any]:
    alpha_chars = [ch for ch in text if ch.isalpha()]
    alpha_n = len(alpha_chars)
    upper_n = sum(ch.isupper() for ch in alpha_chars)
    lower_n = sum(ch.islower() for ch in alpha_chars)
    upper_frac = upper_n / alpha_n if alpha_n else 0.0
    lower_frac = lower_n / alpha_n if alpha_n else 0.0
    return {
        "generated_token_count": int(token_count),
        "output_char_count": len(text),
        "alpha_char_count": int(alpha_n),
        "uppercase_alpha_count": int(upper_n),
        "lowercase_alpha_count": int(lower_n),
        "uppercase_alpha_fraction": float(upper_frac),
        "lowercase_alpha_fraction": float(lower_frac),
        "is_uppercase_style": bool(alpha_n >= 8 and upper_frac >= 0.70),
        "is_lowercase_style": bool(alpha_n >= 8 and upper_frac <= 0.05),
    }


def make_conditions(kinds: list[str], strengths: tuple[float, ...]) -> list[FormatCondition]:
    allowed = {"baseline", "toward_upper", "toward_lower", "orthogonal", "gaussian"}
    unknown = sorted(set(kinds) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    plan: list[FormatCondition] = []
    if "baseline" in kinds:
        plan.append(FormatCondition("baseline", None, 0.0, 0.0))
    for kind in ("toward_upper", "toward_lower", "orthogonal", "gaussian"):
        if kind not in kinds:
            continue
        for strength in strengths:
            if strength <= 0:
                raise ValueError("format steering strengths must be positive")
            signed = -float(strength) if kind == "toward_lower" else float(strength)
            plan.append(FormatCondition(f"{kind}_{strength_label(strength)}", kind, float(strength), signed))
    return plan


def control_adjusted_effect(*, directional_effect: float, control_mean: float | None, control_std: float | None) -> float | None:
    if control_mean is None or control_std is None:
        return None
    adjusted = directional_effect - control_mean
    if control_std == 0.0:
        if adjusted > 0.0:
            return 999.0
        return 0.0
    return float(adjusted / control_std)


def summarize_rows(rows: list[dict[str, Any]], *, min_uppercase_delta: float) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    condition_summary: dict[str, Any] = {}
    for condition, subset in sorted(by_condition.items()):
        n = len(subset)
        condition_summary[condition] = {
            "n": n,
            "mean_generated_tokens": float(np.mean([row["generated_token_count"] for row in subset])) if n else None,
            "mean_uppercase_fraction": float(np.mean([row["uppercase_alpha_fraction"] for row in subset])) if n else None,
            "mean_lowercase_fraction": float(np.mean([row["lowercase_alpha_fraction"] for row in subset])) if n else None,
            "uppercase_style_rate": float(np.mean([row["is_uppercase_style"] for row in subset])) if n else None,
            "lowercase_style_rate": float(np.mean([row["is_lowercase_style"] for row in subset])) if n else None,
        }

    baselines = {row["prompt_id"]: row for row in rows if row["condition"] == "baseline"}
    paired: dict[str, Any] = {}
    control_abs_upper_deltas: list[float] = []
    for condition, subset in sorted(by_condition.items()):
        if condition == "baseline":
            continue
        pairs = [(baselines[row["prompt_id"]], row) for row in subset if row["prompt_id"] in baselines]
        upper_deltas = [row["uppercase_alpha_fraction"] - base["uppercase_alpha_fraction"] for base, row in pairs]
        lower_deltas = [row["lowercase_alpha_fraction"] - base["lowercase_alpha_fraction"] for base, row in pairs]
        token_deltas = [row["generated_token_count"] - base["generated_token_count"] for base, row in pairs]
        if condition.startswith(("orthogonal", "gaussian")):
            control_abs_upper_deltas.extend(abs(float(delta)) for delta in upper_deltas)
        paired[condition] = {
            "paired_n": len(pairs),
            "mean_uppercase_fraction_delta": float(np.mean(upper_deltas)) if upper_deltas else None,
            "median_uppercase_fraction_delta": float(np.median(upper_deltas)) if upper_deltas else None,
            "mean_lowercase_fraction_delta": float(np.mean(lower_deltas)) if lower_deltas else None,
            "mean_token_delta": float(np.mean(token_deltas)) if token_deltas else None,
            "uppercase_delta_gt0_rate": float(np.mean([delta > 0 for delta in upper_deltas])) if upper_deltas else None,
            "uppercase_delta_lt0_rate": float(np.mean([delta < 0 for delta in upper_deltas])) if upper_deltas else None,
            "uppercase_style_false_to_true": int(
                sum((not base["is_uppercase_style"]) and row["is_uppercase_style"] for base, row in pairs)
            ),
            "lowercase_style_false_to_true": int(
                sum((not base["is_lowercase_style"]) and row["is_lowercase_style"] for base, row in pairs)
            ),
        }

    control_mean = float(np.mean(control_abs_upper_deltas)) if control_abs_upper_deltas else None
    control_std = float(np.std(control_abs_upper_deltas, ddof=1)) if len(control_abs_upper_deltas) > 1 else None
    for condition, stats in paired.items():
        mean_delta = stats.get("mean_uppercase_fraction_delta")
        if mean_delta is None:
            stats["effect_over_control_sigma"] = None
            continue
        directional_effect = -mean_delta if condition.startswith("toward_lower") else mean_delta
        stats["effect_over_control_sigma"] = control_adjusted_effect(
            directional_effect=directional_effect,
            control_mean=control_mean,
            control_std=control_std,
        )

    best_upper = max(
        (stats for condition, stats in paired.items() if condition.startswith("toward_upper")),
        key=lambda stats: stats.get("effect_over_control_sigma") if stats.get("effect_over_control_sigma") is not None else -1e9,
        default=None,
    )
    best_lower = max(
        (stats for condition, stats in paired.items() if condition.startswith("toward_lower")),
        key=lambda stats: stats.get("effect_over_control_sigma") if stats.get("effect_over_control_sigma") is not None else -1e9,
        default=None,
    )
    upper_pass = bool(
        best_upper
        and best_upper.get("effect_over_control_sigma") is not None
        and best_upper["effect_over_control_sigma"] >= 2.0
        and best_upper.get("uppercase_delta_gt0_rate", 0.0) >= 0.70
        and best_upper.get("mean_uppercase_fraction_delta", 0.0) >= min_uppercase_delta
    )
    lower_pass = bool(
        best_lower
        and best_lower.get("effect_over_control_sigma") is not None
        and best_lower["effect_over_control_sigma"] >= 2.0
        and best_lower.get("uppercase_delta_lt0_rate", 0.0) >= 0.70
        and (-best_lower.get("mean_uppercase_fraction_delta", 0.0)) >= min_uppercase_delta
    )

    return {
        "by_condition": condition_summary,
        "paired_deltas_vs_baseline": paired,
        "matched_noise_summary": {
            "control_abs_uppercase_delta_mean": control_mean,
            "control_abs_uppercase_delta_std": control_std,
            "control_abs_uppercase_delta_n": len(control_abs_upper_deltas),
            "best_toward_upper": best_upper,
            "best_toward_lower": best_lower,
            "min_uppercase_delta": min_uppercase_delta,
            "passed_positive_control_gate": bool(upper_pass or lower_pass),
            "passed_toward_upper": upper_pass,
            "passed_toward_lower": lower_pass,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--train-limit", type=int, default=24)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--orthogonal-seed", type=int, default=20260606)
    parser.add_argument("--gaussian-seed", type=int, default=20260607)
    parser.add_argument("--conditions", default="baseline,toward_upper,toward_lower,orthogonal,gaussian")
    parser.add_argument("--strengths", default="0.5,1,2,4")
    parser.add_argument("--intervention-scope", choices=("prompt_only", "last_token_each_forward"), default="last_token_each_forward")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-uppercase-delta", type=float, default=0.25)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/positive_control/format_gemma3_27b_l45.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/positive_control/format_gemma3_27b_l45_direction.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/positive_control_format_gemma3_27b_l45.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)

    dtype = torch_dtype(args.dtype)
    conditions = make_conditions(
        [part.strip() for part in args.conditions.split(",") if part.strip()],
        parse_float_list(args.strengths),
    )
    train_questions = TRAIN_QUESTIONS[: args.train_limit]
    eval_questions = EVAL_QUESTIONS[: args.eval_limit]
    if not train_questions or not eval_questions:
        raise ValueError("train/eval prompt sets must be non-empty")

    print("Stage 2 casing positive-control steering", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"train_questions={len(train_questions)} eval_questions={len(eval_questions)}", flush=True)
    print(f"conditions={[condition.label for condition in conditions]}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)

    started = time.time()
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

    upper_acts = []
    lower_acts = []
    prompt_token_counts = []
    for idx, question in enumerate(train_questions, start=1):
        lower_act, lower_tokens = activation_for_prompt(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            prompt_text=prompt_for(question, LOWER_INSTRUCTION),
            model_name=args.model,
        )
        upper_act, upper_tokens = activation_for_prompt(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            prompt_text=prompt_for(question, UPPER_INSTRUCTION),
            model_name=args.model,
        )
        lower_acts.append(lower_act)
        upper_acts.append(upper_act)
        prompt_token_counts.append({"question_index": idx, "lower_tokens": lower_tokens, "upper_tokens": upper_tokens})
        print(f"train {idx}/{len(train_questions)} lower_tokens={lower_tokens} upper_tokens={upper_tokens}", flush=True)

    direction = make_direction(long_acts=np.stack(upper_acts), short_acts=np.stack(lower_acts))
    orthogonal = make_orthogonal_unit_direction(direction["unit_direction"], seed=args.orthogonal_seed)
    gaussian = make_gaussian_unit_direction(direction["unit_direction"], seed=args.gaussian_seed)
    save_direction(args.direction_output, direction=direction, orthogonal=orthogonal, gaussian=gaussian)
    projection_std = float(direction["train_projection_std"])
    vector_by_kind = {
        "toward_upper": direction["unit_direction"],
        "toward_lower": direction["unit_direction"],
        "orthogonal": orthogonal,
        "gaussian": gaussian,
    }
    print(
        "direction: "
        f"train_pair_acc={direction['train_pair_directional_accuracy']:.3f} "
        f"margin_mean={direction['train_pair_margin_mean']:.3f} "
        f"proj_std={projection_std:.3f}",
        flush=True,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for prompt_id, question in enumerate(eval_questions):
            prompt_text = prompt_for(question, BASELINE_INSTRUCTION)
            rendered = render_chat_text(
                tokenizer,
                system=SYSTEM_PROMPT,
                user=prompt_text,
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            print(f"eval {prompt_id + 1}/{len(eval_questions)} prompt_tokens={len(token_ids)} question={question!r}", flush=True)
            for condition in conditions:
                hook_state = {"calls": 0, "applications": 0}
                if condition.direction_kind is None:
                    new_ids, reply = generate_one(
                        model=model,
                        token_ids=token_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=args.do_sample,
                        stop_at_eos=args.stop_at_eos,
                        cache_dtype=dtype,
                    )
                    signed_delta = 0.0
                else:
                    signed_delta = condition.signed_strength * projection_std
                    hook_fn, hook_state = make_steering_hook(
                        vector=vector_by_kind[condition.direction_kind],
                        delta=signed_delta,
                        scope=args.intervention_scope,
                    )
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        new_ids, reply = generate_one(
                            model=model,
                            token_ids=token_ids,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            do_sample=args.do_sample,
                            stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype,
                        )
                metrics = alpha_case_metrics(reply, len(new_ids))
                row = {
                    "schema_version": 1,
                    "prompt_id": int(prompt_id),
                    "question": question,
                    "prompt_text": prompt_text,
                    "model": args.model,
                    "model_key": args.model_key,
                    "task": "format_positive_control",
                    "target_variable": "positive_control_behavior",
                    "layer": args.layer,
                    "hook_name": hook_name,
                    "condition": condition.label,
                    "direction_kind": condition.direction_kind,
                    "strength_sd": condition.strength_sd,
                    "signed_strength_sd": condition.signed_strength,
                    "intervention_delta_l2": abs(float(signed_delta)),
                    "intervention_scope": args.intervention_scope,
                    "hook_calls": int(hook_state["calls"]),
                    "hook_applications": int(hook_state["applications"]),
                    "prompt_token_count": len(token_ids),
                    "model_output": reply,
                    **metrics,
                }
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"  {condition.label}: tokens={row['generated_token_count']} "
                    f"upper_frac={row['uppercase_alpha_fraction']:.3f} hooks={hook_state['applications']}/{hook_state['calls']}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_rows(rows, min_uppercase_delta=args.min_uppercase_delta)
    experiment_report = make_experiment_report(
        model=args.model_key,
        task="format_positive_control",
        target_variable="positive_control_behavior",
        split="synthetic_neutral_prompts",
        site_or_layer=f"L{args.layer}",
        method="mean_difference_casing_steering",
        representation_type="raw_direction",
        result_type="causal",
        controls=["regenerated_baseline", "orthogonal_direction", "matched_gaussian_noise", "positive_control"],
        n=len(eval_questions),
        baseline_metrics={
            "direction_training": {key: value for key, value in direction.items() if not isinstance(value, np.ndarray)},
            "baseline_condition": summary["by_condition"].get("baseline", {}),
            "train_prompt_token_counts": prompt_token_counts,
        },
        intervention_metrics={"by_condition": summary["by_condition"]},
        paired_flips={"paired_deltas_vs_baseline": summary["paired_deltas_vs_baseline"]},
        parse_fail_rate=0.0,
        matched_noise_summary=summary["matched_noise_summary"],
        causal_abstraction_claim=(
            "Auxiliary positive-control machinery check: this tests whether the Gemma 3 27B L45 hook, "
            "generation path, regenerated baseline, orthogonal control, and matched-Gaussian control can move "
            "a simple casing/output-format behavior. It is not an InAbHyD reasoning-variable claim."
        ),
        notes=[
            "Pass/fail thresholds are engineering gates, not manuscript claims.",
            "This format gate was added after the verbosity gate saturated at the generation token cap.",
        ],
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_positive_control_format.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": "format_positive_control",
        "target_variable": "positive_control_behavior",
        "layer": args.layer,
        "hook_name": hook_name,
        "out_jsonl": str(args.out_jsonl),
        "direction_output": str(args.direction_output),
        "prompts": {
            "system": SYSTEM_PROMPT,
            "upper_instruction": UPPER_INSTRUCTION,
            "lower_instruction": LOWER_INSTRUCTION,
            "baseline_instruction": BASELINE_INSTRUCTION,
            "train_questions": train_questions,
            "eval_questions": eval_questions,
        },
        "generation": {
            "conditions": [condition.__dict__ for condition in conditions],
            "strengths_sd": list(parse_float_list(args.strengths)),
            "intervention_scope": args.intervention_scope,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "stop_at_eos": args.stop_at_eos,
            "min_uppercase_delta": args.min_uppercase_delta,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "direction": {key: value for key, value in direction.items() if not isinstance(value, np.ndarray)},
        "summary": summary,
        "experiment_report": experiment_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"wrote {args.direction_output}", flush=True)
    print(f"positive_control_passed={summary['matched_noise_summary']['passed_positive_control_gate']}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
