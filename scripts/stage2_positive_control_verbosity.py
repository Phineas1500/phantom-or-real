#!/usr/bin/env python3
"""Run a Gemma 3 verbosity steering positive control."""

from __future__ import annotations

import argparse
import json
import os
import re
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
from src.stage2_causal_abstraction import make_experiment_report  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    make_gaussian_unit_direction,
    make_orthogonal_unit_direction,
    parse_float_list,
    strength_label,
)


SYSTEM_PROMPT = "You are a helpful assistant."
SHORT_INSTRUCTION = "Answer in one word or one short phrase."
LONG_INSTRUCTION = "Answer in at least three sentences with a brief explanation."
BASELINE_INSTRUCTION = "Answer naturally."

TRAIN_QUESTIONS = [
    "What is a comet?",
    "Why do leaves change color?",
    "What is photosynthesis?",
    "What is a prime number?",
    "Why does bread rise?",
    "What causes tides?",
    "What is a microscope used for?",
    "Why is sleep important?",
    "What is evaporation?",
    "What is a glacier?",
    "Why do magnets attract some metals?",
    "What is a vaccine?",
    "What is a peninsula?",
    "Why do people use maps?",
    "What is friction?",
    "What is a fossil?",
    "Why do shadows change length?",
    "What is a circuit?",
    "What is a metaphor?",
    "Why does metal rust?",
    "What is a habitat?",
    "Why does the moon look different over time?",
    "What is a budget?",
    "What is condensation?",
    "Why do airplanes need wings?",
    "What is a democracy?",
    "What is a decimal?",
    "Why do seasons happen?",
]

EVAL_QUESTIONS = [
    "What is a rainbow?",
    "Why do boats float?",
    "What is an ecosystem?",
    "What is gravity?",
    "Why do cooks use salt?",
    "What is a telescope?",
    "Why does ice melt?",
    "What is a river delta?",
    "What is an algorithm?",
    "Why do plants need sunlight?",
    "What is a thermometer?",
    "Why does exercise help health?",
    "What is a library?",
    "Why do batteries run out?",
    "What is a compass?",
    "What is an island?",
]


@dataclass(frozen=True)
class VerbosityCondition:
    label: str
    direction_kind: str | None
    strength_sd: float
    signed_strength: float


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
    return str(value)


def prompt_for(question: str, instruction: str) -> str:
    return f"{question}\n\n{instruction}"


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    pieces = [part for part in re.split(r"[.!?]+", stripped) if part.strip()]
    return max(1, len(pieces))


def style_metrics(text: str, token_count: int, *, short_token_threshold: int, long_token_threshold: int) -> dict[str, Any]:
    words = word_count(text)
    sentences = sentence_count(text)
    return {
        "generated_token_count": int(token_count),
        "output_char_count": len(text),
        "output_word_count": int(words),
        "output_sentence_count": int(sentences),
        "is_short_style": bool(token_count <= short_token_threshold and sentences <= 1),
        "is_long_style": bool(token_count >= long_token_threshold or sentences >= 3),
    }


def make_conditions(kinds: list[str], strengths: tuple[float, ...]) -> list[VerbosityCondition]:
    allowed = {"baseline", "toward_long", "toward_short", "orthogonal", "gaussian"}
    unknown = sorted(set(kinds) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    plan: list[VerbosityCondition] = []
    if "baseline" in kinds:
        plan.append(VerbosityCondition("baseline", None, 0.0, 0.0))
    for kind in ("toward_long", "toward_short", "orthogonal", "gaussian"):
        if kind not in kinds:
            continue
        for strength in strengths:
            if strength <= 0:
                raise ValueError("verbosity steering strengths must be positive")
            signed = -float(strength) if kind == "toward_short" else float(strength)
            plan.append(VerbosityCondition(f"{kind}_{strength_label(strength)}", kind, float(strength), signed))
    return plan


def activation_for_prompt(*, model, tokenizer, hook_name: str, prompt_text: str, model_name: str) -> tuple[np.ndarray, int]:
    rendered = render_chat_text(
        tokenizer,
        system=SYSTEM_PROMPT,
        user=prompt_text,
        model_name=model_name,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    tokens = torch.tensor([token_ids], dtype=torch.long, device=input_device_for_model(model))
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        captured["x"] = act[:, -1, :].detach().to(torch.float32).cpu()
        return act

    with torch.inference_mode(), model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        model(tokens, return_type="logits")
    if "x" not in captured:
        raise RuntimeError(f"hook {hook_name} did not capture an activation")
    return captured["x"][0].numpy().astype(np.float32), len(token_ids)


def make_direction(*, long_acts: np.ndarray, short_acts: np.ndarray) -> dict[str, Any]:
    if long_acts.shape != short_acts.shape:
        raise ValueError(f"long/short activation shape mismatch: {long_acts.shape} vs {short_acts.shape}")
    raw = long_acts.mean(axis=0).astype(np.float64) - short_acts.mean(axis=0).astype(np.float64)
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        raise ValueError("long-short mean direction has zero norm")
    unit = (raw / norm).astype(np.float32)
    all_acts = np.concatenate([long_acts, short_acts], axis=0).astype(np.float64)
    projections = all_acts @ unit.astype(np.float64)
    projection_std = float(projections.std(ddof=0))
    if projection_std == 0.0:
        raise ValueError("train projection has zero standard deviation")
    long_proj = long_acts.astype(np.float64) @ unit.astype(np.float64)
    short_proj = short_acts.astype(np.float64) @ unit.astype(np.float64)
    pair_margins = long_proj - short_proj
    return {
        "unit_direction": unit,
        "raw_direction": raw.astype(np.float32),
        "raw_direction_norm": norm,
        "train_projection_mean": float(projections.mean()),
        "train_projection_std": projection_std,
        "train_long_projection_mean": float(long_proj.mean()),
        "train_short_projection_mean": float(short_proj.mean()),
        "train_pair_margin_mean": float(pair_margins.mean()),
        "train_pair_margin_min": float(pair_margins.min()),
        "train_pair_margin_max": float(pair_margins.max()),
        "train_pair_directional_accuracy": float((pair_margins > 0).mean()),
        "train_n_pairs": int(long_acts.shape[0]),
        "d_model": int(long_acts.shape[1]),
    }


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


def generate_one(*, model, token_ids: list[int], max_new_tokens: int, temperature: float, do_sample: bool, stop_at_eos: bool, cache_dtype: torch.dtype) -> tuple[list[int], str]:
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("model has no tokenizer")
    tokens = torch.tensor([token_ids], dtype=torch.long, device=input_device_for_model(model))
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
    return new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    condition_summary: dict[str, Any] = {}
    for condition, subset in sorted(by_condition.items()):
        n = len(subset)
        condition_summary[condition] = {
            "n": n,
            "mean_generated_tokens": float(np.mean([row["generated_token_count"] for row in subset])) if n else None,
            "mean_words": float(np.mean([row["output_word_count"] for row in subset])) if n else None,
            "mean_sentences": float(np.mean([row["output_sentence_count"] for row in subset])) if n else None,
            "short_style_rate": float(np.mean([row["is_short_style"] for row in subset])) if n else None,
            "long_style_rate": float(np.mean([row["is_long_style"] for row in subset])) if n else None,
        }

    baselines = {row["prompt_id"]: row for row in rows if row["condition"] == "baseline"}
    paired: dict[str, Any] = {}
    control_abs_deltas: list[float] = []
    for condition, subset in sorted(by_condition.items()):
        if condition == "baseline":
            continue
        pairs = [(baselines[row["prompt_id"]], row) for row in subset if row["prompt_id"] in baselines]
        token_deltas = [row["generated_token_count"] - base["generated_token_count"] for base, row in pairs]
        sentence_deltas = [row["output_sentence_count"] - base["output_sentence_count"] for base, row in pairs]
        if condition.startswith(("orthogonal", "gaussian")):
            control_abs_deltas.extend(abs(float(delta)) for delta in token_deltas)
        paired[condition] = {
            "paired_n": len(pairs),
            "mean_token_delta": float(np.mean(token_deltas)) if token_deltas else None,
            "median_token_delta": float(np.median(token_deltas)) if token_deltas else None,
            "mean_sentence_delta": float(np.mean(sentence_deltas)) if sentence_deltas else None,
            "token_delta_gt0_rate": float(np.mean([delta > 0 for delta in token_deltas])) if token_deltas else None,
            "token_delta_lt0_rate": float(np.mean([delta < 0 for delta in token_deltas])) if token_deltas else None,
            "long_style_false_to_true": int(
                sum((not base["is_long_style"]) and row["is_long_style"] for base, row in pairs)
            ),
            "short_style_false_to_true": int(
                sum((not base["is_short_style"]) and row["is_short_style"] for base, row in pairs)
            ),
        }

    control_mean = float(np.mean(control_abs_deltas)) if control_abs_deltas else None
    control_std = float(np.std(control_abs_deltas, ddof=1)) if len(control_abs_deltas) > 1 else None
    for condition, stats in paired.items():
        mean_delta = stats.get("mean_token_delta")
        if mean_delta is None or control_mean is None or not control_std:
            stats["effect_over_control_sigma"] = None
            continue
        directional_effect = -mean_delta if condition.startswith("toward_short") else mean_delta
        stats["effect_over_control_sigma"] = float((directional_effect - control_mean) / control_std)

    best_long = max(
        (stats for condition, stats in paired.items() if condition.startswith("toward_long")),
        key=lambda stats: stats.get("effect_over_control_sigma") if stats.get("effect_over_control_sigma") is not None else -1e9,
        default=None,
    )
    best_short = max(
        (stats for condition, stats in paired.items() if condition.startswith("toward_short")),
        key=lambda stats: stats.get("effect_over_control_sigma") if stats.get("effect_over_control_sigma") is not None else -1e9,
        default=None,
    )
    long_pass = bool(
        best_long
        and best_long.get("effect_over_control_sigma") is not None
        and best_long["effect_over_control_sigma"] >= 2.0
        and best_long.get("token_delta_gt0_rate", 0.0) >= 0.70
    )
    short_pass = bool(
        best_short
        and best_short.get("effect_over_control_sigma") is not None
        and best_short["effect_over_control_sigma"] >= 2.0
        and best_short.get("token_delta_lt0_rate", 0.0) >= 0.70
    )

    return {
        "by_condition": condition_summary,
        "paired_deltas_vs_baseline": paired,
        "matched_noise_summary": {
            "control_abs_token_delta_mean": control_mean,
            "control_abs_token_delta_std": control_std,
            "control_abs_token_delta_n": len(control_abs_deltas),
            "best_toward_long": best_long,
            "best_toward_short": best_short,
            "passed_positive_control_gate": bool(long_pass or short_pass),
            "passed_toward_long": long_pass,
            "passed_toward_short": short_pass,
        },
    }


def save_direction(path: Path, *, direction: dict[str, Any], orthogonal: np.ndarray, gaussian: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_direction=direction["unit_direction"],
        raw_direction=direction["raw_direction"],
        orthogonal_direction=orthogonal.astype(np.float32),
        gaussian_direction=gaussian.astype(np.float32),
        train_projection_std=np.array(direction["train_projection_std"], dtype=np.float32),
        train_projection_mean=np.array(direction["train_projection_mean"], dtype=np.float32),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--train-limit", type=int, default=24)
    parser.add_argument("--eval-limit", type=int, default=12)
    parser.add_argument("--orthogonal-seed", type=int, default=20260604)
    parser.add_argument("--gaussian-seed", type=int, default=20260605)
    parser.add_argument("--conditions", default="baseline,toward_long,toward_short,orthogonal,gaussian")
    parser.add_argument("--strengths", default="0.5,1,2")
    parser.add_argument("--intervention-scope", choices=("prompt_only", "last_token_each_forward"), default="last_token_each_forward")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--short-token-threshold", type=int, default=12)
    parser.add_argument("--long-token-threshold", type=int, default=48)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/positive_control/verbosity_gemma3_27b_l45.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/positive_control/verbosity_gemma3_27b_l45_direction.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/positive_control_verbosity_gemma3_27b_l45.json"))
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

    print("Stage 2 verbosity positive-control steering", flush=True)
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

    long_acts = []
    short_acts = []
    prompt_token_counts = []
    for idx, question in enumerate(train_questions, start=1):
        short_act, short_tokens = activation_for_prompt(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            prompt_text=prompt_for(question, SHORT_INSTRUCTION),
            model_name=args.model,
        )
        long_act, long_tokens = activation_for_prompt(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            prompt_text=prompt_for(question, LONG_INSTRUCTION),
            model_name=args.model,
        )
        short_acts.append(short_act)
        long_acts.append(long_act)
        prompt_token_counts.append({"question_index": idx, "short_tokens": short_tokens, "long_tokens": long_tokens})
        print(f"train {idx}/{len(train_questions)} short_tokens={short_tokens} long_tokens={long_tokens}", flush=True)

    direction = make_direction(long_acts=np.stack(long_acts), short_acts=np.stack(short_acts))
    orthogonal = make_orthogonal_unit_direction(direction["unit_direction"], seed=args.orthogonal_seed)
    gaussian = make_gaussian_unit_direction(direction["unit_direction"], seed=args.gaussian_seed)
    save_direction(args.direction_output, direction=direction, orthogonal=orthogonal, gaussian=gaussian)
    projection_std = float(direction["train_projection_std"])
    vector_by_kind = {
        "toward_long": direction["unit_direction"],
        "toward_short": direction["unit_direction"],
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
                metrics = style_metrics(
                    reply,
                    len(new_ids),
                    short_token_threshold=args.short_token_threshold,
                    long_token_threshold=args.long_token_threshold,
                )
                row = {
                    "schema_version": 1,
                    "prompt_id": int(prompt_id),
                    "question": question,
                    "prompt_text": prompt_text,
                    "model": args.model,
                    "model_key": args.model_key,
                    "task": "verbosity_positive_control",
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
                    f"sentences={row['output_sentence_count']} hooks={hook_state['applications']}/{hook_state['calls']}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_rows(rows)
    experiment_report = make_experiment_report(
        model=args.model_key,
        task="verbosity_positive_control",
        target_variable="positive_control_behavior",
        split="synthetic_neutral_prompts",
        site_or_layer=f"L{args.layer}",
        method="mean_difference_verbosity_steering",
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
            "a simple output-style behavior. It is not an InAbHyD reasoning-variable claim."
        ),
        notes=[
            "Pass/fail thresholds are engineering gates, not manuscript claims.",
            "A passing result permits interpretation of later correctness-steering nulls as intervention-stack-aware nulls.",
        ],
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_positive_control_verbosity.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": "verbosity_positive_control",
        "target_variable": "positive_control_behavior",
        "layer": args.layer,
        "hook_name": hook_name,
        "out_jsonl": str(args.out_jsonl),
        "direction_output": str(args.direction_output),
        "prompts": {
            "system": SYSTEM_PROMPT,
            "short_instruction": SHORT_INSTRUCTION,
            "long_instruction": LONG_INSTRUCTION,
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
            "short_token_threshold": args.short_token_threshold,
            "long_token_threshold": args.long_token_threshold,
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
