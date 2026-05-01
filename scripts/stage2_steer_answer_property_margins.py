#!/usr/bin/env python3
"""Measure answer-property steering with logprob margins and MCQ choices."""

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
load_dotenv()
if "BD_PATH" not in os.environ:
    scratch_bd = Path(f"/scratch/scholar/{os.environ.get('USER', '')}/beyond-deduction")
    if scratch_bd.exists():
        os.environ["BD_PATH"] = str(scratch_bd)

from scripts.stage2_steer_answer_property_direction import (  # noqa: E402
    answer_label_for_row,
    answer_sign,
    extract_answer_content_from_hypothesis,
    gold_answer_content,
    make_answer_condition_plan,
    make_steering_hook,
    parse_condition_kinds,
    save_direction_artifact,
    select_answer_rows,
    serializable_direction_summary,
    singularize_concept,
    train_answer_probe_direction,
)
from scripts.stage2_steer_forced_choice_direction import (  # noqa: E402
    ForcedChoicePrompt,
    GENERATION_REQUEST_RE,
    build_forced_choice_prompt,
    generate_one,
    parse_choice,
)
from src.activations import (  # noqa: E402
    input_device_for_model,
    load_tl_model,
    render_chat_text,
    validate_hooks,
)
from src.gemma3_parse import parse_hypotheses  # noqa: E402
from src.stage2_probes import read_jsonl, read_split_assignments  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    make_orthogonal_unit_direction,
    parse_float_list,
    parse_int_list,
)


@dataclass(frozen=True)
class ScoredSequence:
    text: str
    token_ids: list[int]
    logprob: float
    mean_logprob: float


@dataclass(frozen=True)
class HypothesisParts:
    subject: str | None
    predicate: str | None
    negated: bool | None


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


def output_row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["source_row_index"]), str(row["condition"])


def singularize_margin_name(value: str | None) -> str | None:
    if value is None:
        return None
    raw = singularize_concept(value)
    if raw is None:
        return None
    if raw.endswith("uses") and len(raw) > 4:
        return raw[:-2]
    return raw


def hypothesis_parts_from_text(text: str | None) -> HypothesisParts:
    if not text:
        return HypothesisParts(subject=None, predicate=None, negated=None)
    import re

    cleaned = re.sub(r"\s+", " ", text.strip()).strip(" .;:")
    patterns = (
        r"^(?:every|each|all)\s+(?P<subject>[A-Za-z]+)\s+(?:is|are)\s+(?P<neg>not\s+)?(?P<predicate>[A-Za-z]+)$",
        r"^(?P<subject>[A-Za-z]+)\s+(?:is|are)\s+(?P<neg>not\s+)?(?P<predicate>[A-Za-z]+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return HypothesisParts(
                subject=singularize_margin_name(match.group("subject")),
                predicate=singularize_margin_name(match.group("predicate")),
                negated=bool(match.group("neg")),
            )
    content = extract_answer_content_from_hypothesis(text)
    return HypothesisParts(subject=None, predicate=content.predicate, negated=content.negated)


def gold_hypothesis_parts(row: dict[str, Any]) -> HypothesisParts:
    hypothesis = row["ontology_fol_structured"]["hypothesis"]
    return HypothesisParts(
        subject=singularize_margin_name(hypothesis.get("subject")),
        predicate=singularize_margin_name(hypothesis.get("predicate")),
        negated=bool(hypothesis.get("negated")),
    )


def same_hypothesis_parts(left: HypothesisParts, right: HypothesisParts) -> bool:
    return (
        left.subject == right.subject
        and left.predicate == right.predicate
        and left.negated == right.negated
    )


def normalize_candidate_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return stripped if stripped.endswith(".") else stripped + "."


def emitted_wrong_foil(row: dict[str, Any]) -> str | None:
    gold_parts = gold_hypothesis_parts(row)
    candidates = parse_hypotheses(row.get("model_output", "") or "")
    if not candidates:
        candidates = [line.strip() for line in (row.get("model_output", "") or "").splitlines() if line.strip()]
    for candidate in candidates:
        candidate = normalize_candidate_text(candidate)
        parts = hypothesis_parts_from_text(candidate)
        if parts.predicate is None or parts.negated is None:
            continue
        if same_hypothesis_parts(parts, gold_parts):
            continue
        return candidate
    return None


def build_margin_prompt(
    row: dict[str, Any],
    *,
    row_index: int,
    option_seed: int,
    foil_source: str,
) -> ForcedChoicePrompt:
    base = build_forced_choice_prompt(row, row_index=row_index, option_seed=option_seed)
    if foil_source == "opposite_polarity":
        return base
    if foil_source != "stage1_model_output":
        raise ValueError(f"unknown foil source {foil_source!r}")

    foil = emitted_wrong_foil(row)
    if foil is None:
        raise ValueError(f"row {row_index} has no emitted wrong foil")
    if base.gold_choice == "A":
        option_a, option_b = base.gold_hypothesis, foil
    else:
        option_a, option_b = foil, base.gold_hypothesis
    base_prompt = GENERATION_REQUEST_RE.sub("", row["prompt_text"]).strip()
    user = (
        f"{base_prompt}\n\n"
        "Which hypothesis best explains the observations?\n"
        f"(A) {option_a}\n"
        f"(B) {option_b}\n\n"
        "Answer with exactly one option: (A) or (B)."
    )
    return ForcedChoicePrompt(
        system=base.system,
        user=user,
        label=base.label,
        gold_choice=base.gold_choice,
        option_a=option_a,
        option_b=option_b,
        gold_hypothesis=base.gold_hypothesis,
        foil_hypothesis=foil,
    )


def select_margin_rows(
    *,
    source_rows: list[dict[str, Any]],
    jsonl_path: Path,
    splits_path: Path,
    source_file: str,
    split_family: str,
    heights: list[int],
    per_height_label: int,
    selection_seed: int,
    label_source: str,
    answer_target: str,
    positive_predicate: str | None,
    negative_predicate: str | None,
    foil_source: str,
    baseline_incorrect_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import random
    from collections import defaultdict

    if not baseline_incorrect_only and foil_source == "opposite_polarity":
        selected, summary = select_answer_rows(
            jsonl_path=jsonl_path,
            splits_path=splits_path,
            source_file=source_file,
            split_family=split_family,
            heights=heights,
            per_height_label=per_height_label,
            seed=selection_seed,
            label_source=label_source,
            answer_target=answer_target,
            positive_predicate=positive_predicate,
            negative_predicate=negative_predicate,
            drop_parse_failed=True,
        )
        summary["selection_mode"] = "height_x_original_correctness"
        summary["foil_source"] = foil_source
        summary["baseline_incorrect_only"] = baseline_incorrect_only
        return selected, summary

    assignments = read_split_assignments(splits_path)
    height_set = set(heights)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    available_counts: dict[str, int] = defaultdict(int)
    skipped_no_answer_label = 0
    skipped_correct = 0
    skipped_parse_failed = 0
    skipped_no_foil = 0

    for row_index, row in enumerate(source_rows):
        height = row.get("height")
        if height not in height_set:
            continue
        if row.get("parse_failed"):
            skipped_parse_failed += 1
            continue
        if baseline_incorrect_only and row.get("is_correct_strong"):
            skipped_correct += 1
            continue
        assignment = assignments.get((source_file, row_index))
        if assignment is None or assignment.get(f"{split_family}_split") != "test":
            continue
        label = answer_label_for_row(
            row,
            label_source=label_source,
            answer_target=answer_target,
            positive_predicate=positive_predicate,
            negative_predicate=negative_predicate,
        )
        if label is None:
            skipped_no_answer_label += 1
            continue
        foil = emitted_wrong_foil(row) if foil_source == "stage1_model_output" else None
        if foil_source == "stage1_model_output" and foil is None:
            skipped_no_foil += 1
            continue
        labeled = dict(row)
        labeled["row_index"] = row_index
        labeled["answer_label"] = int(label)
        labeled["margin_foil_hypothesis"] = foil
        groups[(int(height), int(label))].append(labeled)
        available_counts[f"h{height}_label{label}"] += 1

    selected: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    for height in heights:
        for label in (0, 1):
            rows = list(groups[(height, label)])
            if len(rows) < per_height_label:
                missing[f"h{height}_label{label}"] = len(rows)
                continue
            rng = random.Random(selection_seed + height * 1009 + label * 9176)
            rng.shuffle(rows)
            selected.extend(rows[:per_height_label])
    if missing:
        raise ValueError(
            f"not enough rows for margin subset; requested {per_height_label}, counts={missing}"
        )
    selected.sort(key=lambda row: (int(row["height"]), int(row["answer_label"]), int(row["row_index"])))
    return selected, {
        "source_file": source_file,
        "split_family": split_family,
        "split": "test",
        "heights": heights,
        "per_height_label": per_height_label,
        "drop_parse_failed": True,
        "seed": selection_seed,
        "selection_mode": "height_x_answer_label",
        "foil_source": foil_source,
        "baseline_incorrect_only": baseline_incorrect_only,
        "available_counts": dict(sorted(available_counts.items())),
        "selected_rows": len(selected),
        "selected_counts": {
            f"h{height}_label{label}": sum(
                1
                for row in selected
                if int(row.get("height")) == height and int(row.get("answer_label")) == label
            )
            for height in heights
            for label in (0, 1)
        },
        "skipped_no_answer_label": skipped_no_answer_label,
        "skipped_correct": skipped_correct,
        "skipped_parse_failed": skipped_parse_failed,
        "skipped_no_foil": skipped_no_foil,
        "answer_label_source": label_source,
        "answer_target": answer_target,
        "positive_predicate": positive_predicate,
        "negative_predicate": negative_predicate,
    }


def load_resume_rows(
    path: Path,
    expected_keys: set[tuple[int, str]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], int, int]:
    rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    malformed_rows = 0
    ignored_rows = 0
    if not path.exists():
        return rows_by_key, malformed_rows, ignored_rows

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = output_row_key(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed_rows += 1
                continue
            if key not in expected_keys:
                ignored_rows += 1
                continue
            rows_by_key[key] = row
    return rows_by_key, malformed_rows, ignored_rows


def make_scoring_hook(
    *,
    vector: np.ndarray,
    delta: float,
    prompt_len: int,
    candidate_len: int,
    scope: str,
) -> tuple[Any, dict[str, int]]:
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    state = {"calls": 0, "positions": 0}

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        state["calls"] += 1
        key = (str(act.device), act.dtype)
        direction_tensor = cache.get(key)
        if direction_tensor is None:
            direction_tensor = torch.as_tensor(vector, device=act.device, dtype=act.dtype)
            cache[key] = direction_tensor

        start = prompt_len - 1
        if scope == "prompt_only":
            end = start + 1
        elif scope == "last_token_each_forward":
            end = start + candidate_len
        else:
            raise ValueError(f"unknown intervention scope {scope!r}")
        if start < 0 or end > act.shape[1]:
            raise ValueError(
                f"invalid scoring hook span start={start} end={end} act_len={act.shape[1]}"
            )
        act[:, start:end, :] = act[:, start:end, :] + float(delta) * direction_tensor
        state["positions"] += end - start
        return act

    return hook_fn, state


def score_sequence_logprob(
    *,
    model,
    hook_name: str,
    prompt_token_ids: list[int],
    candidate_text: str,
    vector: np.ndarray | None,
    delta: float,
    scope: str,
) -> tuple[ScoredSequence, dict[str, int]]:
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
    hook_state = {"calls": 0, "positions": 0}

    with torch.inference_mode():
        if vector is None or delta == 0.0:
            logits = model(tokens, return_type="logits", prepend_bos=False)
        else:
            hook_fn, hook_state = make_scoring_hook(
                vector=vector,
                delta=delta,
                prompt_len=len(prompt_token_ids),
                candidate_len=len(candidate_ids),
                scope=scope,
            )
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
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

    return (
        ScoredSequence(
            text=candidate_text,
            token_ids=list(candidate_ids),
            logprob=total,
            mean_logprob=total / len(candidate_ids),
        ),
        hook_state,
    )


def summarize_margin_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    baselines = {int(row["source_row_index"]): row for row in rows if row["condition"] == "baseline"}
    by_condition_summary: dict[str, Any] = {}
    margin_deltas: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        n = len(condition_rows)
        valid_choices = [row for row in condition_rows if row.get("mcq_parsed_choice") is not None]
        by_condition_summary[condition] = {
            "n": n,
            "mean_original_margin": (
                sum(float(row["original_margin_gold_minus_foil"]) for row in condition_rows) / n if n else None
            ),
            "mean_mcq_margin": (
                sum(float(row["mcq_margin_gold_minus_foil"]) for row in condition_rows) / n if n else None
            ),
            "mcq_choice_accuracy": (
                sum(bool(row["mcq_is_correct_choice"]) for row in condition_rows) / n if n else None
            ),
            "mcq_parse_fail_rate": (
                sum(bool(row["mcq_choice_parse_failed"]) for row in condition_rows) / n if n else None
            ),
            "mcq_parsed_choice_counts": {
                choice: sum(row.get("mcq_parsed_choice") == choice for row in valid_choices)
                for choice in ("A", "B")
            },
        }
        if condition == "baseline":
            continue
        paired = [
            (baselines[int(row["source_row_index"])], row)
            for row in condition_rows
            if int(row["source_row_index"]) in baselines
        ]
        if not paired:
            continue
        original_deltas = [
            float(steered["original_margin_gold_minus_foil"])
            - float(base["original_margin_gold_minus_foil"])
            for base, steered in paired
        ]
        mcq_deltas = [
            float(steered["mcq_margin_gold_minus_foil"]) - float(base["mcq_margin_gold_minus_foil"])
            for base, steered in paired
        ]
        margin_deltas[condition] = {
            "paired_n": len(paired),
            "mean_original_margin_delta": sum(original_deltas) / len(original_deltas),
            "mean_mcq_margin_delta": sum(mcq_deltas) / len(mcq_deltas),
            "original_margin_improved": int(sum(delta > 0.0 for delta in original_deltas)),
            "original_margin_worsened": int(sum(delta < 0.0 for delta in original_deltas)),
            "mcq_margin_improved": int(sum(delta > 0.0 for delta in mcq_deltas)),
            "mcq_margin_worsened": int(sum(delta < 0.0 for delta in mcq_deltas)),
            "mcq_choice_false_to_true": int(
                sum((not base["mcq_is_correct_choice"]) and steered["mcq_is_correct_choice"] for base, steered in paired)
            ),
            "mcq_choice_true_to_false": int(
                sum(base["mcq_is_correct_choice"] and (not steered["mcq_is_correct_choice"]) for base, steered in paired)
            ),
            "mcq_choice_changed": int(
                sum(base.get("mcq_parsed_choice") != steered.get("mcq_parsed_choice") for base, steered in paired)
            ),
        }

    return {
        "by_condition": by_condition_summary,
        "margin_deltas_vs_baseline": margin_deltas,
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
    parser.add_argument("--answer-label-source", choices=("gold", "stage1_model_output"), default="gold")
    parser.add_argument(
        "--answer-target",
        choices=("polarity", "predicate_pair", "predicate_one_vs_rest"),
        default="polarity",
    )
    parser.add_argument("--positive-predicate", default=None)
    parser.add_argument("--negative-predicate", default=None)
    parser.add_argument("--option-seed", type=int, default=20260430)
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
    parser.add_argument(
        "--foil-source",
        choices=("opposite_polarity", "stage1_model_output"),
        default="opposite_polarity",
        help="Use the old opposite-polarity foil or the model's emitted wrong Stage 1 hypothesis.",
    )
    parser.add_argument(
        "--baseline-incorrect-only",
        action="store_true",
        help="Restrict row selection to Stage 1 rows that were parsed but originally incorrect.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260427)
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
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/steering/answer_property_margins_27b_l45_polarity_smoke.jsonl"),
    )
    parser.add_argument(
        "--direction-output",
        type=Path,
        default=Path("results/stage2/steering/answer_property_margins_27b_l45_polarity_direction.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/answer_property_margins_27b_l45_polarity_smoke.json"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    source_file = str(args.jsonl)
    source_rows = read_jsonl(args.jsonl)
    heights = parse_int_list(args.heights)
    strengths = parse_float_list(args.strengths)
    condition_plan = make_answer_condition_plan(
        condition_kinds=parse_condition_kinds(args.conditions),
        strengths=strengths,
    )
    dtype = torch_dtype(args.dtype)
    activation_prefix = (
        args.activation_prefix
        if args.activation_prefix is not None
        else args.activation_dir / f"{args.model_key}_{args.task}_L{args.layer}"
    )
    activation_path = activation_prefix.with_suffix(".safetensors")
    sidecar_path = activation_prefix.with_suffix(".example_ids.jsonl")

    selected_rows, selection_summary = select_margin_rows(
        source_rows=source_rows,
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=heights,
        per_height_label=args.per_height_label,
        selection_seed=args.selection_seed,
        label_source=args.answer_label_source,
        answer_target=args.answer_target,
        positive_predicate=args.positive_predicate,
        negative_predicate=args.negative_predicate,
        foil_source=args.foil_source,
        baseline_incorrect_only=args.baseline_incorrect_only,
    )
    selection_summary["selected_answer_label_counts"] = {
        str(label): sum(int(row["answer_label"]) == label for row in selected_rows)
        for label in (0, 1)
    }

    if args.dry_run:
        first = selected_rows[0]
        first_prompt = build_margin_prompt(
            first,
            row_index=int(first["row_index"]),
            option_seed=args.option_seed,
            foil_source=args.foil_source,
        )
        print(json.dumps({
            "selected": selection_summary,
            "first_selected": {
                "row_index": first["row_index"],
                "height": first["height"],
                "answer_label": first["answer_label"],
                "original_correct": first["is_correct_strong"],
                "stage1_model_foil": first.get("margin_foil_hypothesis"),
                "gold_hypothesis": first_prompt.gold_hypothesis,
                "foil_hypothesis": first_prompt.foil_hypothesis,
                "mcq_user": first_prompt.user,
            },
        }, indent=2, sort_keys=True))
        return 0

    print("Stage 2 answer-property margin steering", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

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
    print(f"selected_rows={len(selected_rows)} selection={selection_summary}", flush=True)

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
    if existing_rows_by_key:
        for stage1_row in selected_rows:
            for condition in condition_plan:
                existing_row = existing_rows_by_key.get((int(stage1_row["row_index"]), condition.label))
                if existing_row is not None:
                    rows.append(existing_row)
        with args.out_jsonl.open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    projection_std = float(direction["train_projection_std"])
    vector_by_kind = {
        "toward_gold": direction["unit_direction"],
        "away_gold": direction["unit_direction"],
        "orthogonal": orthogonal_direction,
    }
    output_mode = "a" if args.resume else "w"
    with args.out_jsonl.open(output_mode) as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            source_row_index = int(stage1_row["row_index"])
            original_prompt_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            original_prompt_tokens = tokenizer(original_prompt_text, add_special_tokens=False)["input_ids"]
            forced_prompt = build_margin_prompt(
                stage1_row,
                row_index=source_row_index,
                option_seed=args.option_seed,
                foil_source=args.foil_source,
            )
            mcq_prompt_text = render_chat_text(
                tokenizer,
                system=forced_prompt.system,
                user=forced_prompt.user,
                model_name=args.model,
                add_generation_prompt=True,
            )
            mcq_prompt_tokens = tokenizer(mcq_prompt_text, add_special_tokens=False)["input_ids"]
            gold = gold_answer_content(stage1_row)
            print(
                f"row {row_idx}/{len(selected_rows)} source_row={source_row_index} "
                f"h={stage1_row['height']} original_correct={stage1_row['is_correct_strong']} "
                f"gold={gold.predicate}/negated={gold.negated} "
                f"orig_tokens={len(original_prompt_tokens)} mcq_tokens={len(mcq_prompt_tokens)}",
                flush=True,
            )

            for condition in condition_plan:
                row_key = (source_row_index, condition.label)
                if row_key in existing_rows_by_key:
                    existing_row = existing_rows_by_key[row_key]
                    print(
                        f"  {condition.label}: resume_skip "
                        f"orig_margin={existing_row.get('original_margin_gold_minus_foil')} "
                        f"mcq_margin={existing_row.get('mcq_margin_gold_minus_foil')} "
                        f"choice={existing_row.get('mcq_parsed_choice')}",
                        flush=True,
                    )
                    continue

                if condition.direction_kind is None:
                    signed_delta = 0.0
                    vector = None
                else:
                    sign_to_gold = answer_sign(int(stage1_row["answer_label"]))
                    if condition.direction_kind == "toward_gold":
                        signed_delta = sign_to_gold * condition.strength_sd * projection_std
                    elif condition.direction_kind == "away_gold":
                        signed_delta = -sign_to_gold * condition.strength_sd * projection_std
                    elif condition.direction_kind == "orthogonal":
                        signed_delta = sign_to_gold * condition.strength_sd * projection_std
                    else:
                        raise ValueError(f"unknown direction kind {condition.direction_kind!r}")
                    vector = vector_by_kind[condition.direction_kind]

                original_gold, original_gold_hook = score_sequence_logprob(
                    model=model,
                    hook_name=hook_name,
                    prompt_token_ids=original_prompt_tokens,
                    candidate_text=forced_prompt.gold_hypothesis,
                    vector=vector,
                    delta=signed_delta,
                    scope=args.intervention_scope,
                )
                original_foil, original_foil_hook = score_sequence_logprob(
                    model=model,
                    hook_name=hook_name,
                    prompt_token_ids=original_prompt_tokens,
                    candidate_text=forced_prompt.foil_hypothesis,
                    vector=vector,
                    delta=signed_delta,
                    scope=args.intervention_scope,
                )
                mcq_gold_text = f"({forced_prompt.gold_choice})"
                mcq_foil_text = "(B)" if forced_prompt.gold_choice == "A" else "(A)"
                mcq_gold, mcq_gold_hook = score_sequence_logprob(
                    model=model,
                    hook_name=hook_name,
                    prompt_token_ids=mcq_prompt_tokens,
                    candidate_text=mcq_gold_text,
                    vector=vector,
                    delta=signed_delta,
                    scope=args.intervention_scope,
                )
                mcq_foil, mcq_foil_hook = score_sequence_logprob(
                    model=model,
                    hook_name=hook_name,
                    prompt_token_ids=mcq_prompt_tokens,
                    candidate_text=mcq_foil_text,
                    vector=vector,
                    delta=signed_delta,
                    scope=args.intervention_scope,
                )

                generation_hook_state = {"calls": 0, "applications": 0}
                if vector is None or signed_delta == 0.0:
                    new_ids, reply = generate_one(
                        model=model,
                        token_ids=mcq_prompt_tokens,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        stop_at_eos=args.stop_at_eos,
                        cache_dtype=dtype,
                    )
                else:
                    hook_fn, generation_hook_state = make_steering_hook(
                        vector=vector,
                        delta=signed_delta,
                        scope=args.intervention_scope,
                    )
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        new_ids, reply = generate_one(
                            model=model,
                            token_ids=mcq_prompt_tokens,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype,
                        )

                parsed_choice = parse_choice(reply)
                is_correct_choice = parsed_choice == forced_prompt.gold_choice
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
                    "original_prompt_token_count": len(original_prompt_tokens),
                    "mcq_prompt_token_count": len(mcq_prompt_tokens),
                    "gold_predicate": gold.predicate,
                    "gold_negated": gold.negated,
                    "gold_hypothesis": forced_prompt.gold_hypothesis,
                    "foil_hypothesis": forced_prompt.foil_hypothesis,
                    "foil_source": args.foil_source,
                    "baseline_incorrect_only": bool(args.baseline_incorrect_only),
                    "mcq_gold_choice": forced_prompt.gold_choice,
                    "mcq_option_a": forced_prompt.option_a,
                    "mcq_option_b": forced_prompt.option_b,
                    "original_gold_logprob": original_gold.logprob,
                    "original_foil_logprob": original_foil.logprob,
                    "original_gold_mean_logprob": original_gold.mean_logprob,
                    "original_foil_mean_logprob": original_foil.mean_logprob,
                    "original_margin_gold_minus_foil": original_gold.logprob - original_foil.logprob,
                    "original_gold_token_count": len(original_gold.token_ids),
                    "original_foil_token_count": len(original_foil.token_ids),
                    "mcq_gold_logprob": mcq_gold.logprob,
                    "mcq_foil_logprob": mcq_foil.logprob,
                    "mcq_gold_mean_logprob": mcq_gold.mean_logprob,
                    "mcq_foil_mean_logprob": mcq_foil.mean_logprob,
                    "mcq_margin_gold_minus_foil": mcq_gold.logprob - mcq_foil.logprob,
                    "mcq_gold_token_count": len(mcq_gold.token_ids),
                    "mcq_foil_token_count": len(mcq_foil.token_ids),
                    "original_scoring_hook_calls": (
                        original_gold_hook["calls"] + original_foil_hook["calls"]
                    ),
                    "original_scoring_hook_positions": (
                        original_gold_hook["positions"] + original_foil_hook["positions"]
                    ),
                    "mcq_scoring_hook_calls": mcq_gold_hook["calls"] + mcq_foil_hook["calls"],
                    "mcq_scoring_hook_positions": (
                        mcq_gold_hook["positions"] + mcq_foil_hook["positions"]
                    ),
                    "mcq_generated_token_count": len(new_ids),
                    "mcq_model_output": reply,
                    "mcq_parsed_choice": parsed_choice,
                    "mcq_choice_parse_failed": parsed_choice is None,
                    "mcq_is_correct_choice": bool(is_correct_choice),
                    "mcq_generation_hook_calls": int(generation_hook_state["calls"]),
                    "mcq_generation_hook_applications": int(generation_hook_state["applications"]),
                }
                rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"  {condition.label}: "
                    f"orig_margin={output_row['original_margin_gold_minus_foil']:.3f} "
                    f"mcq_margin={output_row['mcq_margin_gold_minus_foil']:.3f} "
                    f"choice={parsed_choice} correct={is_correct_choice}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_steer_answer_property_margins.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
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
            "option_seed": args.option_seed,
            "foil_source": args.foil_source,
            "baseline_incorrect_only": args.baseline_incorrect_only,
            "resume": args.resume,
            "resume_existing_rows": len(existing_rows_by_key),
            "resume_malformed_rows": resume_malformed_rows,
            "resume_ignored_rows": resume_ignored_rows,
        },
        "summary": summarize_margin_rows(rows),
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
