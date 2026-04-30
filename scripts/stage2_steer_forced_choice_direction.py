#!/usr/bin/env python3
"""Extract, probe, and steer Cox-style forced-choice ontology prompts."""

from __future__ import annotations

import argparse
import json
import os
import random
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
    EncodedExample,
    extract_residual_activations,
    input_device_for_model,
    load_tl_model,
    module_device_summary,
    render_chat_text,
    sha256_file,
    validate_hooks,
)
from src.stage2_probes import (  # noqa: E402
    _class_counts,
    _has_two_classes,
    _make_logistic_pipeline,
    _safe_auc,
    load_activation_matrix,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
)
from src.stage2_steering import (  # noqa: E402
    make_orthogonal_unit_direction,
    parse_float_list,
    parse_int_list,
    strength_label,
)


GENERATION_REQUEST_RE = re.compile(
    r"\s*Please come up with hypothes(?:is|es) to explain observations\.?\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ChoiceCondition:
    label: str
    direction_kind: str | None
    strength_sd: float


@dataclass(frozen=True)
class ForcedChoicePrompt:
    system: str
    user: str
    label: int
    gold_choice: str
    option_a: str
    option_b: str
    gold_hypothesis: str
    foil_hypothesis: str


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


def format_hypothesis(row: dict[str, Any], *, negated: bool) -> str:
    hypothesis = row["ontology_fol_structured"]["hypothesis"]
    subject = str(hypothesis["subject"]).strip().lower()
    predicate = str(hypothesis["predicate"]).strip().lower()
    negation = " not" if negated else ""
    return f"Every {subject} is{negation} {predicate}."


def build_forced_choice_prompt(row: dict[str, Any], *, row_index: int, option_seed: int) -> ForcedChoicePrompt:
    hypothesis = row["ontology_fol_structured"]["hypothesis"]
    gold_negated = bool(hypothesis["negated"])
    gold = format_hypothesis(row, negated=gold_negated)
    foil = format_hypothesis(row, negated=not gold_negated)
    rng = random.Random(option_seed + int(row_index) * 1009)
    gold_is_b = bool(rng.getrandbits(1))
    if gold_is_b:
        option_a, option_b = foil, gold
        gold_choice = "B"
        label = 1
    else:
        option_a, option_b = gold, foil
        gold_choice = "A"
        label = 0

    base_prompt = GENERATION_REQUEST_RE.sub("", row["prompt_text"]).strip()
    user = (
        f"{base_prompt}\n\n"
        "Which hypothesis best explains the observations?\n"
        f"(A) {option_a}\n"
        f"(B) {option_b}\n\n"
        "Answer with exactly one option: (A) or (B)."
    )
    system = (
        "You are a careful reasoner answering forced-choice ontology questions. "
        "Return only the final option, either (A) or (B)."
    )
    return ForcedChoicePrompt(
        system=system,
        user=user,
        label=label,
        gold_choice=gold_choice,
        option_a=option_a,
        option_b=option_b,
        gold_hypothesis=gold,
        foil_hypothesis=foil,
    )


def parse_choice(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\(([AB])\)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([AB])\b", text.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def parse_condition_kinds(value: str) -> list[str]:
    allowed = {"baseline", "toward_gold", "away_gold", "orthogonal"}
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition kind")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    return parsed


def make_condition_plan(*, condition_kinds: list[str], strengths: tuple[float, ...]) -> list[ChoiceCondition]:
    plan: list[ChoiceCondition] = []
    if "baseline" in condition_kinds:
        plan.append(ChoiceCondition("baseline", None, 0.0))
    for kind in ("toward_gold", "away_gold", "orthogonal"):
        if kind not in condition_kinds:
            continue
        for strength in strengths:
            if strength <= 0:
                raise ValueError("forced-choice strengths must be positive; sign is chosen per example")
            plan.append(ChoiceCondition(f"{kind}_{strength_label(strength)}", kind, float(strength)))
    if not plan:
        raise ValueError("condition plan is empty")
    return plan


def positive_scores(model: Any, x: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    logreg = model[-1]
    positive_index = int(np.where(logreg.classes_ == 1)[0][0])
    return [float(score) for score in model.predict_proba(x[indices])[:, positive_index]]


def serializable_direction_summary(direction: dict[str, Any]) -> dict[str, Any]:
    skip = {"unit_direction", "raw_coef", "coef_std", "scaler_mean", "scaler_scale"}
    return {key: value for key, value in direction.items() if key not in skip}


def save_direction_artifact(
    *,
    path: Path,
    direction: dict[str, Any],
    orthogonal_direction: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_direction=direction["unit_direction"],
        orthogonal_direction=orthogonal_direction.astype(np.float32),
        raw_coef=direction["raw_coef"],
        coef_std=direction["coef_std"],
        scaler_mean=direction["scaler_mean"],
        scaler_scale=direction["scaler_scale"],
        train_projection_std=np.array(direction["train_projection_std"], dtype=np.float32),
        train_projection_mean=np.array(direction["train_projection_mean"], dtype=np.float32),
        best_c=np.array(direction["best_c"], dtype=np.float32),
    )


def read_stage1_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def split_counts_from_sidecar(
    *,
    sidecar: list[dict[str, Any]],
    labels: list[int],
    splits_path: Path,
    source_file: str,
    split_family: str,
) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
    assignments = read_split_assignments(splits_path)
    splits = split_indices_from_assignments(
        sidecar,
        assignments=assignments,
        source_file=source_file,
        split_field=f"{split_family}_split",
    )
    counts = {split: _class_counts(labels, indices) for split, indices in splits.items()}
    return splits, counts


def train_forced_choice_direction(
    *,
    activation_path: Path,
    sidecar_path: Path,
    splits_path: Path,
    source_file: str,
    split_family: str,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
) -> dict[str, Any]:
    x = load_activation_matrix(activation_path)
    sidecar = read_jsonl(sidecar_path)
    if x.shape[0] != len(sidecar):
        raise ValueError(f"{activation_path} rows {x.shape[0]} != sidecar rows {len(sidecar)}")
    labels = [int(row["forced_choice_label"]) for row in sidecar]
    splits, split_counts = split_counts_from_sidecar(
        sidecar=sidecar,
        labels=labels,
        splits_path=splits_path,
        source_file=source_file,
        split_family=split_family,
    )
    for split, indices in splits.items():
        if not _has_two_classes(labels, indices):
            raise ValueError(f"{split} split has one class: {split_counts[split]}")

    best: dict[str, Any] | None = None
    for c_value in c_values:
        model = _make_logistic_pipeline(x, c_value=c_value, max_iter=max_iter, solver=solver)
        model.fit(x[splits["train"]], [labels[idx] for idx in splits["train"]])
        val_scores = positive_scores(model, x, splits["val"])
        val_auc = _safe_auc([labels[idx] for idx in splits["val"]], val_scores)
        rank_auc = val_auc if val_auc is not None else -np.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {"model": model, "c": float(c_value), "val_auc": val_auc, "rank_auc": rank_auc}
    if best is None:
        raise ValueError("no C values provided")

    model = best["model"]
    scaler = model[0]
    logreg = model[-1]
    if not np.array_equal(logreg.classes_, np.array([0, 1])):
        raise ValueError(f"expected logistic classes [0, 1], got {logreg.classes_.tolist()}")
    coef_std = np.asarray(logreg.coef_[0], dtype=np.float64)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)
    raw_coef = coef_std / scaler_scale
    raw_norm = float(np.linalg.norm(raw_coef))
    if raw_norm == 0.0:
        raise ValueError("forced-choice probe coefficient has zero norm")
    unit_direction = raw_coef / raw_norm
    train_projection = x[splits["train"]].astype(np.float64) @ unit_direction
    projection_std = float(train_projection.std(ddof=0))
    if projection_std == 0.0:
        raise ValueError("train projection has zero standard deviation")
    test_scores = positive_scores(model, x, splits["test"])
    test_auc = _safe_auc([labels[idx] for idx in splits["test"]], test_scores)
    return {
        "unit_direction": unit_direction.astype(np.float32),
        "raw_coef": raw_coef.astype(np.float32),
        "coef_std": coef_std.astype(np.float32),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "best_c": best["c"],
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "val_auc": best["val_auc"],
        "test_auc": test_auc,
        "split_counts": split_counts,
        "input_rows": len(sidecar),
        "d_model": int(x.shape[1]),
        "raw_coef_norm": raw_norm,
        "train_projection_mean": float(train_projection.mean()),
        "train_projection_std": projection_std,
        "train_projection_min": float(train_projection.min()),
        "train_projection_max": float(train_projection.max()),
        "label_semantics": "1 means gold option is B; 0 means gold option is A",
    }


def forced_choice_sidecar_for_row(
    *,
    row_index: int,
    row: dict[str, Any],
    prompt: ForcedChoicePrompt,
    token_count: int,
    hook_name: str,
) -> dict[str, Any]:
    return {
        "row_index": int(row_index),
        "example_id": row.get("example_id"),
        "height": row.get("height"),
        "task": row.get("task"),
        "model": row.get("model"),
        "is_correct_strong": row.get("is_correct_strong"),
        "parse_failed": row.get("parse_failed"),
        "token_count": int(token_count),
        "last_token_position": int(token_count) - 1,
        "hook_name": hook_name,
        "forced_choice_label": int(prompt.label),
        "gold_choice": prompt.gold_choice,
        "option_a": prompt.option_a,
        "option_b": prompt.option_b,
        "gold_hypothesis": prompt.gold_hypothesis,
        "foil_hypothesis": prompt.foil_hypothesis,
    }


def write_forced_choice_activation_outputs(
    *,
    activations_by_layer: dict[int, torch.Tensor],
    sidecars_by_layer: dict[int, list[dict[str, Any]]],
    metadata: dict[str, Any],
    activation_prefix: Path,
) -> None:
    from safetensors.torch import save_file

    activation_prefix.parent.mkdir(parents=True, exist_ok=True)
    if len(activations_by_layer) != 1:
        raise ValueError("forced-choice runner currently writes one layer per prefix")
    layer = next(iter(activations_by_layer))
    save_file({"activations": activations_by_layer[layer].contiguous()}, activation_prefix.with_suffix(".safetensors"))
    with activation_prefix.with_suffix(".example_ids.jsonl").open("w") as f:
        for row in sidecars_by_layer[layer]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    meta = {
        **metadata,
        "layer": layer,
        "activation_file": str(activation_prefix.with_suffix(".safetensors")),
        "sidecar_file": str(activation_prefix.with_suffix(".example_ids.jsonl")),
        "shape": list(activations_by_layer[layer].shape),
        "dtype": str(activations_by_layer[layer].dtype),
    }
    with activation_prefix.with_suffix(".meta.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


def build_encoded_examples(
    *,
    source_rows: list[dict[str, Any]],
    tokenizer,
    model_name: str,
    option_seed: int,
) -> tuple[list[EncodedExample], dict[int, ForcedChoicePrompt]]:
    encoded: list[EncodedExample] = []
    prompts: dict[int, ForcedChoicePrompt] = {}
    for row_index, row in enumerate(source_rows):
        prompt = build_forced_choice_prompt(row, row_index=row_index, option_seed=option_seed)
        text = render_chat_text(
            tokenizer,
            system=prompt.system,
            user=prompt.user,
            model_name=model_name,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"row {row_index} produced no tokens")
        prompts[row_index] = prompt
        encoded.append(
            EncodedExample(
                row_index=row_index,
                example_id=row.get("example_id"),
                height=row.get("height"),
                task=row.get("task"),
                model=row.get("model"),
                is_correct_strong=row.get("is_correct_strong"),
                parse_failed=row.get("parse_failed"),
                token_ids=list(token_ids),
            )
        )
    return encoded, prompts


def ensure_forced_choice_activations(
    *,
    model,
    source_rows: list[dict[str, Any]],
    model_name: str,
    layer: int,
    hook_name: str,
    batch_size: int,
    output_dtype: torch.dtype,
    activation_prefix: Path,
    jsonl_path: Path,
    option_seed: int,
    overwrite: bool,
    metadata_base: dict[str, Any],
) -> dict[str, Any]:
    activation_path = activation_prefix.with_suffix(".safetensors")
    sidecar_path = activation_prefix.with_suffix(".example_ids.jsonl")
    meta_path = activation_prefix.with_suffix(".meta.json")
    if activation_path.exists() and sidecar_path.exists() and not overwrite:
        return {
            "extracted": False,
            "activation_path": str(activation_path),
            "sidecar_path": str(sidecar_path),
            "meta_path": str(meta_path),
        }

    started = time.monotonic()
    encoded, prompts = build_encoded_examples(
        source_rows=source_rows,
        tokenizer=model.tokenizer,
        model_name=model_name,
        option_seed=option_seed,
    )
    activations, sidecar_rows, stats = extract_residual_activations(
        model,
        encoded,
        layers=[layer],
        batch_size=batch_size,
        output_dtype=output_dtype,
    )
    sidecars_by_layer = {
        layer: [
            forced_choice_sidecar_for_row(
                row_index=example.row_index,
                row=source_rows[example.row_index],
                prompt=prompts[example.row_index],
                token_count=sidecar_rows[idx]["token_count"],
                hook_name=hook_name,
            )
            for idx, example in enumerate(encoded)
        ]
    }
    token_counts = [example.token_count for example in encoded]
    metadata = {
        **metadata_base,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": sha256_file(jsonl_path),
        "row_count": len(encoded),
        "option_seed": option_seed,
        "prompt_type": "forced_choice_gold_vs_opposite_polarity",
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_count_mean": sum(token_counts) / len(token_counts),
        "extraction_stats": stats,
        "extraction_elapsed_seconds_total": time.monotonic() - started,
    }
    write_forced_choice_activation_outputs(
        activations_by_layer=activations,
        sidecars_by_layer=sidecars_by_layer,
        metadata=metadata,
        activation_prefix=activation_prefix,
    )
    return {
        "extracted": True,
        "activation_path": str(activation_path),
        "sidecar_path": str(sidecar_path),
        "meta_path": str(meta_path),
        "stats": stats,
    }


def select_forced_choice_rows(
    *,
    source_rows: list[dict[str, Any]],
    splits_path: Path,
    source_file: str,
    split_family: str,
    heights: list[int],
    per_height_label: int,
    option_seed: int,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from collections import defaultdict

    assignments = read_split_assignments(splits_path)
    height_set = set(heights)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    available_counts: dict[str, int] = defaultdict(int)
    for row_index, row in enumerate(source_rows):
        height = row.get("height")
        if height not in height_set:
            continue
        assignment = assignments.get((source_file, row_index))
        if assignment is None or assignment.get(f"{split_family}_split") != "test":
            continue
        prompt = build_forced_choice_prompt(row, row_index=row_index, option_seed=option_seed)
        labeled = dict(row)
        labeled["row_index"] = row_index
        labeled["forced_choice_label"] = prompt.label
        labeled["gold_choice"] = prompt.gold_choice
        labeled["option_a"] = prompt.option_a
        labeled["option_b"] = prompt.option_b
        labeled["gold_hypothesis"] = prompt.gold_hypothesis
        labeled["foil_hypothesis"] = prompt.foil_hypothesis
        groups[(int(height), int(prompt.label))].append(labeled)
        available_counts[f"h{height}_{prompt.gold_choice}"] += 1

    selected: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    for height in heights:
        for label in (0, 1):
            rows = list(groups[(height, label)])
            if len(rows) < per_height_label:
                missing[f"h{height}_{'B' if label else 'A'}"] = len(rows)
                continue
            rng = random.Random(selection_seed + height * 1009 + label * 9176)
            rng.shuffle(rows)
            selected.extend(rows[:per_height_label])
    if missing:
        raise ValueError(
            f"not enough rows for balanced forced-choice subset; requested {per_height_label}, counts={missing}"
        )
    selected.sort(key=lambda row: (int(row["height"]), int(row["forced_choice_label"]), int(row["row_index"])))
    return selected, {
        "source_file": source_file,
        "split_family": split_family,
        "split": "test",
        "heights": heights,
        "per_height_label": per_height_label,
        "option_seed": option_seed,
        "selection_seed": selection_seed,
        "available_counts": dict(sorted(available_counts.items())),
        "selected_rows": len(selected),
        "selected_counts": {
            f"h{height}_{'B' if label else 'A'}": sum(
                1
                for row in selected
                if row.get("height") == height and int(row.get("forced_choice_label")) == label
            )
            for height in heights
            for label in (0, 1)
        },
    }


def answer_sign(label: int) -> int:
    return 1 if int(label) == 1 else -1


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


def output_row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["source_row_index"]), str(row["condition"])


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


def summarize_choice_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    by_condition_summary: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        n = len(condition_rows)
        valid_choice_rows = [row for row in condition_rows if row.get("parsed_choice") is not None]
        by_condition_summary[condition] = {
            "n": n,
            "choice_accuracy": sum(bool(row["is_correct_choice"]) for row in condition_rows) / n if n else None,
            "choice_parse_fail_rate": (
                sum(bool(row["choice_parse_failed"]) for row in condition_rows) / n if n else None
            ),
            "mean_generated_tokens": (
                sum(int(row.get("generated_token_count", 0)) for row in condition_rows) / n if n else None
            ),
            "mean_output_chars": sum(len(row.get("model_output", "")) for row in condition_rows) / n if n else None,
            "choice_change_rate": (
                sum(bool(row.get("choice_changed_vs_baseline")) for row in condition_rows if row.get("choice_changed_vs_baseline") is not None)
                / len([row for row in condition_rows if row.get("choice_changed_vs_baseline") is not None])
                if any(row.get("choice_changed_vs_baseline") is not None for row in condition_rows)
                else None
            ),
            "parsed_choice_counts": {
                choice: sum(row.get("parsed_choice") == choice for row in valid_choice_rows)
                for choice in ("A", "B")
            },
        }

    baselines = {int(row["source_row_index"]): row for row in rows if row.get("condition") == "baseline"}
    flips: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        if condition == "baseline":
            continue
        paired = [
            (baselines[int(row["source_row_index"])], row)
            for row in condition_rows
            if int(row["source_row_index"]) in baselines
        ]
        if not paired:
            continue
        flips[condition] = {
            "paired_n": len(paired),
            "choice_false_to_true": int(
                sum((not base["is_correct_choice"]) and steered["is_correct_choice"] for base, steered in paired)
            ),
            "choice_true_to_false": int(
                sum(base["is_correct_choice"] and (not steered["is_correct_choice"]) for base, steered in paired)
            ),
            "choice_changed": int(sum(base.get("parsed_choice") != steered.get("parsed_choice") for base, steered in paired)),
            "choice_flips_toward_gold": int(
                sum(
                    base.get("parsed_choice") is not None
                    and steered.get("parsed_choice") == base.get("gold_choice")
                    and base.get("parsed_choice") != base.get("gold_choice")
                    for base, steered in paired
                )
            ),
            "choice_flips_away_from_gold": int(
                sum(
                    base.get("parsed_choice") == base.get("gold_choice")
                    and steered.get("parsed_choice") is not None
                    and steered.get("parsed_choice") != base.get("gold_choice")
                    for base, steered in paired
                )
            ),
            "net_choice_accuracy_delta": (
                sum(bool(steered["is_correct_choice"]) for _, steered in paired)
                - sum(bool(base["is_correct_choice"]) for base, _ in paired)
            )
            / len(paired),
        }
    return {
        "by_condition": by_condition_summary,
        "choice_flips_vs_baseline": flips,
    }


def dry_run(args: argparse.Namespace) -> int:
    source_rows = read_stage1_rows(args.jsonl)
    sidecar = []
    labels = []
    tokenless_counts: dict[str, int] = {"A": 0, "B": 0}
    for row_index, row in enumerate(source_rows):
        prompt = build_forced_choice_prompt(row, row_index=row_index, option_seed=args.option_seed)
        sidecar.append({"row_index": row_index, "forced_choice_label": prompt.label})
        labels.append(prompt.label)
        tokenless_counts[prompt.gold_choice] += 1
    splits, split_counts = split_counts_from_sidecar(
        sidecar=sidecar,
        labels=labels,
        splits_path=args.splits,
        source_file=str(args.jsonl),
        split_family=args.split_family,
    )
    selected, selection = select_forced_choice_rows(
        source_rows=source_rows,
        splits_path=args.splits,
        source_file=str(args.jsonl),
        split_family=args.split_family,
        heights=parse_int_list(args.heights),
        per_height_label=args.per_height_label,
        option_seed=args.option_seed,
        selection_seed=args.selection_seed,
    )
    first = selected[0]
    first_prompt = build_forced_choice_prompt(first, row_index=int(first["row_index"]), option_seed=args.option_seed)
    print(json.dumps({
        "rows": len(source_rows),
        "gold_choice_counts": tokenless_counts,
        "split_counts": split_counts,
        "split_sizes": {split: len(indices) for split, indices in splits.items()},
        "selection": selection,
        "first_selected_row": {
            "row_index": first["row_index"],
            "height": first["height"],
            "gold_choice": first_prompt.gold_choice,
            "option_a": first_prompt.option_a,
            "option_b": first_prompt.option_b,
            "user_prompt": first_prompt.user,
        },
    }, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_4b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model-key", default="gemma3_4b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-prefix", type=Path, default=None)
    parser.add_argument("--overwrite-activations", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits_4b_property.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--option-seed", type=int, default=20260430)
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
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
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/steering/forced_choice_4b_l22_property_smoke.jsonl"),
    )
    parser.add_argument(
        "--direction-output",
        type=Path,
        default=Path("results/stage2/steering/forced_choice_4b_l22_property_direction.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/forced_choice_steering_4b_l22_property_smoke.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.dry_run:
        return dry_run(args)

    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()
    source_file = str(args.jsonl)
    source_rows = read_stage1_rows(args.jsonl)
    heights = parse_int_list(args.heights)
    strengths = parse_float_list(args.strengths)
    condition_plan = make_condition_plan(
        condition_kinds=parse_condition_kinds(args.conditions),
        strengths=strengths,
    )
    dtype = torch_dtype(args.dtype)
    output_dtype = torch_dtype(args.output_dtype)
    activation_prefix = (
        args.activation_prefix
        if args.activation_prefix is not None
        else args.activation_dir / f"{args.model_key}_{args.task}_L{args.layer}_forced_choice"
    )
    activation_path = activation_prefix.with_suffix(".safetensors")
    sidecar_path = activation_prefix.with_suffix(".example_ids.jsonl")

    print("Stage 2 forced-choice steering", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"activation_prefix={activation_prefix}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    hook_name = validate_hooks(model, [args.layer])[0]
    if model.tokenizer is None:
        raise ValueError("loaded model has no tokenizer")

    activation_report = ensure_forced_choice_activations(
        model=model,
        source_rows=source_rows,
        model_name=args.model,
        layer=args.layer,
        hook_name=hook_name,
        batch_size=args.batch_size,
        output_dtype=output_dtype,
        activation_prefix=activation_prefix,
        jsonl_path=args.jsonl,
        option_seed=args.option_seed,
        overwrite=args.overwrite_activations,
        metadata_base={
            "model_name": args.model,
            "model_key": args.model_key,
            "task": args.task,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "load_mode": args.load_mode,
            "dtype": str(dtype),
            "output_dtype": str(output_dtype),
            "batch_size": args.batch_size,
            "hook_name": hook_name,
            "module_devices": module_device_summary(model),
        },
    )
    print(f"activation_report={activation_report}", flush=True)

    direction = train_forced_choice_direction(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
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

    selected_rows, selection_summary = select_forced_choice_rows(
        source_rows=source_rows,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=heights,
        per_height_label=args.per_height_label,
        option_seed=args.option_seed,
        selection_seed=args.selection_seed,
    )
    print(
        f"selected_rows={len(selected_rows)} "
        f"available_counts={selection_summary['available_counts']}",
        flush=True,
    )

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
    baseline_choice_by_source: dict[int, str | None] = {}
    if existing_rows_by_key:
        for stage1_row in selected_rows:
            source_row_index = int(stage1_row["row_index"])
            for condition in condition_plan:
                existing_row = existing_rows_by_key.get((source_row_index, condition.label))
                if existing_row is None:
                    continue
                rows.append(existing_row)
                if condition.direction_kind is None:
                    baseline_choice_by_source[source_row_index] = existing_row.get("parsed_choice")
        with args.out_jsonl.open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    vector_by_kind = {
        "toward_gold": direction["unit_direction"],
        "away_gold": direction["unit_direction"],
        "orthogonal": orthogonal_direction,
    }
    projection_std = float(direction["train_projection_std"])
    output_mode = "a" if args.resume else "w"
    with args.out_jsonl.open(output_mode) as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            source_row_index = int(stage1_row["row_index"])
            prompt = build_forced_choice_prompt(
                stage1_row,
                row_index=source_row_index,
                option_seed=args.option_seed,
            )
            prompt_text = render_chat_text(
                model.tokenizer,
                system=prompt.system,
                user=prompt.user,
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = model.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            print(
                f"row {row_idx}/{len(selected_rows)} source_row={source_row_index} "
                f"h={stage1_row['height']} gold={prompt.gold_choice} "
                f"prompt_tokens={len(token_ids)}",
                flush=True,
            )
            for condition in condition_plan:
                row_key = (source_row_index, condition.label)
                if row_key in existing_rows_by_key:
                    existing_row = existing_rows_by_key[row_key]
                    print(
                        f"  {condition.label}: resume_skip choice={existing_row.get('parsed_choice')} "
                        f"correct={existing_row.get('is_correct_choice')}",
                        flush=True,
                    )
                    continue

                hook_state = {"calls": 0, "applications": 0}
                signed_delta = 0.0
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
                    sign_to_gold = answer_sign(prompt.label)
                    if condition.direction_kind == "toward_gold":
                        signed_delta = sign_to_gold * condition.strength_sd * projection_std
                    elif condition.direction_kind == "away_gold":
                        signed_delta = -sign_to_gold * condition.strength_sd * projection_std
                    elif condition.direction_kind == "orthogonal":
                        signed_delta = sign_to_gold * condition.strength_sd * projection_std
                    else:
                        raise ValueError(f"unknown direction kind {condition.direction_kind!r}")
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
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype,
                        )

                parsed_choice = parse_choice(reply)
                is_correct_choice = parsed_choice == prompt.gold_choice
                baseline_choice = baseline_choice_by_source.get(source_row_index)
                choice_changed = None
                if baseline_choice is not None or condition.direction_kind is None:
                    choice_changed = parsed_choice != (baseline_choice if baseline_choice is not None else parsed_choice)
                if condition.direction_kind is None:
                    baseline_choice_by_source[source_row_index] = parsed_choice
                    choice_changed = False

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
                    "original_parse_failed": bool(stage1_row.get("parse_failed")),
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
                    "forced_choice_label": int(prompt.label),
                    "gold_choice": prompt.gold_choice,
                    "option_a": prompt.option_a,
                    "option_b": prompt.option_b,
                    "gold_hypothesis": prompt.gold_hypothesis,
                    "foil_hypothesis": prompt.foil_hypothesis,
                    "parsed_choice": parsed_choice,
                    "choice_parse_failed": parsed_choice is None,
                    "is_correct_choice": bool(is_correct_choice),
                    "baseline_choice": baseline_choice if baseline_choice is not None else parsed_choice,
                    "choice_changed_vs_baseline": bool(choice_changed) if choice_changed is not None else None,
                }
                rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"  {condition.label}: choice={parsed_choice} correct={is_correct_choice} "
                    f"new_tokens={len(new_ids)} hooks={hook_state['applications']}/{hook_state['calls']}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_steer_forced_choice_direction.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "layer": args.layer,
        "hook_name": hook_name,
        "activation_path": str(activation_path),
        "sidecar_path": str(sidecar_path),
        "activation_report": activation_report,
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
            "resume": args.resume,
            "resume_existing_rows": len(existing_rows_by_key),
            "resume_malformed_rows": resume_malformed_rows,
            "resume_ignored_rows": resume_ignored_rows,
        },
        "summary": summarize_choice_rows(rows),
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
