#!/usr/bin/env python3
"""Steer Gemma 3 with answer-content directions instead of correctness labels."""

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
from src.bd_path import ensure_on_path  # noqa: E402
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
    score_reply,
    strength_label,
)


@dataclass(frozen=True)
class AnswerCondition:
    label: str
    direction_kind: str | None
    strength_sd: float


@dataclass(frozen=True)
class AnswerContent:
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


def canonical_word(value: str | None) -> str | None:
    if value is None:
        return None
    word = re.sub(r"[^A-Za-z]", "", value).lower()
    return word or None


def singularize_concept(value: str | None) -> str | None:
    word = canonical_word(value)
    if word is None:
        return None
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def extract_answer_content_from_hypothesis(text: str | None) -> AnswerContent:
    if not text:
        return AnswerContent(predicate=None, negated=None)
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.strip(" .;:")
    patterns = (
        r"^(?:every|each|all)\s+(?P<subject>[A-Za-z]+)\s+(?:is|are)\s+(?P<neg>not\s+)?(?P<predicate>[A-Za-z]+)$",
        r"^(?P<subject>[A-Za-z]+)\s+(?:is|are)\s+(?P<neg>not\s+)?(?P<predicate>[A-Za-z]+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return AnswerContent(
                predicate=singularize_concept(match.group("predicate")),
                negated=bool(match.group("neg")),
            )
    # Last-resort parser for already canonical strings like "not angry".
    lowered = cleaned.lower()
    match = re.search(r"\b(?:is|are)\s+(?P<neg>not\s+)?(?P<predicate>[A-Za-z]+)\b", lowered)
    if match:
        return AnswerContent(
            predicate=singularize_concept(match.group("predicate")),
            negated=bool(match.group("neg")),
        )
    return AnswerContent(predicate=None, negated=None)


def extract_answer_content_from_text(text: str | None) -> AnswerContent:
    if not text:
        return AnswerContent(predicate=None, negated=None)
    try:
        from src.gemma3_parse import parse_hypotheses

        parsed = parse_hypotheses(text)
    except Exception:
        parsed = []
    candidates = parsed if parsed else [line.strip() for line in text.splitlines() if line.strip()]
    for candidate in candidates:
        content = extract_answer_content_from_hypothesis(candidate)
        if content.predicate is not None or content.negated is not None:
            return content
    return AnswerContent(predicate=None, negated=None)


def gold_answer_content(row: dict[str, Any]) -> AnswerContent:
    hypothesis = row["ontology_fol_structured"]["hypothesis"]
    return AnswerContent(
        predicate=singularize_concept(hypothesis.get("predicate")),
        negated=bool(hypothesis.get("negated")),
    )


def answer_label_for_row(
    row: dict[str, Any],
    *,
    label_source: str,
    answer_target: str,
    positive_predicate: str | None,
    negative_predicate: str | None,
) -> int | None:
    if label_source == "gold":
        content = gold_answer_content(row)
    elif label_source == "stage1_model_output":
        content = extract_answer_content_from_text(row.get("model_output", ""))
    else:
        raise ValueError(f"unknown answer label source {label_source!r}")

    if answer_target == "polarity":
        if content.negated is None:
            return None
        return int(content.negated)
    if answer_target == "predicate_pair":
        if content.predicate is None:
            return None
        pos = singularize_concept(positive_predicate)
        neg = singularize_concept(negative_predicate)
        if pos is None or neg is None:
            raise ValueError("predicate_pair requires --positive-predicate and --negative-predicate")
        if content.predicate == pos:
            return 1
        if content.predicate == neg:
            return 0
        return None
    if answer_target == "predicate_one_vs_rest":
        if content.predicate is None:
            return None
        pos = singularize_concept(positive_predicate)
        if pos is None:
            raise ValueError("predicate_one_vs_rest requires --positive-predicate")
        return int(content.predicate == pos)
    raise ValueError(f"unknown answer target {answer_target!r}")


def parse_condition_kinds(value: str) -> list[str]:
    allowed = {"baseline", "toward_gold", "away_gold", "orthogonal"}
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition kind")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    return parsed


def make_answer_condition_plan(
    *,
    condition_kinds: list[str],
    strengths: tuple[float, ...],
) -> list[AnswerCondition]:
    plan: list[AnswerCondition] = []
    if "baseline" in condition_kinds:
        plan.append(AnswerCondition("baseline", None, 0.0))
    for kind in ("toward_gold", "away_gold", "orthogonal"):
        if kind not in condition_kinds:
            continue
        for strength in strengths:
            if strength <= 0:
                raise ValueError("answer steering strengths must be positive; direction is chosen per example")
            plan.append(AnswerCondition(f"{kind}_{strength_label(strength)}", kind, float(strength)))
    if not plan:
        raise ValueError("condition plan is empty")
    return plan


def positive_scores(model: Any, x: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    logreg = model[-1]
    positive_index = int(np.where(logreg.classes_ == 1)[0][0])
    return [float(score) for score in model.predict_proba(x[indices])[:, positive_index]]


def load_answer_probe_dataset(
    *,
    activation_path: Path,
    sidecar_path: Path,
    source_rows: list[dict[str, Any]],
    label_source: str,
    answer_target: str,
    positive_predicate: str | None,
    negative_predicate: str | None,
    drop_parse_failed: bool = True,
) -> dict[str, Any]:
    x_all = load_activation_matrix(activation_path)
    sidecar_all = read_jsonl(sidecar_path)
    if x_all.shape[0] != len(sidecar_all):
        raise ValueError(f"{activation_path} rows {x_all.shape[0]} != sidecar rows {len(sidecar_all)}")

    keep_indices: list[int] = []
    labels: list[int] = []
    skipped_no_label = 0
    for idx, sidecar_row in enumerate(sidecar_all):
        if drop_parse_failed and sidecar_row.get("parse_failed"):
            continue
        source_row = source_rows[int(sidecar_row["row_index"])]
        label = answer_label_for_row(
            source_row,
            label_source=label_source,
            answer_target=answer_target,
            positive_predicate=positive_predicate,
            negative_predicate=negative_predicate,
        )
        if label is None:
            skipped_no_label += 1
            continue
        keep_indices.append(idx)
        labels.append(int(label))

    x = x_all[keep_indices]
    sidecar = [sidecar_all[idx] for idx in keep_indices]
    return {
        "x": x,
        "sidecar": sidecar,
        "labels": labels,
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
        "skipped_no_label": skipped_no_label,
        "d_model": int(x.shape[1]) if x.ndim == 2 else None,
    }


def train_answer_probe_direction(
    *,
    activation_path: Path,
    sidecar_path: Path,
    source_rows: list[dict[str, Any]],
    splits_path: Path,
    source_file: str,
    split_family: str,
    label_source: str,
    answer_target: str,
    positive_predicate: str | None,
    negative_predicate: str | None,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
) -> dict[str, Any]:
    dataset = load_answer_probe_dataset(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        source_rows=source_rows,
        label_source=label_source,
        answer_target=answer_target,
        positive_predicate=positive_predicate,
        negative_predicate=negative_predicate,
        drop_parse_failed=True,
    )
    x = dataset["x"]
    labels = dataset["labels"]
    sidecar = dataset["sidecar"]
    assignments = read_split_assignments(splits_path)
    splits = split_indices_from_assignments(
        sidecar,
        assignments=assignments,
        source_file=source_file,
        split_field=f"{split_family}_split",
    )
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    split_counts = {split: _class_counts(labels, indices) for split, indices in splits.items()}
    for split, indices in splits.items():
        if not _has_two_classes(labels, indices):
            raise ValueError(f"{split} split has one class: {split_counts[split]}")

    best: dict[str, Any] | None = None
    for c_value in c_values:
        model = _make_logistic_pipeline(x, c_value=c_value, max_iter=max_iter, solver=solver)
        model.fit(x[train_indices], [labels[idx] for idx in train_indices])
        val_scores = positive_scores(model, x, val_indices)
        val_auc = _safe_auc([labels[idx] for idx in val_indices], val_scores)
        rank_auc = val_auc if val_auc is not None else -np.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "c": float(c_value),
                "val_auc": val_auc,
                "rank_auc": rank_auc,
            }
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
        raise ValueError("answer probe coefficient has zero norm")
    unit_direction = raw_coef / raw_norm
    train_projection = x[train_indices].astype(np.float64) @ unit_direction
    projection_std = float(train_projection.std(ddof=0))
    if projection_std == 0.0:
        raise ValueError("train projection has zero standard deviation")
    test_scores = positive_scores(model, x, test_indices)
    test_auc = _safe_auc([labels[idx] for idx in test_indices], test_scores)

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
        "input_rows": dataset["input_rows"],
        "kept_rows": dataset["kept_rows"],
        "skipped_no_label": dataset["skipped_no_label"],
        "d_model": dataset["d_model"],
        "raw_coef_norm": raw_norm,
        "train_projection_mean": float(train_projection.mean()),
        "train_projection_std": projection_std,
        "train_projection_min": float(train_projection.min()),
        "train_projection_max": float(train_projection.max()),
        "label_source": label_source,
        "answer_target": answer_target,
        "positive_predicate": singularize_concept(positive_predicate),
        "negative_predicate": singularize_concept(negative_predicate),
    }


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


def make_steering_hook(
    *,
    vector: np.ndarray,
    delta: float,
    scope: str,
) -> tuple[Any, dict[str, int]]:
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


def select_answer_rows(
    *,
    jsonl_path: Path,
    splits_path: Path,
    source_file: str,
    split_family: str,
    heights: list[int],
    per_height_label: int,
    seed: int,
    label_source: str,
    answer_target: str,
    positive_predicate: str | None,
    negative_predicate: str | None,
    drop_parse_failed: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import random
    from collections import defaultdict

    assignments = read_split_assignments(splits_path)
    target_split = "test"
    height_set = set(heights)
    groups: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    available_counts = defaultdict(int)
    skipped_no_label = 0
    with jsonl_path.open() as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            height = row.get("height")
            if height not in height_set:
                continue
            if drop_parse_failed and row.get("parse_failed"):
                continue
            assignment = assignments.get((source_file, row_index))
            if assignment is None or assignment.get(f"{split_family}_split") != target_split:
                continue
            label = answer_label_for_row(
                row,
                label_source=label_source,
                answer_target=answer_target,
                positive_predicate=positive_predicate,
                negative_predicate=negative_predicate,
            )
            if label is None:
                skipped_no_label += 1
                continue
            labeled = dict(row)
            labeled["row_index"] = row_index
            labeled["answer_label"] = int(label)
            groups[(int(height), bool(row["is_correct_strong"]))].append(labeled)
            available_counts[f"h{height}_{'correct' if row['is_correct_strong'] else 'incorrect'}"] += 1

    selected: list[dict[str, Any]] = []
    missing = {}
    for height in heights:
        for correct in (False, True):
            key = (height, correct)
            rows = list(groups[key])
            if len(rows) < per_height_label:
                missing[f"h{height}_{'correct' if correct else 'incorrect'}"] = len(rows)
                continue
            rng = random.Random(seed + height * 1009 + int(correct) * 9176)
            rng.shuffle(rows)
            selected.extend(rows[:per_height_label])
    if missing:
        raise ValueError(
            f"not enough rows for balanced steering subset; requested {per_height_label}, counts={missing}"
        )
    selected.sort(key=lambda row: (int(row["height"]), bool(row["is_correct_strong"]), int(row["row_index"])))
    summary = {
        "source_file": source_file,
        "split_family": split_family,
        "split": target_split,
        "heights": heights,
        "per_height_label": per_height_label,
        "drop_parse_failed": drop_parse_failed,
        "seed": seed,
        "available_counts": dict(sorted(available_counts.items())),
        "selected_rows": len(selected),
        "selected_counts": {
            f"h{height}_{'correct' if correct else 'incorrect'}": sum(
                1
                for row in selected
                if row.get("height") == height and bool(row.get("is_correct_strong")) is correct
            )
            for height in heights
            for correct in (False, True)
        },
        "skipped_no_answer_label": skipped_no_label,
        "answer_label_source": label_source,
        "answer_target": answer_target,
        "positive_predicate": singularize_concept(positive_predicate),
        "negative_predicate": singularize_concept(negative_predicate),
    }
    return selected, summary


def answer_sign(label: int) -> int:
    return 1 if int(label) == 1 else -1


def answer_match(value: Any, target: Any) -> bool | None:
    if value is None or target is None:
        return None
    return value == target


def add_answer_metrics(
    *,
    stage1_row: dict[str, Any],
    score: dict[str, Any],
    baseline_content: AnswerContent | None,
) -> dict[str, Any]:
    gold = gold_answer_content(stage1_row)
    parsed_hypotheses = score.get("parsed_hypotheses") or []
    parsed = AnswerContent(predicate=None, negated=None)
    for hypothesis in parsed_hypotheses:
        parsed = extract_answer_content_from_hypothesis(hypothesis)
        if parsed.predicate is not None or parsed.negated is not None:
            break
    if parsed.predicate is None and parsed.negated is None:
        parsed = extract_answer_content_from_text("\n".join(parsed_hypotheses))

    polarity_matches = answer_match(parsed.negated, gold.negated)
    predicate_matches = answer_match(parsed.predicate, gold.predicate)
    content_changed = None
    if baseline_content is not None:
        if parsed.predicate is not None or parsed.negated is not None:
            content_changed = (parsed.predicate, parsed.negated) != (
                baseline_content.predicate,
                baseline_content.negated,
            )
    return {
        "gold_predicate": gold.predicate,
        "gold_negated": gold.negated,
        "baseline_predicate": baseline_content.predicate if baseline_content else None,
        "baseline_negated": baseline_content.negated if baseline_content else None,
        "parsed_predicate": parsed.predicate,
        "parsed_negated": parsed.negated,
        "has_parsed_answer_content": parsed.predicate is not None or parsed.negated is not None,
        "polarity_matches_gold": polarity_matches,
        "predicate_matches_gold": predicate_matches,
        "answer_content_changed_vs_baseline": content_changed,
    }


def summarize_bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    valid = [row[key] for row in rows if row.get(key) is not None]
    if not valid:
        return None
    return sum(bool(value) for value in valid) / len(valid)


def summarize_answer_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    by_condition_summary: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        n = len(condition_rows)
        by_condition_summary[condition] = {
            "n": n,
            "strong_accuracy": sum(bool(row["is_correct_strong"]) for row in condition_rows) / n if n else None,
            "weak_accuracy": sum(bool(row["is_correct_weak"]) for row in condition_rows) / n if n else None,
            "parse_fail_rate": sum(bool(row["parse_failed"]) for row in condition_rows) / n if n else None,
            "mean_quality": sum(float(row["quality_score"]) for row in condition_rows) / n if n else None,
            "mean_generated_tokens": (
                sum(int(row.get("generated_token_count", 0)) for row in condition_rows) / n if n else None
            ),
            "mean_output_chars": sum(len(row.get("model_output", "")) for row in condition_rows) / n if n else None,
            "parsed_answer_content_rate": summarize_bool_rate(condition_rows, "has_parsed_answer_content"),
            "polarity_match_rate": summarize_bool_rate(condition_rows, "polarity_matches_gold"),
            "predicate_match_rate": summarize_bool_rate(condition_rows, "predicate_matches_gold"),
            "answer_content_change_rate": summarize_bool_rate(condition_rows, "answer_content_changed_vs_baseline"),
        }

    baselines = {
        int(row["source_row_index"]): row
        for row in rows
        if row.get("condition") == "baseline"
    }
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

        def polarity_toward(base: dict[str, Any], steered: dict[str, Any]) -> bool:
            return (
                base.get("parsed_negated") is not None
                and steered.get("parsed_negated") is not None
                and base.get("gold_negated") is not None
                and base.get("parsed_negated") != base.get("gold_negated")
                and steered.get("parsed_negated") == base.get("gold_negated")
            )

        def polarity_away(base: dict[str, Any], steered: dict[str, Any]) -> bool:
            return (
                base.get("parsed_negated") is not None
                and steered.get("parsed_negated") is not None
                and base.get("gold_negated") is not None
                and base.get("parsed_negated") == base.get("gold_negated")
                and steered.get("parsed_negated") != base.get("gold_negated")
            )

        def predicate_toward(base: dict[str, Any], steered: dict[str, Any]) -> bool:
            return (
                base.get("parsed_predicate") is not None
                and steered.get("parsed_predicate") is not None
                and base.get("gold_predicate") is not None
                and base.get("parsed_predicate") != base.get("gold_predicate")
                and steered.get("parsed_predicate") == base.get("gold_predicate")
            )

        def predicate_away(base: dict[str, Any], steered: dict[str, Any]) -> bool:
            return (
                base.get("parsed_predicate") is not None
                and steered.get("parsed_predicate") is not None
                and base.get("gold_predicate") is not None
                and base.get("parsed_predicate") == base.get("gold_predicate")
                and steered.get("parsed_predicate") != base.get("gold_predicate")
            )

        flips[condition] = {
            "paired_n": len(paired),
            "strong_false_to_true": int(
                sum((not base["is_correct_strong"]) and steered["is_correct_strong"] for base, steered in paired)
            ),
            "strong_true_to_false": int(
                sum(base["is_correct_strong"] and (not steered["is_correct_strong"]) for base, steered in paired)
            ),
            "strong_changed": int(
                sum(base["is_correct_strong"] != steered["is_correct_strong"] for base, steered in paired)
            ),
            "polarity_flips_toward_gold": int(sum(polarity_toward(base, steered) for base, steered in paired)),
            "polarity_flips_away_from_gold": int(sum(polarity_away(base, steered) for base, steered in paired)),
            "predicate_flips_toward_gold": int(sum(predicate_toward(base, steered) for base, steered in paired)),
            "predicate_flips_away_from_gold": int(sum(predicate_away(base, steered) for base, steered in paired)),
            "answer_content_changed": int(
                sum(bool(steered.get("answer_content_changed_vs_baseline")) for _, steered in paired)
            ),
            "net_strong_accuracy_delta": (
                sum(bool(steered["is_correct_strong"]) for _, steered in paired)
                - sum(bool(base["is_correct_strong"]) for base, _ in paired)
            )
            / len(paired),
        }

    return {
        "by_condition": by_condition_summary,
        "answer_flips_vs_baseline": flips,
    }


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_4b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model-key", default="gemma3_4b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-prefix", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits_4b_property.jsonl"))
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
    parser.add_argument("--per-height-label", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260427)
    parser.add_argument("--orthogonal-seed", type=int, default=20260545)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--conditions", default="baseline,toward_gold,away_gold,orthogonal")
    parser.add_argument("--strengths", default="0.5,1,2")
    parser.add_argument(
        "--intervention-scope",
        choices=("prompt_only", "last_token_each_forward"),
        default="last_token_each_forward",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/steering/answer_property_4b_l22_polarity_decode_sweep.jsonl"),
    )
    parser.add_argument(
        "--direction-output",
        type=Path,
        default=Path("results/stage2/steering/answer_property_4b_l22_polarity_direction.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/answer_property_steering_4b_l22_polarity_decode_sweep.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed (source_row_index, condition) rows already present in --out-jsonl.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)

    # Preflight external parser path early because scoring needs Beyond Deduction.
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

    print("Stage 2 answer/property steering", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"answer_label_source={args.answer_label_source}", flush=True)
    print(f"answer_target={args.answer_target}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"intervention_scope={args.intervention_scope}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
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

        # Rewrite a valid, deduplicated prefix before appending new generations.
        with args.out_jsonl.open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    projection_std = float(direction["train_projection_std"])
    output_mode = "a" if args.resume else "w"
    with args.out_jsonl.open(output_mode) as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            source_row_index = int(stage1_row["row_index"])
            prompt_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            gold = gold_answer_content(stage1_row)
            print(
                f"row {row_idx}/{len(selected_rows)} "
                f"source_row={stage1_row['row_index']} h={stage1_row['height']} "
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
                    signed_delta = 0.0
                else:
                    sign_to_gold = answer_sign(int(stage1_row["answer_label"]))
                    if condition.direction_kind == "toward_gold":
                        signed_delta = sign_to_gold * condition.strength_sd * projection_std
                        vector = direction["unit_direction"]
                    elif condition.direction_kind == "away_gold":
                        signed_delta = -sign_to_gold * condition.strength_sd * projection_std
                        vector = direction["unit_direction"]
                    elif condition.direction_kind == "orthogonal":
                        signed_delta = sign_to_gold * condition.strength_sd * projection_std
                        vector = orthogonal_direction
                    else:
                        raise ValueError(f"unknown direction kind {condition.direction_kind!r}")
                    hook_fn, hook_state = make_steering_hook(
                        vector=vector,
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

    steering_summary = summarize_answer_rows(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_steer_answer_property_direction.py",
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
            "resume": args.resume,
            "resume_existing_rows": len(existing_rows_by_key),
            "resume_malformed_rows": resume_malformed_rows,
            "resume_ignored_rows": resume_ignored_rows,
        },
        "summary": steering_summary,
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
