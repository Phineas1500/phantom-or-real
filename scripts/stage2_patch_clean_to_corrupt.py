#!/usr/bin/env python3
"""Patch clean h1 residual states into corrupt h4 prompts at semantic landmarks."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
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

from scripts.stage2_steer_answer_property_margins import (  # noqa: E402
    build_margin_prompt,
    emitted_wrong_foil,
)
from scripts.stage2_steer_forced_choice_direction import generate_one  # noqa: E402
from src.activations import (  # noqa: E402
    input_device_for_model,
    load_tl_model,
    render_chat_text,
    validate_hooks,
)
from src.stage2_probes import read_jsonl, read_split_assignments  # noqa: E402
from src.stage2_steering import parse_int_list, score_reply  # noqa: E402


GENERATION_REQUEST = "Please come up with hypothesis"


@dataclass(frozen=True)
class PatchPair:
    pair_id: int
    clean_row_index: int
    corrupt_row_index: int
    clean_row: dict[str, Any]
    corrupt_row: dict[str, Any]
    key: tuple[str, str, bool]
    foil_hypothesis: str
    gold_hypothesis: str


@dataclass(frozen=True)
class ScoredSequence:
    text: str
    token_ids: list[int]
    logprob: float
    mean_logprob: float


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


def hypothesis_key(row: dict[str, Any]) -> tuple[str, str, bool]:
    hypothesis = row["ontology_fol_structured"]["hypothesis"]
    return (
        str(hypothesis["subject"]).strip().lower(),
        str(hypothesis["predicate"]).strip().lower(),
        bool(hypothesis["negated"]),
    )


def plural_variants(value: str) -> list[str]:
    value = value.strip().lower()
    variants = [value]
    if value.endswith("s"):
        variants.append(value + "es")
    else:
        variants.append(value + "s")
    return list(dict.fromkeys(variants))


def token_variants(tokenizer, text: str) -> list[list[int]]:
    variants: list[list[int]] = []
    for raw in (text, text.capitalize(), " " + text, " " + text.capitalize()):
        ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
        if ids and ids not in variants:
            variants.append(list(ids))
    return variants


def find_subsequence_positions(haystack: list[int], needles: list[list[int]]) -> list[int]:
    positions: list[int] = []
    for needle in needles:
        n = len(needle)
        if n == 0 or n > len(haystack):
            continue
        for start in range(0, len(haystack) - n + 1):
            if haystack[start : start + n] == needle:
                positions.append(start)
    return sorted(set(positions))


def choose_position(positions: list[int], *, before: int | None, policy: str) -> int | None:
    candidates = [pos for pos in positions if before is None or pos < before]
    if not candidates:
        candidates = positions
    if not candidates:
        return None
    if policy == "first":
        return min(candidates)
    if policy == "last":
        return max(candidates)
    raise ValueError(f"unknown position policy {policy!r}")


def find_landmarks(
    *,
    tokenizer,
    token_ids: list[int],
    row: dict[str, Any],
) -> dict[str, int | None]:
    subject, predicate, _negated = hypothesis_key(row)
    q_positions = find_subsequence_positions(
        token_ids,
        token_variants(tokenizer, GENERATION_REQUEST),
    )
    q_pos = choose_position(q_positions, before=None, policy="first")

    subject_needles: list[list[int]] = []
    for variant in plural_variants(subject):
        subject_needles.extend(token_variants(tokenizer, variant))
    predicate_needles = token_variants(tokenizer, predicate)

    subject_positions = find_subsequence_positions(token_ids, subject_needles)
    predicate_positions = find_subsequence_positions(token_ids, predicate_needles)

    return {
        "last_prompt": len(token_ids) - 1,
        "question_stem": q_pos,
        "subject": choose_position(subject_positions, before=q_pos, policy="last"),
        "predicate": choose_position(predicate_positions, before=q_pos, policy="last"),
    }


def select_pairs(
    *,
    rows: list[dict[str, Any]],
    jsonl_path: Path,
    splits_path: Path,
    split_family: str,
    split: str,
    limit: int,
    seed: int,
) -> tuple[list[PatchPair], dict[str, Any]]:
    assignments = read_split_assignments(splits_path)
    source_file = str(jsonl_path)
    clean_by_key: dict[tuple[str, str, bool], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    corrupt_by_key: dict[tuple[str, str, bool], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    skipped_corrupt_no_foil = 0

    for row_index, row in enumerate(rows):
        assignment = assignments.get((source_file, row_index))
        if assignment is None or assignment.get(f"{split_family}_split") != split:
            continue
        if row.get("parse_failed"):
            continue
        key = hypothesis_key(row)
        if int(row.get("height", -1)) == 1 and bool(row.get("is_correct_strong")):
            clean_by_key[key].append((row_index, row))
        elif int(row.get("height", -1)) == 4 and not bool(row.get("is_correct_strong")):
            if emitted_wrong_foil(row) is None:
                skipped_corrupt_no_foil += 1
                continue
            corrupt_by_key[key].append((row_index, row))

    rng = random.Random(seed)
    pairs: list[PatchPair] = []
    shared_keys = sorted(set(clean_by_key) & set(corrupt_by_key))
    rng.shuffle(shared_keys)
    for key in shared_keys:
        clean_candidates = list(clean_by_key[key])
        corrupt_candidates = list(corrupt_by_key[key])
        rng.shuffle(clean_candidates)
        rng.shuffle(corrupt_candidates)
        for clean, corrupt in zip(clean_candidates, corrupt_candidates, strict=False):
            clean_index, clean_row = clean
            corrupt_index, corrupt_row = corrupt
            forced = build_margin_prompt(
                corrupt_row,
                row_index=corrupt_index,
                option_seed=seed,
                foil_source="stage1_model_output",
            )
            pairs.append(
                PatchPair(
                    pair_id=len(pairs),
                    clean_row_index=clean_index,
                    corrupt_row_index=corrupt_index,
                    clean_row=clean_row,
                    corrupt_row=corrupt_row,
                    key=key,
                    foil_hypothesis=forced.foil_hypothesis,
                    gold_hypothesis=forced.gold_hypothesis,
                )
            )
            if len(pairs) >= limit:
                break
        if len(pairs) >= limit:
            break

    summary = {
        "source_file": source_file,
        "split_family": split_family,
        "split": split,
        "pair_match": "full_gold_hypothesis_h1_correct_to_h4_incorrect",
        "limit": limit,
        "seed": seed,
        "shared_full_hypothesis_keys": len(set(clean_by_key) & set(corrupt_by_key)),
        "available_pair_capacity": sum(
            min(len(clean_by_key[key]), len(corrupt_by_key[key]))
            for key in set(clean_by_key) & set(corrupt_by_key)
        ),
        "selected_pairs": len(pairs),
        "h1_clean_rows": sum(len(value) for value in clean_by_key.values()),
        "h4_corrupt_rows": sum(len(value) for value in corrupt_by_key.values()),
        "skipped_corrupt_no_foil": skipped_corrupt_no_foil,
    }
    return pairs, summary


def score_sequence_logprob(
    *,
    model,
    hook_name: str | None,
    patch_vector: torch.Tensor | None,
    patch_position: int | None,
    prompt_token_ids: list[int],
    candidate_text: str,
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
    state = {"calls": 0, "applications": 0}

    def patch_hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        state["calls"] += 1
        if patch_vector is None or patch_position is None or patch_position >= act.shape[1]:
            return act
        vector = patch_vector.to(device=act.device, dtype=act.dtype)
        act[:, patch_position, :] = vector
        state["applications"] += 1
        return act

    with torch.inference_mode():
        if hook_name is None or patch_vector is None or patch_position is None:
            logits = model(tokens, return_type="logits", prepend_bos=False)
        else:
            with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
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
        state,
    )


def capture_vectors(
    *,
    model,
    hook_names_by_layer: dict[int, str],
    token_ids: list[int],
    layers: list[int],
    landmarks: dict[str, int | None],
) -> dict[tuple[int, str], torch.Tensor]:
    input_device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    captured: dict[tuple[int, str], torch.Tensor] = {}
    valid_landmarks = {name: pos for name, pos in landmarks.items() if pos is not None}

    def make_hook(layer: int):
        def hook_fn(act: torch.Tensor, hook) -> None:  # noqa: ARG001
            for name, pos in valid_landmarks.items():
                if pos < act.shape[1]:
                    captured[(layer, name)] = act[0, pos, :].detach().cpu().float()

        return hook_fn

    hooks = [(hook_names_by_layer[layer], make_hook(layer)) for layer in layers]
    with torch.inference_mode():
        model.run_with_hooks(tokens, return_type=None, prepend_bos=False, fwd_hooks=hooks)
    return captured


def make_generation_patch_hook(
    *,
    patch_vector: torch.Tensor,
    patch_position: int,
) -> tuple[Any, dict[str, int]]:
    state = {"calls": 0, "applications": 0}

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        state["calls"] += 1
        if state["calls"] > 1 or patch_position >= act.shape[1]:
            return act
        act[:, patch_position, :] = patch_vector.to(device=act.device, dtype=act.dtype)
        state["applications"] += 1
        return act

    return hook_fn, state


def summarize_patch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["patch_mode"], int(row["layer"]), row["landmark"])].append(row)
    out: dict[str, Any] = {}
    for (mode, layer, landmark), group in sorted(groups.items()):
        key = f"{mode}_L{layer}_{landmark}"
        out[key] = {
            "n": len(group),
            "mean_margin": sum(float(row["patched_margin_gold_minus_foil"]) for row in group) / len(group),
            "mean_margin_delta": sum(float(row["margin_delta_vs_corrupt"]) for row in group) / len(group),
            "mean_recovery_fraction": sum(float(row["recovery_fraction"]) for row in group) / len(group),
            "margin_improved_count": sum(float(row["margin_delta_vs_corrupt"]) > 0 for row in group),
            "positive_recovery_count": sum(float(row["recovery_fraction"]) > 0 for row in group),
            "above_0p25_recovery_count": sum(float(row["recovery_fraction"]) >= 0.25 for row in group),
            "generated_strong_accuracy": (
                sum(bool(row.get("generated_is_correct_strong")) for row in group) / len(group)
                if "generated_is_correct_strong" in group[0]
                else None
            ),
        }
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pair-limit", type=int, default=8)
    parser.add_argument("--pair-seed", type=int, default=20260501)
    parser.add_argument("--layers", default="30,35,40,45,50")
    parser.add_argument("--landmarks", default="last_prompt,subject,predicate,question_stem")
    parser.add_argument("--patch-modes", default="clean,noise")
    parser.add_argument("--noise-seed", type=int, default=20260502)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/patching/clean_to_corrupt_27b_property_margin_pilot.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/clean_to_corrupt_patching_27b_property_margin_pilot.json"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    started = time.time()
    torch.set_grad_enabled(False)
    rows = read_jsonl(args.jsonl)
    layers = parse_int_list(args.layers)
    landmarks = [part.strip() for part in args.landmarks.split(",") if part.strip()]
    patch_modes = [part.strip() for part in args.patch_modes.split(",") if part.strip()]
    unknown_modes = sorted(set(patch_modes) - {"clean", "noise"})
    if unknown_modes:
        raise ValueError(f"unknown patch mode(s): {unknown_modes}")

    pairs, pair_summary = select_pairs(
        rows=rows,
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        split_family=args.split_family,
        split=args.split,
        limit=args.pair_limit,
        seed=args.pair_seed,
    )
    if len(pairs) < args.pair_limit:
        print(f"warning: selected only {len(pairs)} pairs for requested limit {args.pair_limit}", flush=True)
    if not pairs:
        raise ValueError("no clean/corrupt pairs selected")

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        dry_pairs = []
        missing_counts: dict[str, int] = defaultdict(int)
        for pair in pairs:
            clean_text = render_chat_text(
                tokenizer,
                system=pair.clean_row["system_prompt"],
                user=pair.clean_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            corrupt_text = render_chat_text(
                tokenizer,
                system=pair.corrupt_row["system_prompt"],
                user=pair.corrupt_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            clean_tokens = tokenizer(clean_text, add_special_tokens=False)["input_ids"]
            corrupt_tokens = tokenizer(corrupt_text, add_special_tokens=False)["input_ids"]
            clean_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=clean_tokens, row=pair.clean_row)
            corrupt_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=corrupt_tokens, row=pair.corrupt_row)
            for landmark in landmarks:
                if clean_landmarks.get(landmark) is None:
                    missing_counts[f"clean_{landmark}"] += 1
                if corrupt_landmarks.get(landmark) is None:
                    missing_counts[f"corrupt_{landmark}"] += 1
            dry_pairs.append(
                {
                    "pair_id": pair.pair_id,
                    "clean_row_index": pair.clean_row_index,
                    "corrupt_row_index": pair.corrupt_row_index,
                    "key": pair.key,
                    "gold_hypothesis": pair.gold_hypothesis,
                    "foil_hypothesis": pair.foil_hypothesis,
                    "clean_token_count": len(clean_tokens),
                    "corrupt_token_count": len(corrupt_tokens),
                    "clean_landmarks": clean_landmarks,
                    "corrupt_landmarks": corrupt_landmarks,
                }
            )
        print(json.dumps({"selection": pair_summary, "missing_landmarks": dict(missing_counts), "pairs": dry_pairs}, indent=2, sort_keys=True, default=json_default))
        return 0

    dtype = torch_dtype(args.dtype)
    print("Stage 2 clean-to-corrupt residual patching", flush=True)
    print(f"model={args.model} task={args.task} layers={layers} landmarks={landmarks}", flush=True)
    print(f"pairs={len(pairs)} selection={pair_summary}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')} torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}", flush=True)

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    hook_names = validate_hooks(model, layers)
    hook_names_by_layer = dict(zip(layers, hook_names, strict=True))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    all_patch_rows: list[dict[str, Any]] = []
    pair_baselines: list[dict[str, Any]] = []
    missing_landmarks: dict[str, int] = defaultdict(int)

    with args.out_jsonl.open("w") as fout:
        for pair in pairs:
            clean_text = render_chat_text(
                tokenizer,
                system=pair.clean_row["system_prompt"],
                user=pair.clean_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            corrupt_text = render_chat_text(
                tokenizer,
                system=pair.corrupt_row["system_prompt"],
                user=pair.corrupt_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            clean_tokens = tokenizer(clean_text, add_special_tokens=False)["input_ids"]
            corrupt_tokens = tokenizer(corrupt_text, add_special_tokens=False)["input_ids"]
            clean_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=clean_tokens, row=pair.clean_row)
            corrupt_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=corrupt_tokens, row=pair.corrupt_row)

            corrupt_gold, _ = score_sequence_logprob(
                model=model,
                hook_name=None,
                patch_vector=None,
                patch_position=None,
                prompt_token_ids=corrupt_tokens,
                candidate_text=pair.gold_hypothesis,
            )
            corrupt_foil, _ = score_sequence_logprob(
                model=model,
                hook_name=None,
                patch_vector=None,
                patch_position=None,
                prompt_token_ids=corrupt_tokens,
                candidate_text=pair.foil_hypothesis,
            )
            clean_gold, _ = score_sequence_logprob(
                model=model,
                hook_name=None,
                patch_vector=None,
                patch_position=None,
                prompt_token_ids=clean_tokens,
                candidate_text=pair.gold_hypothesis,
            )
            clean_foil, _ = score_sequence_logprob(
                model=model,
                hook_name=None,
                patch_vector=None,
                patch_position=None,
                prompt_token_ids=clean_tokens,
                candidate_text=pair.foil_hypothesis,
            )
            corrupt_margin = corrupt_gold.logprob - corrupt_foil.logprob
            clean_margin = clean_gold.logprob - clean_foil.logprob
            denominator = clean_margin - corrupt_margin
            pair_baseline = {
                "pair_id": pair.pair_id,
                "clean_row_index": pair.clean_row_index,
                "corrupt_row_index": pair.corrupt_row_index,
                "key": pair.key,
                "gold_hypothesis": pair.gold_hypothesis,
                "foil_hypothesis": pair.foil_hypothesis,
                "clean_token_count": len(clean_tokens),
                "corrupt_token_count": len(corrupt_tokens),
                "clean_landmarks": clean_landmarks,
                "corrupt_landmarks": corrupt_landmarks,
                "corrupt_baseline_margin_gold_minus_foil": corrupt_margin,
                "clean_reference_margin_gold_minus_foil": clean_margin,
                "recovery_denominator": denominator,
            }
            pair_baselines.append(pair_baseline)
            print(
                f"pair {pair.pair_id}/{len(pairs)-1} clean={pair.clean_row_index} corrupt={pair.corrupt_row_index} "
                f"margin_corrupt={corrupt_margin:.3f} margin_clean={clean_margin:.3f}",
                flush=True,
            )

            clean_vectors = capture_vectors(
                model=model,
                hook_names_by_layer=hook_names_by_layer,
                token_ids=clean_tokens,
                layers=layers,
                landmarks=clean_landmarks,
            )
            corrupt_vectors = capture_vectors(
                model=model,
                hook_names_by_layer=hook_names_by_layer,
                token_ids=corrupt_tokens,
                layers=layers,
                landmarks=corrupt_landmarks,
            )

            for layer in layers:
                for landmark in landmarks:
                    clean_pos = clean_landmarks.get(landmark)
                    corrupt_pos = corrupt_landmarks.get(landmark)
                    if clean_pos is None or corrupt_pos is None:
                        if clean_pos is None:
                            missing_landmarks[f"clean_{landmark}"] += 1
                        if corrupt_pos is None:
                            missing_landmarks[f"corrupt_{landmark}"] += 1
                        continue
                    clean_vector = clean_vectors.get((layer, landmark))
                    corrupt_vector = corrupt_vectors.get((layer, landmark))
                    if clean_vector is None or corrupt_vector is None:
                        missing_landmarks[f"capture_L{layer}_{landmark}"] += 1
                        continue
                    delta = clean_vector - corrupt_vector
                    delta_norm = float(torch.linalg.vector_norm(delta).item())
                    vectors: dict[str, torch.Tensor] = {}
                    if "clean" in patch_modes:
                        vectors["clean"] = clean_vector
                    if "noise" in patch_modes:
                        rng = np.random.default_rng(args.noise_seed + pair.pair_id * 100003 + layer * 1009 + len(landmark))
                        noise = torch.as_tensor(rng.standard_normal(clean_vector.shape[0]), dtype=torch.float32)
                        noise = noise / torch.linalg.vector_norm(noise).clamp_min(1e-12) * delta_norm
                        vectors["noise"] = corrupt_vector + noise

                    for patch_mode, patch_vector in vectors.items():
                        patched_gold, gold_hook = score_sequence_logprob(
                            model=model,
                            hook_name=hook_names_by_layer[layer],
                            patch_vector=patch_vector,
                            patch_position=corrupt_pos,
                            prompt_token_ids=corrupt_tokens,
                            candidate_text=pair.gold_hypothesis,
                        )
                        patched_foil, foil_hook = score_sequence_logprob(
                            model=model,
                            hook_name=hook_names_by_layer[layer],
                            patch_vector=patch_vector,
                            patch_position=corrupt_pos,
                            prompt_token_ids=corrupt_tokens,
                            candidate_text=pair.foil_hypothesis,
                        )
                        patched_margin = patched_gold.logprob - patched_foil.logprob
                        margin_delta = patched_margin - corrupt_margin
                        recovery = margin_delta / denominator if abs(denominator) > 1e-9 else 0.0
                        row: dict[str, Any] = {
                            "schema_version": 1,
                            "pair_id": pair.pair_id,
                            "clean_row_index": pair.clean_row_index,
                            "corrupt_row_index": pair.corrupt_row_index,
                            "key": pair.key,
                            "layer": layer,
                            "hook_name": hook_names_by_layer[layer],
                            "landmark": landmark,
                            "patch_mode": patch_mode,
                            "clean_position": clean_pos,
                            "corrupt_position": corrupt_pos,
                            "clean_token_count": len(clean_tokens),
                            "corrupt_token_count": len(corrupt_tokens),
                            "gold_hypothesis": pair.gold_hypothesis,
                            "foil_hypothesis": pair.foil_hypothesis,
                            "corrupt_baseline_margin_gold_minus_foil": corrupt_margin,
                            "clean_reference_margin_gold_minus_foil": clean_margin,
                            "patched_gold_logprob": patched_gold.logprob,
                            "patched_foil_logprob": patched_foil.logprob,
                            "patched_margin_gold_minus_foil": patched_margin,
                            "margin_delta_vs_corrupt": margin_delta,
                            "recovery_denominator": denominator,
                            "recovery_fraction": recovery,
                            "patch_delta_l2": delta_norm if patch_mode == "clean" else float(torch.linalg.vector_norm(patch_vector - corrupt_vector).item()),
                            "gold_hook_applications": gold_hook["applications"],
                            "foil_hook_applications": foil_hook["applications"],
                        }
                        if args.generate:
                            hook_fn, gen_state = make_generation_patch_hook(
                                patch_vector=patch_vector,
                                patch_position=corrupt_pos,
                            )
                            with model.hooks(fwd_hooks=[(hook_names_by_layer[layer], hook_fn)]):
                                new_ids, reply = generate_one(
                                    model=model,
                                    token_ids=corrupt_tokens,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=args.do_sample,
                                    temperature=args.temperature,
                                    stop_at_eos=args.stop_at_eos,
                                    cache_dtype=dtype,
                                )
                            score = score_reply(pair.corrupt_row, reply)
                            row.update(
                                {
                                    "generated_token_count": len(new_ids),
                                    "generated_text": reply,
                                    "generation_hook_calls": gen_state["calls"],
                                    "generation_hook_applications": gen_state["applications"],
                                    "generated_is_correct_strong": bool(score["is_correct_strong"]),
                                    "generated_is_correct_weak": bool(score["is_correct_weak"]),
                                    "generated_parse_failed": bool(score["parse_failed"]),
                                    "generated_parsed_hypotheses": score["parsed_hypotheses"],
                                    "generated_quality_score": score["quality_score"],
                                }
                            )
                        all_patch_rows.append(row)
                        fout.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                        fout.flush()
                        print(
                            f"  L{layer} {landmark} {patch_mode}: "
                            f"delta={margin_delta:.3f} recovery={recovery:.3f}",
                            flush=True,
                        )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_patch_clean_to_corrupt.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "selection": pair_summary,
        "layers": layers,
        "landmarks": landmarks,
        "patch_modes": patch_modes,
        "generation": {
            "enabled": args.generate,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "missing_landmarks": dict(sorted(missing_landmarks.items())),
        "pair_baselines": pair_baselines,
        "summary": summarize_patch_rows(all_patch_rows),
        "out_jsonl": str(args.out_jsonl),
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
