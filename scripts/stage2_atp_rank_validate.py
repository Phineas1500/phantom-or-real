#!/usr/bin/env python3
"""AtP-style first-order site ranking with exact patch validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
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

from scripts.stage2_das_subspace_interchange import (  # noqa: E402
    load_positive_control_summary,
    summarize_baselines,
)
from scripts.stage2_patch_clean_to_corrupt import (  # noqa: E402
    ScoredSequence,
    capture_vectors,
    find_landmarks,
    json_default,
    score_sequence_logprob,
    select_pairs,
    torch_dtype,
)
from src.activations import input_device_for_model, load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.stage2_causal_abstraction import make_experiment_report  # noqa: E402
from src.stage2_probes import read_jsonl  # noqa: E402
from src.stage2_steering import parse_int_list  # noqa: E402


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def vector_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.detach().float()).item())


def tokenized_candidate(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"candidate text produced no tokens: {text!r}")
    return list(ids)


def score_candidate_with_gradients(
    *,
    model,
    token_ids: list[int],
    candidate_text: str,
    layer: int,
    landmarks: dict[str, int | None],
) -> tuple[ScoredSequence, dict[tuple[int, str], torch.Tensor], dict[str, Any]]:
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("model has no tokenizer")
    candidate_ids = tokenized_candidate(tokenizer, candidate_text)
    input_ids = token_ids + candidate_ids[:-1]
    target_ids = candidate_ids
    input_device = input_device_for_model(model)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=input_device)
    valid_landmarks = {name: pos for name, pos in landmarks.items() if pos is not None}
    stop_at_layer = layer + 1
    if stop_at_layer > model.cfg.n_layers:
        raise ValueError(f"cannot truncate after layer {layer}; model has {model.cfg.n_layers} layers")

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        residual = model(
            tokens,
            return_type=None,
            prepend_bos=False,
            stop_at_layer=stop_at_layer,
        )
    residual = residual.detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(
            residual,
            return_type="logits",
            prepend_bos=False,
            start_at_layer=stop_at_layer,
        )
        positions = torch.arange(
            len(token_ids) - 1,
            len(token_ids) - 1 + len(candidate_ids),
            device=logits.device,
        )
        target = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
        selected_logits = logits[0, positions, :]
        log_probs = torch.log_softmax(selected_logits.float(), dim=-1)
        token_logprob = log_probs[torch.arange(len(candidate_ids), device=logits.device), target]
        total_logprob_tensor = token_logprob.sum()
        total_logprob = float(total_logprob_tensor.detach().cpu())
        total_logprob_tensor.backward()

    grads: dict[tuple[int, str], torch.Tensor] = {}
    if residual.grad is not None:
        for landmark, pos in valid_landmarks.items():
            if pos < residual.grad.shape[1]:
                grads[(layer, landmark)] = residual.grad[0, pos, :].detach().cpu().float()
    hook_state: dict[str, Any] = {
        "mode": "truncated_tail_gradient",
        "layer": layer,
        "stop_at_layer": stop_at_layer,
        "start_at_layer": stop_at_layer,
        "captured_landmarks": sorted(landmark for (_layer, landmark) in grads),
    }
    model.zero_grad(set_to_none=True)
    del logits, selected_logits, log_probs, token_logprob, total_logprob_tensor, residual
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        ScoredSequence(
            text=candidate_text,
            token_ids=candidate_ids,
            logprob=total_logprob,
            mean_logprob=total_logprob / len(candidate_ids),
        ),
        grads,
        hook_state,
    )

def make_noise_like(delta: torch.Tensor, *, seed: int, norm: float | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    noise = torch.as_tensor(rng.standard_normal(delta.shape[0]), dtype=torch.float32)
    noise = noise / torch.linalg.vector_norm(noise).clamp_min(1e-12)
    return noise * (vector_norm(delta) if norm is None else norm)


def summarize_ranking(rows: list[dict[str, Any]], *, patch_direction: str) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["layer"]), row["landmark"])].append(row)
    out: dict[str, Any] = {}
    for (layer, landmark), group in sorted(groups.items()):
        approx = [float(row["approx_margin_delta"]) for row in group]
        desired = [float(row["desired_effect_fraction"]) for row in group]
        captures = [float(row["delta_l2"]) for row in group]
        key = f"L{layer}_{landmark}"
        out[key] = {
            "n": len(group),
            "mean_approx_margin_delta": float(np.mean(approx)),
            "std_approx_margin_delta": float(np.std(approx, ddof=1)) if len(approx) > 1 else 0.0,
            "mean_desired_effect_fraction": float(np.mean(desired)),
            "positive_desired_count": int(sum(value > 0 for value in desired)),
            "mean_delta_l2": float(np.mean(captures)),
            "rank_score": float(np.mean(desired)),
            "patch_direction": patch_direction,
        }
    return out


def summarize_exact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["validation_mode"], int(row["layer"]), row["landmark"])].append(row)
    out: dict[str, Any] = {}
    for (mode, layer, landmark), group in sorted(groups.items()):
        deltas = [float(row["margin_delta_vs_baseline"]) for row in group]
        recoveries = [float(row["recovery_fraction"]) for row in group]
        breakages = [float(row["breakage_fraction"]) for row in group if row.get("breakage_fraction") is not None]
        key = f"{mode}_L{layer}_{landmark}"
        out[key] = {
            "n": len(group),
            "mean_margin": float(np.mean([float(row["patched_margin_gold_minus_foil"]) for row in group])),
            "mean_margin_delta": float(np.mean(deltas)),
            "std_margin_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "mean_recovery_fraction": float(np.mean(recoveries)),
            "mean_breakage_fraction": float(np.mean(breakages)) if breakages else None,
            "margin_improved_count": int(sum(delta > 0 for delta in deltas)),
            "margin_decreased_count": int(sum(delta < 0 for delta in deltas)),
            "false_to_true_repair_count": int(sum(bool(row["false_to_true_repair"]) for row in group)),
            "true_to_false_disruption_count": int(sum(bool(row["true_to_false_disruption"]) for row in group)),
        }
    for key, value in list(out.items()):
        if not key.startswith("source_"):
            continue
        _, layer_part, *landmark_parts = key.split("_")
        layer = int(layer_part[1:])
        landmark = "_".join(landmark_parts)
        noise = out.get(f"matched_gaussian_L{layer}_{landmark}")
        if not noise:
            continue
        noise_std = float(noise["std_margin_delta"])
        value["matched_gaussian_mean_margin_delta"] = noise["mean_margin_delta"]
        value["matched_gaussian_std_margin_delta"] = noise_std
        value["mean_delta_minus_matched_gaussian"] = (
            float(value["mean_margin_delta"]) - float(noise["mean_margin_delta"])
        )
        value["mean_delta_vs_noise_sigma"] = (
            value["mean_delta_minus_matched_gaussian"] / noise_std if noise_std > 1e-9 else None
        )
    return out


def rank_sites(ranking_summary: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = []
    for key, value in ranking_summary.items():
        layer_text, landmark = key.split("_", 1)
        ranked.append(
            {
                "site": key,
                "layer": int(layer_text.removeprefix("L")),
                "landmark": landmark,
                **value,
            }
        )
    ranked.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pair-limit", type=int, default=4)
    parser.add_argument("--pair-seed", type=int, default=20260501)
    parser.add_argument("--layers", default="35,40,45,50")
    parser.add_argument("--landmarks", default="last_prompt")
    parser.add_argument(
        "--patch-direction",
        choices=("clean_to_corrupt", "corrupt_to_clean"),
        default="corrupt_to_clean",
    )
    parser.add_argument("--top-k-sites", type=int, default=3)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--noise-seed", type=int, default=20260606)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--positive-control-report",
        type=Path,
        default=Path("docs/positive_control_format_gemma3_27b_l45.json"),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/atp/atp_rank_validate_27b_property_corrupt_to_clean_pilot.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/atp_rank_validate_27b_property_corrupt_to_clean_pilot.json"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    started = time.time()
    rows = read_jsonl(args.jsonl)
    layers = parse_int_list(args.layers)
    landmarks = parse_csv(args.landmarks)
    positive_control = load_positive_control_summary(args.positive_control_report)

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
        print(
            json.dumps(
                {
                    "selection": pair_summary,
                    "layers": layers,
                    "landmarks": landmarks,
                    "patch_direction": args.patch_direction,
                    "top_k_sites": args.top_k_sites,
                    "positive_control": positive_control,
                    "missing_landmarks": dict(missing_counts),
                    "pairs": dry_pairs,
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            )
        )
        return 0

    dtype = torch_dtype(args.dtype)
    print("Stage 2 AtP-style ranking with exact patch validation", flush=True)
    print(f"model={args.model} task={args.task} direction={args.patch_direction}", flush=True)
    print(f"layers={layers} landmarks={landmarks} pairs={len(pairs)} top_k={args.top_k_sites}", flush=True)
    print(f"selection={pair_summary}", flush=True)
    print(f"positive_control={positive_control}", flush=True)
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
    ranking_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    pair_baselines: list[dict[str, Any]] = []
    pair_payloads: dict[int, dict[str, Any]] = {}
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

            if args.patch_direction == "clean_to_corrupt":
                target_tokens = corrupt_tokens
                target_landmarks = corrupt_landmarks
                baseline_margin = corrupt_margin
                reference_margin = clean_margin
                target_label = "corrupt"
                source_label = "clean"
            else:
                target_tokens = clean_tokens
                target_landmarks = clean_landmarks
                baseline_margin = clean_margin
                reference_margin = corrupt_margin
                target_label = "clean"
                source_label = "corrupt"

            pair_payloads[pair.pair_id] = {
                "pair": pair,
                "clean_tokens": clean_tokens,
                "corrupt_tokens": corrupt_tokens,
                "clean_landmarks": clean_landmarks,
                "corrupt_landmarks": corrupt_landmarks,
                "clean_vectors": clean_vectors,
                "corrupt_vectors": corrupt_vectors,
                "corrupt_margin": corrupt_margin,
                "clean_margin": clean_margin,
                "denominator": denominator,
            }

            for layer in layers:
                gold_grad_score, gold_grads, gold_hook_state = score_candidate_with_gradients(
                    model=model,
                    token_ids=target_tokens,
                    candidate_text=pair.gold_hypothesis,
                    layer=layer,
                    landmarks=target_landmarks,
                )
                foil_grad_score, foil_grads, foil_hook_state = score_candidate_with_gradients(
                    model=model,
                    token_ids=target_tokens,
                    candidate_text=pair.foil_hypothesis,
                    layer=layer,
                    landmarks=target_landmarks,
                )
                grad_margin = gold_grad_score.logprob - foil_grad_score.logprob
                layer_grads: dict[tuple[int, str], torch.Tensor] = {}
                for key in set(gold_grads) & set(foil_grads):
                    layer_grads[key] = gold_grads[key] - foil_grads[key]
                print(
                    f"  layer={layer} grad_margin={grad_margin:.3f} baseline_margin={baseline_margin:.3f}",
                    flush=True,
                )
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
                    grad = layer_grads.get((layer, landmark))
                    if grad is None:
                        missing_landmarks[f"grad_L{layer}_{landmark}"] += 1
                        continue
                    if args.patch_direction == "clean_to_corrupt":
                        source_vector = clean_vector
                        target_vector = corrupt_vector
                    else:
                        source_vector = corrupt_vector
                        target_vector = clean_vector
                    delta = (source_vector - target_vector).detach().cpu().float()
                    approx_delta = float(torch.dot(grad, delta).item())
                    recovery = approx_delta / denominator if abs(denominator) > 1e-9 else 0.0
                    breakage = -approx_delta / denominator if args.patch_direction == "corrupt_to_clean" and abs(denominator) > 1e-9 else None
                    desired = recovery if args.patch_direction == "clean_to_corrupt" else (breakage or 0.0)
                    row = {
                        "schema_version": 1,
                        "row_type": "attribution_ranking",
                        "method": "first_order_attribution_patch_ranking",
                        "patch_direction": args.patch_direction,
                        "pair_id": pair.pair_id,
                        "clean_row_index": pair.clean_row_index,
                        "corrupt_row_index": pair.corrupt_row_index,
                        "key": pair.key,
                        "layer": layer,
                        "hook_name": hook_names_by_layer[layer],
                        "landmark": landmark,
                        "clean_position": clean_pos,
                        "corrupt_position": corrupt_pos,
                        "target_label": target_label,
                        "source_label": source_label,
                        "gold_hypothesis": pair.gold_hypothesis,
                        "foil_hypothesis": pair.foil_hypothesis,
                        "baseline_margin_gold_minus_foil": baseline_margin,
                        "reference_margin_gold_minus_foil": reference_margin,
                        "grad_margin_gold_minus_foil": grad_margin,
                        "approx_margin_delta": approx_delta,
                        "recovery_denominator": denominator,
                        "approx_recovery_fraction": recovery,
                        "approx_breakage_fraction": breakage,
                        "desired_effect_fraction": desired,
                        "delta_l2": vector_norm(delta),
                        "grad_l2": vector_norm(grad),
                        "gold_gradient_state": gold_hook_state,
                        "foil_gradient_state": foil_hook_state,
                    }
                    ranking_rows.append(row)
                    fout.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                    print(
                        f"  rank-candidate L{layer} {landmark}: approx_delta={approx_delta:.3f} desired={desired:.3f}",
                        flush=True,
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranking_summary = summarize_ranking(ranking_rows, patch_direction=args.patch_direction)
        ranked_sites = rank_sites(ranking_summary)
        top_sites = ranked_sites[: args.top_k_sites]
        print(f"top_sites={top_sites}", flush=True)

        for site in top_sites:
            layer = int(site["layer"])
            landmark = str(site["landmark"])
            for pair in pairs:
                payload = pair_payloads[pair.pair_id]
                clean_tokens = payload["clean_tokens"]
                corrupt_tokens = payload["corrupt_tokens"]
                clean_landmarks = payload["clean_landmarks"]
                corrupt_landmarks = payload["corrupt_landmarks"]
                clean_vectors = payload["clean_vectors"]
                corrupt_vectors = payload["corrupt_vectors"]
                corrupt_margin = float(payload["corrupt_margin"])
                clean_margin = float(payload["clean_margin"])
                denominator = float(payload["denominator"])
                clean_pos = clean_landmarks.get(landmark)
                corrupt_pos = corrupt_landmarks.get(landmark)
                clean_vector = clean_vectors.get((layer, landmark))
                corrupt_vector = corrupt_vectors.get((layer, landmark))
                if clean_pos is None or corrupt_pos is None or clean_vector is None or corrupt_vector is None:
                    continue
                if args.patch_direction == "clean_to_corrupt":
                    source_vector = clean_vector
                    target_vector = corrupt_vector
                    patch_position = corrupt_pos
                    prompt_tokens = corrupt_tokens
                    baseline_margin = corrupt_margin
                    reference_margin = clean_margin
                    baseline_label = "corrupt"
                    reference_label = "clean"
                else:
                    source_vector = corrupt_vector
                    target_vector = clean_vector
                    patch_position = clean_pos
                    prompt_tokens = clean_tokens
                    baseline_margin = clean_margin
                    reference_margin = corrupt_margin
                    baseline_label = "clean"
                    reference_label = "corrupt"
                delta = (source_vector - target_vector).detach().cpu().float()
                validation_vectors: list[tuple[str, torch.Tensor, int | None]] = [("source", source_vector, None)]
                delta_norm = vector_norm(delta)
                for repeat in range(args.noise_repeats):
                    seed = args.noise_seed + pair.pair_id * 100003 + layer * 1009 + len(landmark) * 389 + repeat * 17
                    noise = make_noise_like(delta, seed=seed, norm=delta_norm)
                    validation_vectors.append(("matched_gaussian", target_vector + noise, repeat))
                for mode, patch_vector, repeat in validation_vectors:
                    patched_gold, gold_hook = score_sequence_logprob(
                        model=model,
                        hook_name=hook_names_by_layer[layer],
                        patch_vector=patch_vector,
                        patch_position=patch_position,
                        prompt_token_ids=prompt_tokens,
                        candidate_text=pair.gold_hypothesis,
                    )
                    patched_foil, foil_hook = score_sequence_logprob(
                        model=model,
                        hook_name=hook_names_by_layer[layer],
                        patch_vector=patch_vector,
                        patch_position=patch_position,
                        prompt_token_ids=prompt_tokens,
                        candidate_text=pair.foil_hypothesis,
                    )
                    patched_margin = patched_gold.logprob - patched_foil.logprob
                    margin_delta = patched_margin - baseline_margin
                    recovery = margin_delta / denominator if abs(denominator) > 1e-9 else 0.0
                    breakage = -margin_delta / denominator if args.patch_direction == "corrupt_to_clean" and abs(denominator) > 1e-9 else None
                    row = {
                        "schema_version": 1,
                        "row_type": "exact_validation",
                        "method": "exact_patch_validation_of_attribution_ranked_site",
                        "patch_direction": args.patch_direction,
                        "site_rank": site["rank"],
                        "site_rank_score": site["rank_score"],
                        "pair_id": pair.pair_id,
                        "clean_row_index": pair.clean_row_index,
                        "corrupt_row_index": pair.corrupt_row_index,
                        "key": pair.key,
                        "layer": layer,
                        "hook_name": hook_names_by_layer[layer],
                        "landmark": landmark,
                        "validation_mode": mode,
                        "noise_repeat": repeat,
                        "clean_position": clean_pos,
                        "corrupt_position": corrupt_pos,
                        "patch_position": patch_position,
                        "gold_hypothesis": pair.gold_hypothesis,
                        "foil_hypothesis": pair.foil_hypothesis,
                        "baseline_label": baseline_label,
                        "reference_label": reference_label,
                        "baseline_margin_gold_minus_foil": baseline_margin,
                        "reference_margin_gold_minus_foil": reference_margin,
                        "patched_gold_logprob": patched_gold.logprob,
                        "patched_foil_logprob": patched_foil.logprob,
                        "patched_margin_gold_minus_foil": patched_margin,
                        "margin_delta_vs_baseline": margin_delta,
                        "margin_delta_vs_corrupt": patched_margin - corrupt_margin,
                        "recovery_denominator": denominator,
                        "recovery_fraction": recovery,
                        "breakage_fraction": breakage,
                        "patch_delta_l2": delta_norm if mode == "source" else vector_norm(patch_vector - target_vector),
                        "false_to_true_repair": bool(
                            args.patch_direction == "clean_to_corrupt" and baseline_margin < 0 and patched_margin > 0
                        ),
                        "true_to_false_disruption": bool(
                            args.patch_direction == "corrupt_to_clean" and baseline_margin > 0 and patched_margin < 0
                        ),
                        "gold_hook_applications": gold_hook["applications"],
                        "foil_hook_applications": foil_hook["applications"],
                    }
                    exact_rows.append(row)
                    fout.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                    print(
                        f"  exact {mode} rank={site['rank']} L{layer} {landmark} pair={pair.pair_id}: "
                        f"delta={margin_delta:.3f} recovery={recovery:.3f}"
                        + (f" breakage={breakage:.3f}" if breakage is not None else ""),
                        flush=True,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    exact_summary = summarize_exact(exact_rows)
    paired_flips = {
        key: {
            "n": value["n"],
            "false_to_true_repair_count": value["false_to_true_repair_count"],
            "true_to_false_disruption_count": value["true_to_false_disruption_count"],
        }
        for key, value in exact_summary.items()
    }
    matched_noise_summary = {
        key: value for key, value in exact_summary.items() if key.startswith("matched_gaussian_")
    }
    controls = ["regenerated_baseline", "matched_gaussian_noise", "exact_patch_validation"]
    if positive_control and positive_control.get("passed_positive_control_gate") is True:
        controls.append("positive_control")

    experiment_report = make_experiment_report(
        model=args.model,
        task=args.task,
        target_variable="gold_vs_foil_margin",
        split=args.split,
        site_or_layer=",".join(f"L{layer}:{','.join(landmarks)}" for layer in layers),
        method="first_order_attribution_ranking_with_exact_patch_validation",
        representation_type="patched_residual_state",
        result_type="causal",
        controls=controls,
        n=len(pairs),
        baseline_metrics=summarize_baselines(pair_baselines),
        intervention_metrics={
            "attribution_ranking": ranking_summary,
            "ranked_sites": ranked_sites,
            "top_validated_sites": top_sites,
            "exact_validation": exact_summary,
        },
        paired_flips=paired_flips,
        parse_fail_rate=None,
        matched_noise_summary=matched_noise_summary,
        causal_abstraction_claim=(
            "AtP-style preflight for localizing residual-stream sites whose first-order "
            "grad-dot-delta estimate predicts movement of the InAbHyD gold_vs_foil_margin variable. "
            "The attribution ranking is treated only as a hypothesis generator; causal evidence comes "
            "from exact source-patch validation against matched-Gaussian controls. This is not a full AtP* "
            "implementation with Q/K correction or GradDrop."
        ),
        notes=[
            "Ranking score is approximate recovery for clean_to_corrupt and approximate breakage for corrupt_to_clean.",
            "Exact validation is run only on the top-ranked sites in this checkpoint.",
            "Teacher-forced gold-vs-foil margins are used; no free-form parse rate is available.",
        ],
    )

    report = {
        **experiment_report,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_atp_rank_validate.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "model_key": args.model_key,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "selection": pair_summary,
        "patch_direction": args.patch_direction,
        "layers": layers,
        "landmarks": landmarks,
        "top_k_sites": args.top_k_sites,
        "positive_control": positive_control,
        "environment": {
            "transformer_lens": package_version("transformer-lens"),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "missing_landmarks": dict(sorted(missing_landmarks.items())),
        "pair_baselines": pair_baselines,
        "ranking_row_count": len(ranking_rows),
        "exact_validation_row_count": len(exact_rows),
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
