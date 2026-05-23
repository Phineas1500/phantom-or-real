#!/usr/bin/env python3
"""Patch Qwen Hugging Face residual states between clean and corrupt prompts.

This mirrors the Gemma TransformerLens patching pilot but hooks
``model.model.layers[L]`` directly, which is the same residual site used by the
Qwen-Scope extraction scripts.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_patch_clean_to_corrupt import (  # noqa: E402
    PatchPair,
    find_landmarks,
    json_default,
    select_pairs,
    summarize_patch_rows,
)
from src.activations import render_chat_text  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.stage2_probes import read_jsonl  # noqa: E402
from src.stage2_steering import parse_int_list  # noqa: E402


@dataclass(frozen=True)
class ScoredSequence:
    text: str
    token_ids: list[int]
    logprob: float
    mean_logprob: float


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def first_parameter_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def input_device_for_model(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        return first_parameter_device(model.get_input_embeddings())
    return first_parameter_device(model)


def load_hf_model(
    model_name: str,
    *,
    dtype: torch.dtype,
    device_map: str,
    device: str,
    attn_implementation: str | None,
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map != "none":
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device_map == "none":
        model.to(device)
    model.eval()
    model.tokenizer = tokenizer
    validate_hf_layers(model, [])
    return model, tokenizer


def validate_hf_layers(model: torch.nn.Module, layers: list[int]) -> None:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("expected a Hugging Face causal LM with model.model.layers")
    n_layers = len(model.model.layers)
    missing = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if missing:
        raise ValueError(f"layers {missing} outside available range 0..{n_layers - 1}")


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden_in_output(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def score_sequence_logprob(
    *,
    model,
    layer: int | None,
    patch_vector: torch.Tensor | None,
    patch_position: int | None,
    prompt_token_ids: list[int],
    candidate_text: str,
) -> tuple[ScoredSequence, dict[str, int]]:
    tokenizer = model.tokenizer
    candidate_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
    if not candidate_ids:
        raise ValueError(f"candidate text produced no tokens: {candidate_text!r}")

    input_ids = prompt_token_ids + candidate_ids[:-1]
    target_ids = candidate_ids
    device = input_device_for_model(model)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    state = {"calls": 0, "applications": 0}
    handle = None

    def patch_hook(_module, _inputs, output):
        state["calls"] += 1
        if patch_vector is None or patch_position is None:
            return output
        hidden = hidden_from_output(output)
        if patch_position >= hidden.shape[1]:
            return output
        patched = hidden.clone()
        patched[:, patch_position, :] = patch_vector.to(device=hidden.device, dtype=hidden.dtype)
        state["applications"] += 1
        return replace_hidden_in_output(output, patched)

    try:
        if layer is not None and patch_vector is not None and patch_position is not None:
            handle = model.model.layers[layer].register_forward_hook(patch_hook)
        with torch.inference_mode():
            outputs = model(input_ids=tokens, use_cache=False)
            logits = outputs.logits
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
    finally:
        if handle is not None:
            handle.remove()

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
    token_ids: list[int],
    layers: list[int],
    landmarks: dict[str, int | None],
) -> dict[tuple[int, str], torch.Tensor]:
    device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    captured: dict[tuple[int, str], torch.Tensor] = {}
    valid_landmarks = {name: pos for name, pos in landmarks.items() if pos is not None}

    def make_hook(layer: int):
        def hook_fn(_module, _inputs, output) -> None:
            hidden = hidden_from_output(output)
            for name, pos in valid_landmarks.items():
                if pos < hidden.shape[1]:
                    captured[(layer, name)] = hidden[0, pos, :].detach().cpu().float()

        return hook_fn

    handles = [model.model.layers[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    try:
        with torch.inference_mode():
            model(input_ids=tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def render_tokens(
    *,
    tokenizer,
    row: dict[str, Any],
    model_name: str,
    chat_template_kwargs: dict[str, Any] | None,
) -> list[int]:
    text = render_chat_text(
        tokenizer,
        system=row["system_prompt"],
        user=row["prompt_text"],
        model_name=model_name,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("rendered prompt produced no tokens")
    return list(token_ids)


def pair_to_dry_row(
    *,
    tokenizer,
    pair: PatchPair,
    model_name: str,
    chat_template_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    clean_tokens = render_tokens(
        tokenizer=tokenizer,
        row=pair.clean_row,
        model_name=model_name,
        chat_template_kwargs=chat_template_kwargs,
    )
    corrupt_tokens = render_tokens(
        tokenizer=tokenizer,
        row=pair.corrupt_row,
        model_name=model_name,
        chat_template_kwargs=chat_template_kwargs,
    )
    return {
        "pair_id": pair.pair_id,
        "clean_row_index": pair.clean_row_index,
        "corrupt_row_index": pair.corrupt_row_index,
        "key": pair.key,
        "gold_hypothesis": pair.gold_hypothesis,
        "foil_hypothesis": pair.foil_hypothesis,
        "clean_token_count": len(clean_tokens),
        "corrupt_token_count": len(corrupt_tokens),
        "clean_landmarks": find_landmarks(tokenizer=tokenizer, token_ids=clean_tokens, row=pair.clean_row),
        "corrupt_landmarks": find_landmarks(tokenizer=tokenizer, token_ids=corrupt_tokens, row=pair.corrupt_row),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_subtype.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model-key", default="qwen35_27b")
    parser.add_argument("--task", default="infer_subtype")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pair-limit", type=int, default=7)
    parser.add_argument("--pair-seed", type=int, default=20260523)
    parser.add_argument("--layers", default="35,40,45")
    parser.add_argument("--landmarks", default="last_prompt")
    parser.add_argument("--patch-modes", default="clean,noise")
    parser.add_argument(
        "--patch-direction",
        choices=("clean_to_corrupt", "corrupt_to_clean"),
        default="clean_to_corrupt",
    )
    parser.add_argument("--noise-seed", type=int, default=20260524)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/patching/qwen35_27b_subtype_clean_to_corrupt_margin_pilot.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/qwen35_27b_subtype_clean_to_corrupt_patching_margin_pilot.json"),
    )
    return parser


def main() -> int:
    load_env()
    args = build_arg_parser().parse_args()
    started = time.time()
    torch.set_grad_enabled(False)
    rows = read_jsonl(args.jsonl)
    layers = parse_int_list(args.layers)
    landmarks = [part.strip() for part in args.landmarks.split(",") if part.strip()]
    patch_modes = [part.strip() for part in args.patch_modes.split(",") if part.strip()]
    unknown_modes = sorted(set(patch_modes) - {"clean", "corrupt", "source", "noise"})
    if unknown_modes:
        raise ValueError(f"unknown patch mode(s): {unknown_modes}")
    if args.patch_direction == "clean_to_corrupt":
        unsupported_modes = sorted(set(patch_modes) - {"clean", "source", "noise"})
    else:
        unsupported_modes = sorted(set(patch_modes) - {"corrupt", "source", "noise"})
    if unsupported_modes:
        raise ValueError(f"patch mode(s) {unsupported_modes} do not match {args.patch_direction}")

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
        raise ValueError(f"no clean/corrupt pairs selected: {pair_summary}")

    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else None
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        dry_pairs = [
            pair_to_dry_row(
                tokenizer=tokenizer,
                pair=pair,
                model_name=args.model,
                chat_template_kwargs=chat_template_kwargs,
            )
            for pair in pairs
        ]
        missing_counts: dict[str, int] = defaultdict(int)
        for row in dry_pairs:
            for landmark in landmarks:
                if row["clean_landmarks"].get(landmark) is None:
                    missing_counts[f"clean_{landmark}"] += 1
                if row["corrupt_landmarks"].get(landmark) is None:
                    missing_counts[f"corrupt_{landmark}"] += 1
        print(
            json.dumps(
                {
                    "selection": pair_summary,
                    "layers": layers,
                    "landmarks": landmarks,
                    "missing_landmarks": dict(sorted(missing_counts.items())),
                    "pairs": dry_pairs,
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            )
        )
        return 0

    dtype = torch_dtype(args.dtype)
    print("Qwen HF residual patching", flush=True)
    print(f"direction={args.patch_direction} model={args.model} task={args.task}", flush=True)
    print(f"layers={layers} landmarks={landmarks} pairs={len(pairs)} selection={pair_summary}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}", flush=True)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        device=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    validate_hf_layers(model, layers)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    all_patch_rows: list[dict[str, Any]] = []
    pair_baselines: list[dict[str, Any]] = []
    missing_landmarks: dict[str, int] = defaultdict(int)

    with args.out_jsonl.open("w") as fout:
        for pair in pairs:
            clean_tokens = render_tokens(
                tokenizer=tokenizer,
                row=pair.clean_row,
                model_name=args.model,
                chat_template_kwargs=chat_template_kwargs,
            )
            corrupt_tokens = render_tokens(
                tokenizer=tokenizer,
                row=pair.corrupt_row,
                model_name=args.model,
                chat_template_kwargs=chat_template_kwargs,
            )
            if len(clean_tokens) > args.n_ctx or len(corrupt_tokens) > args.n_ctx:
                raise ValueError(
                    f"pair {pair.pair_id} exceeds n_ctx={args.n_ctx}: "
                    f"clean={len(clean_tokens)} corrupt={len(corrupt_tokens)}"
                )
            clean_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=clean_tokens, row=pair.clean_row)
            corrupt_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=corrupt_tokens, row=pair.corrupt_row)

            corrupt_gold, _ = score_sequence_logprob(
                model=model,
                layer=None,
                patch_vector=None,
                patch_position=None,
                prompt_token_ids=corrupt_tokens,
                candidate_text=pair.gold_hypothesis,
            )
            corrupt_foil, _ = score_sequence_logprob(
                model=model,
                layer=None,
                patch_vector=None,
                patch_position=None,
                prompt_token_ids=corrupt_tokens,
                candidate_text=pair.foil_hypothesis,
            )
            clean_gold, _ = score_sequence_logprob(
                model=model,
                layer=None,
                patch_vector=None,
                patch_position=None,
                prompt_token_ids=clean_tokens,
                candidate_text=pair.gold_hypothesis,
            )
            clean_foil, _ = score_sequence_logprob(
                model=model,
                layer=None,
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
                f"pair {pair.pair_id}/{len(pairs)-1} clean={pair.clean_row_index} "
                f"corrupt={pair.corrupt_row_index} margin_corrupt={corrupt_margin:.3f} "
                f"margin_clean={clean_margin:.3f}",
                flush=True,
            )

            clean_vectors = capture_vectors(
                model=model,
                token_ids=clean_tokens,
                layers=layers,
                landmarks=clean_landmarks,
            )
            corrupt_vectors = capture_vectors(
                model=model,
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

                    delta = source_vector - target_vector
                    delta_norm = float(torch.linalg.vector_norm(delta).item())
                    vectors: dict[str, torch.Tensor] = {}
                    if "clean" in patch_modes or "corrupt" in patch_modes or "source" in patch_modes:
                        source_mode = "source"
                        if "clean" in patch_modes:
                            source_mode = "clean"
                        elif "corrupt" in patch_modes:
                            source_mode = "corrupt"
                        vectors[source_mode] = source_vector
                    if "noise" in patch_modes:
                        rng = np.random.default_rng(args.noise_seed + pair.pair_id * 100003 + layer * 1009 + len(landmark))
                        noise = torch.as_tensor(rng.standard_normal(target_vector.shape[0]), dtype=torch.float32)
                        noise = noise / torch.linalg.vector_norm(noise).clamp_min(1e-12) * delta_norm
                        vectors["noise"] = target_vector + noise

                    for patch_mode, patch_vector in vectors.items():
                        patched_gold, gold_hook = score_sequence_logprob(
                            model=model,
                            layer=layer,
                            patch_vector=patch_vector,
                            patch_position=patch_position,
                            prompt_token_ids=prompt_tokens,
                            candidate_text=pair.gold_hypothesis,
                        )
                        patched_foil, foil_hook = score_sequence_logprob(
                            model=model,
                            layer=layer,
                            patch_vector=patch_vector,
                            patch_position=patch_position,
                            prompt_token_ids=prompt_tokens,
                            candidate_text=pair.foil_hypothesis,
                        )
                        patched_margin = patched_gold.logprob - patched_foil.logprob
                        margin_delta = patched_margin - baseline_margin
                        recovery = margin_delta / denominator if abs(denominator) > 1e-9 else 0.0
                        breakage = None
                        if args.patch_direction == "corrupt_to_clean":
                            breakage = -margin_delta / denominator if abs(denominator) > 1e-9 else 0.0
                        output_row = {
                            "schema_version": 1,
                            "patch_direction": args.patch_direction,
                            "pair_id": pair.pair_id,
                            "clean_row_index": pair.clean_row_index,
                            "corrupt_row_index": pair.corrupt_row_index,
                            "key": pair.key,
                            "layer": layer,
                            "hook_name": f"model.model.layers.{layer}.output",
                            "landmark": landmark,
                            "patch_mode": patch_mode,
                            "clean_position": clean_pos,
                            "corrupt_position": corrupt_pos,
                            "patch_position": patch_position,
                            "clean_token_count": len(clean_tokens),
                            "corrupt_token_count": len(corrupt_tokens),
                            "gold_hypothesis": pair.gold_hypothesis,
                            "foil_hypothesis": pair.foil_hypothesis,
                            "corrupt_baseline_margin_gold_minus_foil": corrupt_margin,
                            "clean_reference_margin_gold_minus_foil": clean_margin,
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
                            "patch_delta_l2": (
                                delta_norm
                                if patch_mode in {"clean", "corrupt", "source"}
                                else float(torch.linalg.vector_norm(patch_vector - target_vector).item())
                            ),
                            "gold_hook_applications": gold_hook["applications"],
                            "foil_hook_applications": foil_hook["applications"],
                        }
                        all_patch_rows.append(output_row)
                        fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                        fout.flush()
                        print(
                            f"  L{layer} {landmark} {patch_mode}: "
                            f"delta={margin_delta:.3f} recovery={recovery:.3f}"
                            + (f" breakage={breakage:.3f}" if breakage is not None else ""),
                            flush=True,
                        )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_qwen_patch_hf.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "selection": pair_summary,
        "patch_direction": args.patch_direction,
        "layers": layers,
        "landmarks": landmarks,
        "patch_modes": patch_modes,
        "generation": {
            "enabled": False,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "device_map": args.device_map,
            "chat_template_kwargs": chat_template_kwargs or {},
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
