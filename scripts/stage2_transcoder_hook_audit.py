#!/usr/bin/env python3
"""Audit exact Gemma Scope transcoder input/output hook alignment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import (  # noqa: E402
    encode_stage1_rows,
    input_device_for_model,
    load_tl_model,
    make_padded_batch,
    module_device_summary,
    read_stage1_rows,
    sha256_file,
    tokenizer_pad_token_id,
)
from src.env_loader import load_env  # noqa: E402
from src.stage2_reconstruction import ReconstructionStats  # noqa: E402
from src.stage2_sae import summarize_sae_cfg, write_json  # noqa: E402


HOOKS = (
    "blocks.{layer}.hook_mlp_in",
    "blocks.{layer}.ln2.hook_normalized",
    "blocks.{layer}.ln2.hook_scale",
    "blocks.{layer}.hook_mlp_out",
    "blocks.{layer}.ln2_post.hook_normalized",
    "blocks.{layer}.ln2_post.hook_scale",
)


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def read_task_examples(
    *,
    jsonl_path: Path,
    tokenizer: Any,
    model_name: str,
    rows_per_task: int,
    drop_parse_failed: bool,
) -> list[Any]:
    rows = read_stage1_rows(
        jsonl_path,
        limit=rows_per_task,
        drop_parse_failed=drop_parse_failed,
    )
    return encode_stage1_rows(rows, tokenizer=tokenizer, model_name=model_name)


def capture_hooks(
    *,
    model: Any,
    examples: list[Any],
    layer: int,
    batch_size: int,
    output_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    hook_names = [hook.format(layer=layer) for hook in HOOKS]
    missing = [hook for hook in hook_names if hook not in model.hook_dict]
    if missing:
        sample = sorted(model.hook_dict.keys())[:40]
        raise ValueError(f"missing hooks {missing}; first available hooks: {sample}")

    pad_token_id = tokenizer_pad_token_id(model.tokenizer)
    input_device = input_device_for_model(model)
    captured: dict[str, list[torch.Tensor]] = {hook: [] for hook in hook_names}

    def make_hook(hook_name: str, last_positions: list[int]):
        def _capture(act: torch.Tensor, hook) -> None:  # noqa: ARG001
            positions = torch.tensor(last_positions, device=act.device)
            batch_indices = torch.arange(len(last_positions), device=act.device)
            value = act[batch_indices, positions, :].detach().to("cpu", dtype=output_dtype)
            captured[hook_name].append(value.contiguous())

        return _capture

    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            tokens, attention_mask, last_positions = make_padded_batch(
                chunk,
                pad_token_id=pad_token_id,
                device=input_device,
            )
            fwd_hooks = [(hook_name, make_hook(hook_name, last_positions)) for hook_name in hook_names]
            model.run_with_hooks(
                tokens,
                return_type=None,
                attention_mask=attention_mask,
                fwd_hooks=fwd_hooks,
            )
            del tokens, attention_mask

    return {hook: torch.cat(chunks, dim=0).contiguous() for hook, chunks in captured.items()}


def compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int | list[int]]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch {tuple(left.shape)} != {tuple(right.shape)}")
    diff = left.float() - right.float()
    left_f = left.float()
    right_f = right.float()
    dot = (left_f * right_f).sum(dim=-1)
    left_norm = left_f.square().sum(dim=-1).sqrt()
    right_norm = right_f.square().sum(dim=-1).sqrt()
    cosine = dot / (left_norm * right_norm).clamp_min(1e-12)
    return {
        "shape": list(left.shape),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(diff.square().mean().sqrt().item()),
        "left_mean_l2": float(left_norm.mean().item()),
        "right_mean_l2": float(right_norm.mean().item()),
        "mean_row_cosine": float(cosine.mean().item()),
        "global_cosine": float(
            (left_f * right_f).sum().item()
            / math.sqrt(left_f.square().sum().item() * right_f.square().sum().item())
        ),
    }


def reconstruction_stats(target: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, Any]:
    stats = ReconstructionStats()
    stats.update(target.float(), reconstruction.float())
    return stats.to_dict()


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    tensor_f = tensor.float()
    norms = tensor_f.square().sum(dim=-1).sqrt()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(tensor_f.mean().item()),
        "std": float(tensor_f.std(unbiased=False).item()),
        "min": float(tensor_f.min().item()),
        "max": float(tensor_f.max().item()),
        "mean_l2": float(norms.mean().item()),
        "max_l2": float(norms.max().item()),
    }


def run_transcoder_candidates(
    *,
    sae: Any,
    input_candidates: dict[str, torch.Tensor],
    target_candidates: dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for input_name, input_tensor in input_candidates.items():
        x = input_tensor.to(device=device, dtype=dtype)
        with torch.inference_mode():
            features = sae.encode(x)
            latent = sae.decode(features)
            skip_transposed = x @ sae.W_skip.T.to(device=device, dtype=dtype)
            skip_untransposed = x @ sae.W_skip.to(device=device, dtype=dtype)
            full_transposed = latent + skip_transposed
            full_untransposed = latent + skip_untransposed

        feature_counts = (features.float() > 0).sum(dim=-1)
        input_result: dict[str, Any] = {
            "feature_l0_mean": float(feature_counts.float().mean().item()),
            "feature_l0_min": int(feature_counts.min().item()),
            "feature_l0_max": int(feature_counts.max().item()),
            "feature_abs_max": float(features.float().abs().max().item()),
            "component_norms": {
                "latent": summarize_tensor(latent.detach().cpu()),
                "skip_transposed": summarize_tensor(skip_transposed.detach().cpu()),
                "skip_untransposed": summarize_tensor(skip_untransposed.detach().cpu()),
                "full_transposed": summarize_tensor(full_transposed.detach().cpu()),
                "full_untransposed": summarize_tensor(full_untransposed.detach().cpu()),
            },
            "targets": {},
        }
        for target_name, target_tensor in target_candidates.items():
            target = target_tensor.to(device=device, dtype=dtype)
            input_result["targets"][target_name] = {
                "latent": reconstruction_stats(target, latent),
                "skip_transposed": reconstruction_stats(target, skip_transposed),
                "skip_untransposed": reconstruction_stats(target, skip_untransposed),
                "full_transposed": reconstruction_stats(target, full_transposed),
                "full_untransposed": reconstruction_stats(target, full_untransposed),
            }
            del target
        results[input_name] = input_result
        del x, features, latent, skip_transposed, skip_untransposed, full_transposed, full_untransposed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def rank_pairs(results: dict[str, Any]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for input_name, input_result in results.items():
        for target_name, target_result in input_result["targets"].items():
            for component_name, stats in target_result.items():
                ranked.append(
                    {
                        "input": input_name,
                        "target": target_name,
                        "component": component_name,
                        "energy_explained": stats.get("energy_explained"),
                        "relative_error_l2": stats.get("relative_error_l2"),
                        "global_cosine": stats.get("global_cosine"),
                        "mean_row_cosine": stats.get("mean_row_cosine"),
                        "mean_target_l2": stats.get("mean_raw_l2"),
                        "mean_reconstruction_l2": stats.get("mean_reconstruction_l2"),
                    }
                )
    ranked.sort(
        key=lambda item: (
            item["energy_explained"] if item["energy_explained"] is not None else -float("inf"),
            item["global_cosine"] if item["global_cosine"] is not None else -float("inf"),
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--property-jsonl", type=Path, required=True)
    parser.add_argument("--subtype-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--rows-per-task", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--drop-parse-failed", action="store_true")
    parser.add_argument("--transcoder-release", required=True)
    parser.add_argument("--transcoder-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    dtype = torch_dtype(args.dtype)
    output_dtype = torch_dtype(args.output_dtype)
    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode="no-processing",
    )
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    else:
        model.cfg.use_hook_mlp_in = True

    examples = []
    for jsonl_path in (args.property_jsonl, args.subtype_jsonl):
        examples.extend(
            read_task_examples(
                jsonl_path=jsonl_path,
                tokenizer=model.tokenizer,
                model_name=args.model,
                rows_per_task=args.rows_per_task,
                drop_parse_failed=args.drop_parse_failed,
            )
        )

    captured = capture_hooks(
        model=model,
        examples=examples,
        layer=args.layer,
        batch_size=args.batch_size,
        output_dtype=output_dtype,
    )
    layer_module = model.blocks[args.layer]
    ln2_w = layer_module.ln2.w.detach().to("cpu", dtype=output_dtype)
    ln2_post_w = layer_module.ln2_post.w.detach().to("cpu", dtype=output_dtype)
    ln2_norm = captured[f"blocks.{args.layer}.ln2.hook_normalized"]
    ln2_post_norm = captured[f"blocks.{args.layer}.ln2_post.hook_normalized"]
    hook_mlp_in = captured[f"blocks.{args.layer}.hook_mlp_in"]
    hook_mlp_out = captured[f"blocks.{args.layer}.hook_mlp_out"]
    ln2_weighted = ln2_norm * ln2_w
    ln2_post_weighted = ln2_post_norm * ln2_post_w

    input_candidates = {
        "hook_mlp_in": hook_mlp_in,
        "ln2_normalized": ln2_norm,
        "ln2_weighted": ln2_weighted,
    }
    target_candidates = {
        "hook_mlp_out": hook_mlp_out,
        "ln2_post_normalized": ln2_post_norm,
        "ln2_post_weighted": ln2_post_weighted,
    }

    from sae_lens import SAE

    sae, sae_cfg, _sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=args.transcoder_release,
        sae_id=args.transcoder_id,
        device=args.device,
        dtype=args.dtype,
    )
    sae.eval()

    transcoder_results = run_transcoder_candidates(
        sae=sae,
        input_candidates=input_candidates,
        target_candidates=target_candidates,
        device=args.device,
        dtype=dtype,
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "model": args.model,
        "model_key": args.model_key,
        "layer": args.layer,
        "rows_per_task": args.rows_per_task,
        "row_count": len(examples),
        "batch_size": args.batch_size,
        "n_devices": args.n_devices,
        "dtype": str(dtype),
        "output_dtype": str(output_dtype),
        "drop_parse_failed": args.drop_parse_failed,
        "property_jsonl": str(args.property_jsonl),
        "property_jsonl_sha256": sha256_file(args.property_jsonl),
        "subtype_jsonl": str(args.subtype_jsonl),
        "subtype_jsonl_sha256": sha256_file(args.subtype_jsonl),
        "module_devices": module_device_summary(model),
        "transcoder_release": args.transcoder_release,
        "transcoder_id": args.transcoder_id,
        "transcoder_cfg": summarize_sae_cfg(sae, sae_cfg),
        "captured_hooks": {name: summarize_tensor(tensor) for name, tensor in captured.items()},
        "derived_tensors": {
            "ln2_weighted": summarize_tensor(ln2_weighted),
            "ln2_post_weighted": summarize_tensor(ln2_post_weighted),
        },
        "hook_equivalence_checks": {
            "hook_mlp_out_vs_ln2_post_weighted": compare_tensors(hook_mlp_out, ln2_post_weighted),
            "hook_mlp_out_vs_ln2_post_normalized": compare_tensors(hook_mlp_out, ln2_post_norm),
            "ln2_weighted_vs_ln2_normalized": compare_tensors(ln2_weighted, ln2_norm),
        },
        "transcoder_candidates": transcoder_results,
        "ranked_pairs": rank_pairs(transcoder_results),
    }
    write_json(args.output, report)
    print(args.output)
    print("Top candidate pairs:")
    for row in report["ranked_pairs"][:12]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
