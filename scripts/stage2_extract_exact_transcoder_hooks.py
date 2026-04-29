#!/usr/bin/env python3
"""Extract exact weighted Gemma Scope transcoder input and output hooks."""

from __future__ import annotations

import argparse
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
    write_activation_outputs,
)
from src.env_loader import load_env  # noqa: E402


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def capture_exact_pair(
    *,
    model: Any,
    examples: list[Any],
    layer: int,
    batch_size: int,
    output_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    input_hook = f"blocks.{layer}.ln2.hook_normalized"
    output_hook = f"blocks.{layer}.hook_mlp_out"
    for hook_name in (input_hook, output_hook):
        if hook_name not in model.hook_dict:
            sample = sorted(model.hook_dict.keys())[:40]
            raise ValueError(f"missing hook {hook_name}; first available hooks: {sample}")

    pad_token_id = tokenizer_pad_token_id(model.tokenizer)
    input_device = input_device_for_model(model)
    ln2_w = model.blocks[layer].ln2.w.detach()
    input_chunks: list[torch.Tensor] = []
    output_chunks: list[torch.Tensor] = []
    token_counts = [example.token_count for example in examples]
    rows_done = 0
    started = time.monotonic()

    def make_hook(name: str, last_positions: list[int], captured: dict[str, torch.Tensor]):
        def _capture(act: torch.Tensor, hook) -> None:  # noqa: ARG001
            positions = torch.tensor(last_positions, device=act.device)
            batch_indices = torch.arange(len(last_positions), device=act.device)
            captured[name] = act[batch_indices, positions, :].detach()

        return _capture

    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            tokens, attention_mask, last_positions = make_padded_batch(
                chunk,
                pad_token_id=pad_token_id,
                device=input_device,
            )
            captured: dict[str, torch.Tensor] = {}
            model.run_with_hooks(
                tokens,
                return_type=None,
                attention_mask=attention_mask,
                fwd_hooks=[
                    (input_hook, make_hook("ln2_normalized", last_positions, captured)),
                    (output_hook, make_hook("hook_mlp_out", last_positions, captured)),
                ],
            )
            ln2_normalized = captured["ln2_normalized"]
            weighted_input = ln2_normalized * ln2_w.to(device=ln2_normalized.device, dtype=ln2_normalized.dtype)
            input_chunks.append(weighted_input.to("cpu", dtype=output_dtype).contiguous())
            output_chunks.append(captured["hook_mlp_out"].to("cpu", dtype=output_dtype).contiguous())
            rows_done += len(chunk)
            print(f"exact hooks rows {rows_done}/{len(examples)}", flush=True)
            del tokens, attention_mask, captured, ln2_normalized, weighted_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    stats = {
        "elapsed_seconds": time.monotonic() - started,
        "rows_done": rows_done,
        "rows_per_second": rows_done / (time.monotonic() - started),
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_count_mean": sum(token_counts) / len(token_counts),
    }
    return {
        "mlp_in_weighted": torch.cat(input_chunks, dim=0).contiguous(),
        "mlp_out_hook": torch.cat(output_chunks, dim=0).contiguous(),
    }, stats


def extract_one_task(
    *,
    model: Any,
    jsonl_path: Path,
    model_name: str,
    model_key: str,
    task: str,
    layer: int,
    batch_size: int,
    out_dir: Path,
    output_dtype: torch.dtype,
) -> None:
    rows = read_stage1_rows(jsonl_path)
    examples = encode_stage1_rows(rows, tokenizer=model.tokenizer, model_name=model_name)
    activations, stats = capture_exact_pair(
        model=model,
        examples=examples,
        layer=layer,
        batch_size=batch_size,
        output_dtype=output_dtype,
    )
    sidecar_rows = [
        {
            "row_index": example.row_index,
            "example_id": example.example_id,
            "height": example.height,
            "task": example.task,
            "model": example.model,
            "is_correct_strong": example.is_correct_strong,
            "parse_failed": example.parse_failed,
            "token_count": example.token_count,
            "last_token_position": example.token_count - 1,
        }
        for example in examples
    ]
    common_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": sha256_file(jsonl_path),
        "model_name": model_name,
        "model_key": model_key,
        "task": task,
        "layers": [layer],
        "batch_size": batch_size,
        "n_devices": getattr(model.cfg, "n_devices", None),
        "n_ctx": getattr(model.cfg, "n_ctx", None),
        "load_mode": "no-processing",
        "dtype": str(model.cfg.dtype),
        "output_dtype": str(output_dtype),
        "row_count": len(examples),
        "drop_parse_failed": False,
        "height": None,
        "skip": 0,
        "limit": None,
        "transformerlens_n_layers": model.cfg.n_layers,
        "transformerlens_d_model": model.cfg.d_model,
        "transformerlens_n_devices": getattr(model.cfg, "n_devices", None),
        "module_devices": module_device_summary(model),
        "extraction_stats": stats,
    }
    write_activation_outputs(
        {layer: activations["mlp_in_weighted"]},
        sidecar_rows,
        out_dir=out_dir,
        model_key=model_key,
        task=task,
        metadata={
            **common_metadata,
            "source_hook_name": f"blocks.{layer}.ln2.hook_normalized",
            "source_weight_name": f"blocks.{layer}.ln2.w",
            "exact_transcoder_role": "input",
        },
        activation_site="mlp_in_weighted",
        hook_template="blocks.{layer}.ln2.hook_normalized * blocks.{layer}.ln2.w",
    )
    write_activation_outputs(
        {layer: activations["mlp_out_hook"]},
        sidecar_rows,
        out_dir=out_dir,
        model_key=model_key,
        task=task,
        metadata={
            **common_metadata,
            "source_hook_name": f"blocks.{layer}.hook_mlp_out",
            "exact_transcoder_role": "output",
        },
        activation_site="mlp_out_hook",
        hook_template="blocks.{layer}.hook_mlp_out",
    )


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--property-jsonl", type=Path, required=True)
    parser.add_argument("--subtype-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dtype", default="bfloat16")
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage2/activations"))
    args = parser.parse_args()

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=torch_dtype(args.dtype),
        load_mode="no-processing",
    )
    model.eval()
    extract_one_task(
        model=model,
        jsonl_path=args.property_jsonl,
        model_name=args.model,
        model_key=args.model_key,
        task="infer_property",
        layer=args.layer,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        output_dtype=torch_dtype(args.output_dtype),
    )
    extract_one_task(
        model=model,
        jsonl_path=args.subtype_jsonl,
        model_name=args.model,
        model_key=args.model_key,
        task="infer_subtype",
        layer=args.layer,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        output_dtype=torch_dtype(args.output_dtype),
    )


if __name__ == "__main__":
    main()
