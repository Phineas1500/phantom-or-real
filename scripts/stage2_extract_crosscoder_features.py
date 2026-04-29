#!/usr/bin/env python3
"""Encode cached residual activations with a Gemma Scope crosscoder."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list, sha256_file  # noqa: E402
from src.stage2_crosscoder import combine_shard_topk, crosscoder_feature_prefix, verify_matching_sidecars  # noqa: E402
from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_sae import display_path, read_jsonl, slice_rows, snapshot_revision_from_path, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crosscoder-id", required=True)
    parser.add_argument("--hf-repo-id", default="google/gemma-scope-2-27b-it")
    parser.add_argument("--hf-subfolder", default="crosscoder")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def crosscoder_file_name(subfolder: str, crosscoder_id: str, file_name: str) -> str:
    return f"{subfolder.strip('/')}/{crosscoder_id}/{file_name}"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_encoder_shard(path: Path, *, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as tensors:
        return {
            "w_enc": tensors.get_tensor("w_enc").to(device=device, dtype=dtype),
            "b_enc": tensors.get_tensor("b_enc").to(device=device, dtype=dtype),
            "threshold": tensors.get_tensor("threshold").to(device=device, dtype=dtype),
        }


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not args.layers:
        raise ValueError("--layers must not be empty")
    if args.skip < 0:
        raise ValueError("--skip must be non-negative")

    dtype = getattr(torch, args.dtype)
    output_prefix = crosscoder_feature_prefix(
        out_dir=args.out_dir,
        model_key=args.model_key,
        task=args.task,
        crosscoder_id=args.crosscoder_id,
        top_k=args.top_k,
        skip=args.skip,
        limit=args.limit,
    )
    feature_file = output_prefix.with_suffix(".safetensors")
    sidecar_file = output_prefix.with_suffix(".example_ids.jsonl")
    meta_file = output_prefix.with_suffix(".meta.json")
    feature_file.parent.mkdir(parents=True, exist_ok=True)

    config_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=crosscoder_file_name(args.hf_subfolder, args.crosscoder_id, "config.json"),
            revision=args.hf_revision,
            local_files_only=args.local_files_only,
        )
    )
    config = load_json(config_path)
    param_paths = [
        Path(
            hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=crosscoder_file_name(args.hf_subfolder, args.crosscoder_id, f"params_layer_{idx}.safetensors"),
                revision=args.hf_revision,
                local_files_only=args.local_files_only,
            )
        )
        for idx in range(len(args.layers))
    ]
    hf_snapshot_revision = snapshot_revision_from_path(config_path)

    activation_tensors = []
    activation_files = []
    activation_metas = []
    sidecar_rows_by_layer = []
    for layer in args.layers:
        prefix = args.activation_dir / activation_stem(model_key=args.model_key, task=args.task, layer=layer)
        activation_file = prefix.with_suffix(".safetensors")
        sidecar_path = prefix.with_suffix(".example_ids.jsonl")
        meta_path = prefix.with_suffix(".meta.json")
        activation_files.append(activation_file)
        activation_tensors.append(load_file(activation_file)["activations"])
        sidecar_rows_by_layer.append(read_jsonl(sidecar_path))
        activation_metas.append(load_json(meta_path))
    reference_rows = sidecar_rows_by_layer[0]
    for rows in sidecar_rows_by_layer[1:]:
        verify_matching_sidecars(reference_rows, rows)
    row_count = len(reference_rows)
    for activation_file, tensor in zip(activation_files, activation_tensors, strict=True):
        if tensor.shape[0] != row_count:
            raise ValueError(f"{activation_file} rows {tensor.shape[0]} != sidecar rows {row_count}")

    selected_rows = slice_rows(reference_rows, skip=args.skip, limit=args.limit)
    end = None if args.limit is None else args.skip + args.limit
    selected_tensors = [tensor[args.skip:end] for tensor in activation_tensors]
    selected_count = len(selected_rows)
    if selected_count == 0:
        raise ValueError("selected row count is zero")

    print(f"Loading crosscoder {args.crosscoder_id} encoder shards", flush=True)
    encoder_shards = [load_encoder_shard(path, device=args.device, dtype=dtype) for path in param_paths]
    shard_widths = [int(shard["b_enc"].shape[0]) for shard in encoder_shards]
    total_width = sum(shard_widths)
    offsets = []
    running = 0
    for width in shard_widths:
        offsets.append(running)
        running += width

    top_values = []
    top_indices = []
    l0_counts = []
    started = time.time()
    for start in range(0, selected_count, args.chunk_size):
        chunk_end = min(start + args.chunk_size, selected_count)
        shard_top_values = []
        shard_top_indices = []
        shard_l0 = []
        for shard_idx, (tensor, shard, offset) in enumerate(
            zip(selected_tensors, encoder_shards, offsets, strict=True)
        ):
            chunk = tensor[start:chunk_end].to(device=args.device, dtype=dtype)
            with torch.inference_mode():
                pre_acts = chunk @ shard["w_enc"] + shard["b_enc"]
                active = pre_acts > shard["threshold"]
                feature_acts = torch.where(active, pre_acts, torch.zeros((), device=args.device, dtype=dtype))
                k = min(args.top_k, feature_acts.shape[-1])
                values, indices = torch.topk(feature_acts, k=k, dim=-1)
                shard_l0.append(active.sum(dim=-1).to(torch.int32))
                shard_top_values.append(values)
                shard_top_indices.append((indices + offset).to(torch.int64))
            del chunk, pre_acts, active, feature_acts, values, indices
            if shard_idx % 2 == 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        values, indices = combine_shard_topk(shard_top_values, shard_top_indices, top_k=args.top_k)
        top_values.append(values.cpu())
        top_indices.append(indices.cpu())
        l0_counts.append(torch.stack(shard_l0, dim=0).sum(dim=0).cpu())
        del shard_top_values, shard_top_indices, shard_l0, values, indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"encoded rows {chunk_end}/{selected_count}", flush=True)

    top_values_tensor = torch.cat(top_values, dim=0)
    top_indices_tensor = torch.cat(top_indices, dim=0)
    l0_tensor = torch.cat(l0_counts, dim=0)
    elapsed = time.time() - started
    save_file(
        {
            "top_values": top_values_tensor,
            "top_indices": top_indices_tensor,
            "l0": l0_tensor,
        },
        feature_file,
    )
    write_jsonl(sidecar_file, selected_rows)

    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_file": display_path(feature_file),
        "sidecar_file": display_path(sidecar_file),
        "row_count": int(top_values_tensor.shape[0]),
        "input_rows": row_count,
        "skip": args.skip,
        "limit": args.limit,
        "chunk_size": args.chunk_size,
        "top_k": int(top_values_tensor.shape[1]),
        "crosscoder_id": args.crosscoder_id,
        "hf_repo_id": args.hf_repo_id,
        "hf_subfolder": args.hf_subfolder,
        "hf_revision_requested": args.hf_revision,
        "hf_snapshot_revision": hf_snapshot_revision,
        "crosscoder_config_file": display_path(config_path),
        "crosscoder_config_sha256": sha256_file(config_path),
        "crosscoder_param_files": [display_path(path) for path in param_paths],
        "crosscoder_param_sha256": {f"params_layer_{idx}": sha256_file(path) for idx, path in enumerate(param_paths)},
        "crosscoder_config": config,
        "crosscoder_cfg": {
            "architecture": config.get("architecture"),
            "type": config.get("type"),
            "d_in": int(selected_tensors[0].shape[1]),
            "d_sae": int(total_width),
            "shard_widths": shard_widths,
            "layers": args.layers,
            "hf_hook_point_in": config.get("hf_hook_point_in"),
            "hf_hook_point_out": config.get("hf_hook_point_out"),
            "affine_connection": config.get("affine_connection"),
            "l0": config.get("l0"),
        },
        "activation_files": [display_path(path) for path in activation_files],
        "activation_sha256": {f"L{layer}": sha256_file(path) for layer, path in zip(args.layers, activation_files)},
        "source_activation_meta": activation_metas,
        "source_file": activation_metas[0].get("jsonl_path"),
        "device": args.device,
        "dtype": args.dtype,
        "elapsed_seconds": elapsed,
        "rows_per_second": top_values_tensor.shape[0] / elapsed if elapsed > 0 else None,
        "l0_mean": float(l0_tensor.float().mean().item()) if l0_tensor.numel() else None,
        "top1_mean": float(top_values_tensor[:, 0].float().mean().item()) if top_values_tensor.numel() else None,
    }
    write_json(meta_file, meta)
    print(meta_file, flush=True)


if __name__ == "__main__":
    main()
