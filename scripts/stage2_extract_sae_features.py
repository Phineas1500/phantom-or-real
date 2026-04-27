#!/usr/bin/env python3
"""Encode cached Stage 2 residual activations with a Gemma Scope SAE."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import sha256_file
from src.stage2_sae import (
    derive_sae_feature_prefix,
    display_path,
    read_jsonl,
    sae_file_name,
    slice_rows,
    snapshot_revision_from_path,
    summarize_sae_cfg,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-prefix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sae-release", default="gemma-scope-2-27b-it-res-all")
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--hf-repo-id", default="google/gemma-scope-2-27b-it")
    parser.add_argument("--hf-subfolder", default="resid_post_all")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    activation_file = args.activation_prefix.with_suffix(".safetensors")
    source_sidecar_file = args.activation_prefix.with_suffix(".example_ids.jsonl")
    activation_meta_file = args.activation_prefix.with_suffix(".meta.json")
    output_prefix = derive_sae_feature_prefix(
        activation_prefix=args.activation_prefix,
        out_dir=args.out_dir,
        sae_id=args.sae_id,
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
            filename=sae_file_name(args.hf_subfolder, args.sae_id, "config.json"),
            revision=args.hf_revision,
            local_files_only=args.local_files_only,
        )
    )
    params_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=sae_file_name(args.hf_subfolder, args.sae_id, "params.safetensors"),
            revision=args.hf_revision,
            local_files_only=args.local_files_only,
        )
    )
    hf_snapshot_revision = snapshot_revision_from_path(config_path)

    from sae_lens import SAE

    print(f"Loading SAE {args.sae_release} / {args.sae_id}", flush=True)
    sae, sae_cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
        dtype=args.dtype,
    )
    sae.eval()
    sae_cfg = summarize_sae_cfg(sae, sae_cfg_dict)

    print(f"Loading residuals {activation_file}", flush=True)
    residuals = load_file(activation_file)["activations"]
    sidecar_rows = read_jsonl(source_sidecar_file)
    if residuals.shape[0] != len(sidecar_rows):
        raise ValueError(f"{activation_file} rows {residuals.shape[0]} != sidecar rows {len(sidecar_rows)}")
    selected_residuals = residuals[args.skip : None if args.limit is None else args.skip + args.limit]
    selected_rows = slice_rows(sidecar_rows, skip=args.skip, limit=args.limit)
    if selected_residuals.shape[0] != len(selected_rows):
        raise ValueError("residual slice and sidecar slice have different lengths")
    if selected_residuals.shape[1] != sae.cfg.d_in:
        raise ValueError(f"activation dim {selected_residuals.shape[1]} != SAE d_in {sae.cfg.d_in}")

    dtype = getattr(torch, args.dtype)
    top_values = []
    top_indices = []
    l0_counts = []
    started = time.time()
    for start in range(0, selected_residuals.shape[0], args.chunk_size):
        end = min(start + args.chunk_size, selected_residuals.shape[0])
        chunk = selected_residuals[start:end].to(device=args.device, dtype=dtype)
        with torch.inference_mode():
            feature_acts = sae.encode(chunk)
            k = min(args.top_k, feature_acts.shape[-1])
            values, indices = torch.topk(feature_acts, k=k, dim=-1)
            l0 = (feature_acts > 0).sum(dim=-1).to(torch.int32)
        top_values.append(values.cpu())
        top_indices.append(indices.to(torch.int64).cpu())
        l0_counts.append(l0.cpu())
        del chunk, feature_acts, values, indices, l0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"encoded rows {end}/{selected_residuals.shape[0]}", flush=True)

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

    activation_meta = {}
    if activation_meta_file.exists():
        import json

        with activation_meta_file.open() as f:
            activation_meta = json.load(f)

    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_file": display_path(activation_file),
        "activation_sha256": sha256_file(activation_file),
        "source_activation_meta": activation_meta,
        "source_sidecar_file": display_path(source_sidecar_file),
        "source_sidecar_sha256": sha256_file(source_sidecar_file),
        "feature_file": display_path(feature_file),
        "sidecar_file": display_path(sidecar_file),
        "row_count": int(top_values_tensor.shape[0]),
        "input_rows": int(residuals.shape[0]),
        "skip": args.skip,
        "limit": args.limit,
        "chunk_size": args.chunk_size,
        "top_k": int(top_values_tensor.shape[1]),
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "hf_repo_id": args.hf_repo_id,
        "hf_subfolder": args.hf_subfolder,
        "hf_revision_requested": args.hf_revision,
        "hf_snapshot_revision": hf_snapshot_revision,
        "sae_config_file": display_path(config_path),
        "sae_config_sha256": sha256_file(config_path),
        "sae_params_file": display_path(params_path),
        "sae_params_sha256": sha256_file(params_path),
        "sae_cfg": sae_cfg,
        "sparsity_shape": list(sparsity.shape) if sparsity is not None else None,
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
