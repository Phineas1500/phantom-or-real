#!/usr/bin/env python3
"""Encode cached Stage 2 residual activations with a Qwen-Scope SAE."""

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

from src.activations import sha256_file  # noqa: E402
from src.qwen_scope import (  # noqa: E402
    encode_qwen_scope_topk,
    infer_qwen_scope_top_k,
    load_qwen_scope_sae,
    qwen_scope_filename,
    qwen_scope_sae_id,
    qwen_scope_sae_summary,
)
from src.stage2_probes import read_json  # noqa: E402
from src.stage2_sae import (  # noqa: E402
    derive_sae_feature_prefix,
    display_path,
    read_jsonl,
    slice_rows,
    snapshot_revision_from_path,
    write_json,
    write_jsonl,
)


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-prefix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sae-repo-id", required=True)
    parser.add_argument("--sae-id", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    activation_file = args.activation_prefix.with_suffix(".safetensors")
    source_sidecar_file = args.activation_prefix.with_suffix(".example_ids.jsonl")
    activation_meta_file = args.activation_prefix.with_suffix(".meta.json")
    activation_meta = read_json(activation_meta_file)
    layer = args.layer if args.layer is not None else int(activation_meta["layer"])
    top_k = args.top_k if args.top_k is not None else infer_qwen_scope_top_k(args.sae_repo_id)
    if top_k is None:
        raise ValueError("--top-k is required when it cannot be inferred from --sae-repo-id")
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    sae_id = args.sae_id or qwen_scope_sae_id(args.sae_repo_id)

    output_prefix = derive_sae_feature_prefix(
        activation_prefix=args.activation_prefix,
        out_dir=args.out_dir,
        sae_id=sae_id,
        top_k=top_k,
        skip=args.skip,
        limit=args.limit,
    )
    feature_file = output_prefix.with_suffix(".safetensors")
    sidecar_file = output_prefix.with_suffix(".example_ids.jsonl")
    meta_file = output_prefix.with_suffix(".meta.json")
    feature_file.parent.mkdir(parents=True, exist_ok=True)

    params_path = Path(
        hf_hub_download(
            repo_id=args.sae_repo_id,
            filename=qwen_scope_filename(layer),
            revision=args.hf_revision,
            local_files_only=args.local_files_only,
        )
    )
    hf_snapshot_revision = snapshot_revision_from_path(params_path)

    dtype = torch_dtype(args.dtype)
    print(f"Loading Qwen-Scope SAE {args.sae_repo_id} {qwen_scope_filename(layer)}", flush=True)
    sae = load_qwen_scope_sae(
        params_path,
        repo_id=args.sae_repo_id,
        layer=layer,
        device=args.device,
        dtype=dtype,
        load_encoder=True,
        load_decoder=False,
        top_k=top_k,
    )
    sae_cfg = qwen_scope_sae_summary(sae)

    print(f"Loading residuals {activation_file}", flush=True)
    residuals = load_file(activation_file)["activations"]
    sidecar_rows = read_jsonl(source_sidecar_file)
    if residuals.shape[0] != len(sidecar_rows):
        raise ValueError(f"{activation_file} rows {residuals.shape[0]} != sidecar rows {len(sidecar_rows)}")
    selected_residuals = residuals[args.skip : None if args.limit is None else args.skip + args.limit]
    selected_rows = slice_rows(sidecar_rows, skip=args.skip, limit=args.limit)
    if selected_residuals.shape[0] != len(selected_rows):
        raise ValueError("residual slice and sidecar slice have different lengths")
    if selected_residuals.shape[1] != sae.d_model:
        raise ValueError(f"activation dim {selected_residuals.shape[1]} != SAE d_model {sae.d_model}")

    top_values = []
    top_indices = []
    l0_counts = []
    started = time.time()
    for start in range(0, selected_residuals.shape[0], args.chunk_size):
        end = min(start + args.chunk_size, selected_residuals.shape[0])
        chunk = selected_residuals[start:end]
        with torch.inference_mode():
            values, indices = encode_qwen_scope_topk(sae, chunk, top_k=top_k)
            l0 = torch.count_nonzero(values, dim=-1).to(torch.int32)
        top_values.append(values.cpu())
        top_indices.append(indices.cpu())
        l0_counts.append(l0.cpu())
        del chunk, values, indices, l0
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
        "sae_release": "qwen-scope",
        "sae_id": sae_id,
        "sae_repo_id": args.sae_repo_id,
        "hf_repo_id": args.sae_repo_id,
        "hf_revision_requested": args.hf_revision,
        "hf_snapshot_revision": hf_snapshot_revision,
        "sae_params_file": display_path(params_path),
        "sae_params_sha256": sha256_file(params_path),
        "sae_cfg": sae_cfg,
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
