#!/usr/bin/env python3
"""Decode SAE top-k features and probe reconstruction/error activations."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import sha256_file  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    read_json,
    read_split_assignments,
    run_raw_activation_probe,
    write_json,
)
from src.stage2_reconstruction import (  # noqa: E402
    ReconstructionStats,
    decode_topk_linear,
    dense_topk_features,
)
from src.stage2_sae import display_path  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def output_prefix(
    *,
    out_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    sae_id: str,
    top_k: int,
    kind: str,
) -> Path:
    return out_dir / f"{model_key}_{task}_L{layer}_{sae_id}_top{top_k}_{kind}"


def verify_sparse_decode(
    *,
    sae: Any,
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    rows: int,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any] | None:
    if rows <= 0:
        return None
    rows = min(rows, int(top_indices.shape[0]))
    indices = top_indices[:rows].to(device=device, dtype=torch.long)
    values = top_values[:rows].to(device=device, dtype=dtype)
    with torch.inference_mode():
        sparse_recon = decode_topk_linear(sae, indices, values, dtype=dtype).float()
        dense_features = dense_topk_features(
            indices,
            values,
            d_sae=int(sae.cfg.d_sae),
            dtype=dtype,
            device=device,
        )
        dense_recon = sae.decode(dense_features).float()
        diff = (sparse_recon - dense_recon).abs()
    return {
        "rows": rows,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def decode_one_dataset(
    *,
    sae: Any,
    activation_prefix: Path,
    feature_prefix: Path,
    recon_prefix: Path,
    error_prefix: Path,
    device: str,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    chunk_size: int,
    verify_dense_rows: int,
) -> dict[str, Any]:
    started = time.time()
    raw_file = activation_prefix.with_suffix(".safetensors")
    raw_sidecar_file = activation_prefix.with_suffix(".example_ids.jsonl")
    raw_meta_file = activation_prefix.with_suffix(".meta.json")
    feature_file = feature_prefix.with_suffix(".safetensors")
    feature_sidecar_file = feature_prefix.with_suffix(".example_ids.jsonl")
    feature_meta_file = feature_prefix.with_suffix(".meta.json")

    raw_activations = load_file(raw_file)["activations"]
    feature_tensors = load_file(feature_file)
    top_values = feature_tensors["top_values"]
    top_indices = feature_tensors["top_indices"]
    if raw_activations.shape[0] != top_values.shape[0]:
        raise ValueError(f"raw rows {raw_activations.shape[0]} != feature rows {top_values.shape[0]}")
    if raw_activations.shape[1] != sae.cfg.d_in:
        raise ValueError(f"raw dim {raw_activations.shape[1]} != SAE d_in {sae.cfg.d_in}")

    sparse_decode_check = verify_sparse_decode(
        sae=sae,
        top_indices=top_indices,
        top_values=top_values,
        rows=verify_dense_rows,
        dtype=dtype,
        device=device,
    )

    stats = ReconstructionStats()
    recon_chunks = []
    error_chunks = []
    for start in range(0, raw_activations.shape[0], chunk_size):
        end = min(start + chunk_size, raw_activations.shape[0])
        raw_chunk = raw_activations[start:end].to(device=device, dtype=dtype)
        indices_chunk = top_indices[start:end].to(device=device, dtype=torch.long)
        values_chunk = top_values[start:end].to(device=device, dtype=dtype)
        with torch.inference_mode():
            reconstruction = decode_topk_linear(sae, indices_chunk, values_chunk, dtype=dtype)
            error = raw_chunk - reconstruction
        stats.update(raw_chunk, reconstruction)
        recon_chunks.append(reconstruction.to(dtype=output_dtype).cpu())
        error_chunks.append(error.to(dtype=output_dtype).cpu())
        del raw_chunk, indices_chunk, values_chunk, reconstruction, error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{feature_prefix.name}: decoded rows {end}/{raw_activations.shape[0]}", flush=True)

    recon_tensor = torch.cat(recon_chunks, dim=0)
    error_tensor = torch.cat(error_chunks, dim=0)
    save_file({"activations": recon_tensor}, recon_prefix.with_suffix(".safetensors"))
    save_file({"activations": error_tensor}, error_prefix.with_suffix(".safetensors"))
    shutil.copyfile(feature_sidecar_file, recon_prefix.with_suffix(".example_ids.jsonl"))
    shutil.copyfile(feature_sidecar_file, error_prefix.with_suffix(".example_ids.jsonl"))

    raw_meta = read_json(raw_meta_file)
    feature_meta = read_json(feature_meta_file)
    common_meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_activation_file": display_path(raw_file),
        "raw_activation_sha256": sha256_file(raw_file),
        "raw_sidecar_file": display_path(raw_sidecar_file),
        "raw_sidecar_sha256": sha256_file(raw_sidecar_file),
        "raw_activation_meta": raw_meta,
        "feature_file": display_path(feature_file),
        "feature_sha256": sha256_file(feature_file),
        "feature_sidecar_file": display_path(feature_sidecar_file),
        "feature_sidecar_sha256": sha256_file(feature_sidecar_file),
        "feature_meta": feature_meta,
        "sae_release": feature_meta["sae_release"],
        "sae_id": feature_meta["sae_id"],
        "hf_snapshot_revision": feature_meta.get("hf_snapshot_revision"),
        "layer": raw_meta.get("layer"),
        "task": raw_meta.get("task"),
        "model_key": raw_meta.get("model_key"),
        "shape": list(recon_tensor.shape),
        "dtype": str(output_dtype),
        "decode_dtype": str(dtype),
        "device": device,
        "chunk_size": chunk_size,
        "top_k": int(top_values.shape[1]),
        "sparse_decode_check": sparse_decode_check,
        "reconstruction_stats": stats.to_dict(),
        "elapsed_seconds": time.time() - started,
    }
    recon_meta = {
        **common_meta,
        "kind": "reconstruction",
        "activation_file": display_path(recon_prefix.with_suffix(".safetensors")),
        "sidecar_file": display_path(recon_prefix.with_suffix(".example_ids.jsonl")),
        "jsonl_path": raw_meta.get("jsonl_path"),
    }
    error_meta = {
        **common_meta,
        "kind": "reconstruction_error",
        "activation_file": display_path(error_prefix.with_suffix(".safetensors")),
        "sidecar_file": display_path(error_prefix.with_suffix(".example_ids.jsonl")),
        "jsonl_path": raw_meta.get("jsonl_path"),
    }
    write_json(recon_prefix.with_suffix(".meta.json"), recon_meta)
    write_json(error_prefix.with_suffix(".meta.json"), error_meta)
    return {
        "reconstruction_prefix": str(recon_prefix),
        "error_prefix": str(error_prefix),
        "reconstruction_stats": stats.to_dict(),
        "sparse_decode_check": sparse_decode_check,
        "elapsed_seconds": time.time() - started,
    }


def run_diagnostics(
    *,
    activation_dir: Path,
    feature_dir: Path,
    out_dir: Path,
    model_key: str,
    tasks: list[str],
    layer: int,
    sae_ids: list[str],
    top_k: int,
    splits_path: Path,
    split_family: str,
    device: str,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    chunk_size: int,
    verify_dense_rows: int,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    from sae_lens import SAE

    out_dir.mkdir(parents=True, exist_ok=True)
    split_assignments = read_split_assignments(splits_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_dir": str(activation_dir),
        "feature_dir": str(feature_dir),
        "out_dir": str(out_dir),
        "model_key": model_key,
        "tasks": tasks,
        "layer": layer,
        "sae_ids": sae_ids,
        "top_k": top_k,
        "splits_path": str(splits_path),
        "split_family": split_family,
        "device": device,
        "dtype": str(dtype),
        "output_dtype": str(output_dtype),
        "chunk_size": chunk_size,
        "verify_dense_rows": verify_dense_rows,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "results": {},
    }

    for sae_id in sae_ids:
        print(f"Loading SAE {sae_id}", flush=True)
        sae, _cfg, _sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release="gemma-scope-2-27b-it-res-all",
            sae_id=sae_id,
            device=device,
            dtype=str(dtype).removeprefix("torch."),
        )
        sae.eval()
        report["results"][sae_id] = {}
        for task in tasks:
            activation_prefix = activation_dir / f"{model_key}_{task}_L{layer}"
            feature_prefix = feature_dir / f"{model_key}_{task}_L{layer}_{sae_id}_top{top_k}"
            recon_prefix = output_prefix(
                out_dir=out_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                sae_id=sae_id,
                top_k=top_k,
                kind="reconstruction",
            )
            error_prefix = output_prefix(
                out_dir=out_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                sae_id=sae_id,
                top_k=top_k,
                kind="error",
            )
            decode_result = decode_one_dataset(
                sae=sae,
                activation_prefix=activation_prefix,
                feature_prefix=feature_prefix,
                recon_prefix=recon_prefix,
                error_prefix=error_prefix,
                device=device,
                dtype=dtype,
                output_dtype=output_dtype,
                chunk_size=chunk_size,
                verify_dense_rows=verify_dense_rows,
            )
            raw_meta = read_json(activation_prefix.with_suffix(".meta.json"))
            probe_results = {}
            for kind, prefix in (("reconstruction", recon_prefix), ("error", error_prefix)):
                probe_results[kind] = run_raw_activation_probe(
                    activation_path=prefix.with_suffix(".safetensors"),
                    sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
                    seed=seed + layer,
                    drop_parse_failed=True,
                    split_assignments=split_assignments,
                    source_file=raw_meta.get("jsonl_path"),
                    split_family=split_family,
                    c_values=c_values,
                    max_iter=max_iter,
                    solver=solver,
                    bootstrap_samples=bootstrap_samples,
                )
            report["results"][sae_id][task] = {
                **decode_result,
                "probes": probe_results,
            }
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


def run_existing_component_probes(
    *,
    activation_dir: Path,
    feature_dir: Path,
    out_dir: Path,
    model_key: str,
    tasks: list[str],
    layer: int,
    sae_ids: list[str],
    top_k: int,
    splits_path: Path,
    split_family: str,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "probe_existing_components",
        "activation_dir": str(activation_dir),
        "feature_dir": str(feature_dir),
        "out_dir": str(out_dir),
        "model_key": model_key,
        "tasks": tasks,
        "layer": layer,
        "sae_ids": sae_ids,
        "top_k": top_k,
        "splits_path": str(splits_path),
        "split_family": split_family,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "results": {},
    }
    for sae_id in sae_ids:
        report["results"][sae_id] = {}
        for task in tasks:
            recon_prefix = output_prefix(
                out_dir=out_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                sae_id=sae_id,
                top_k=top_k,
                kind="reconstruction",
            )
            error_prefix = output_prefix(
                out_dir=out_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                sae_id=sae_id,
                top_k=top_k,
                kind="error",
            )
            recon_meta = read_json(recon_prefix.with_suffix(".meta.json"))
            probe_results = {}
            for kind, prefix in (("reconstruction", recon_prefix), ("error", error_prefix)):
                probe_results[kind] = run_raw_activation_probe(
                    activation_path=prefix.with_suffix(".safetensors"),
                    sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
                    seed=seed + layer,
                    drop_parse_failed=True,
                    split_assignments=split_assignments,
                    source_file=recon_meta.get("jsonl_path"),
                    split_family=split_family,
                    c_values=c_values,
                    max_iter=max_iter,
                    solver=solver,
                    bootstrap_samples=bootstrap_samples,
                )
            report["results"][sae_id][task] = {
                "reconstruction_prefix": str(recon_prefix),
                "error_prefix": str(error_prefix),
                "reconstruction_stats": recon_meta.get("reconstruction_stats"),
                "sparse_decode_check": recon_meta.get("sparse_decode_check"),
                "probes": probe_results,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-ids", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Probe existing reconstruction/error files without re-decoding SAE features.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dtype", default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--verify-dense-rows", type=int, default=8)
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args()

    if args.reuse_existing:
        report = run_existing_component_probes(
            activation_dir=args.activation_dir,
            feature_dir=args.feature_dir,
            out_dir=args.out_dir,
            model_key=args.model_key,
            tasks=args.tasks,
            layer=args.layer,
            sae_ids=args.sae_ids,
            top_k=args.top_k,
            splits_path=args.splits,
            split_family=args.split_family,
            c_values=args.c_values,
            max_iter=args.max_iter,
            solver=args.solver,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    else:
        report = run_diagnostics(
            activation_dir=args.activation_dir,
            feature_dir=args.feature_dir,
            out_dir=args.out_dir,
            model_key=args.model_key,
            tasks=args.tasks,
            layer=args.layer,
            sae_ids=args.sae_ids,
            top_k=args.top_k,
            splits_path=args.splits,
            split_family=args.split_family,
            device=args.device,
            dtype=torch_dtype(args.dtype),
            output_dtype=torch_dtype(args.output_dtype),
            chunk_size=args.chunk_size,
            verify_dense_rows=args.verify_dense_rows,
            c_values=args.c_values,
            max_iter=args.max_iter,
            solver=args.solver,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    write_json(args.output, report)
    print(args.output)
    for sae_id, by_task in report["results"].items():
        for task, result in by_task.items():
            recon_auc = result["probes"]["reconstruction"].get("test_auc")
            error_auc = result["probes"]["error"].get("test_auc")
            energy = result["reconstruction_stats"].get("energy_explained")
            print(f"{sae_id} {task}: recon_auc={recon_auc} error_auc={error_auc} energy={energy}")


if __name__ == "__main__":
    main()
