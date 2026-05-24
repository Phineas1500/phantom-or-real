#!/usr/bin/env python3
"""Decode local top-k dictionaries and probe target/reconstruction/error activations."""

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
    read_jsonl,
    read_split_assignments,
    run_raw_activation_probe,
    write_json,
)
from src.stage2_reconstruction import ReconstructionStats  # noqa: E402
from src.stage2_sae import display_path  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def strip_artifacts(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_artifacts(item)
            for key, item in value.items()
            if not str(key).startswith("_artifact_")
        }
    if isinstance(value, list):
        return [strip_artifacts(item) for item in value]
    return value


def sidecar_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("row_index"), row.get("example_id")


def selected_rows_tensor(
    *,
    activation_file: Path,
    activation_sidecar_file: Path,
    reference_rows: list[dict[str, Any]],
) -> torch.Tensor:
    activations = load_file(activation_file)["activations"]
    activation_sidecar = read_jsonl(activation_sidecar_file)
    index_by_key = {sidecar_key(row): idx for idx, row in enumerate(activation_sidecar)}
    selected_indices = []
    missing = []
    for row in reference_rows:
        idx = index_by_key.get(sidecar_key(row))
        if idx is None:
            missing.append(sidecar_key(row))
        else:
            selected_indices.append(idx)
    if missing:
        raise KeyError(f"{len(missing)} rows missing from {activation_sidecar_file}; sample={missing[:5]}")
    return activations[selected_indices].float().cpu()


def load_target_matrix(feature_meta: dict[str, Any], reference_rows: list[dict[str, Any]]) -> tuple[torch.Tensor, list[Path], list[dict[str, Any]]]:
    if "target_activation_file" in feature_meta:
        target_files = [Path(feature_meta["target_activation_file"])]
    elif "source_activation_files" in feature_meta:
        target_files = [Path(path) for path in feature_meta["source_activation_files"]]
    else:
        target_files = [Path(feature_meta["activation_file"])]

    parts = []
    metas = []
    for target_file in target_files:
        target_meta_file = target_file.with_suffix(".meta.json")
        target_sidecar_file = target_file.with_suffix(".example_ids.jsonl")
        metas.append(read_json(target_meta_file))
        parts.append(
            selected_rows_tensor(
                activation_file=target_file,
                activation_sidecar_file=target_sidecar_file,
                reference_rows=reference_rows,
            )
        )
    return torch.cat(parts, dim=1), target_files, metas


def decode_topk_local(
    *,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if top_indices.shape != top_values.shape:
        raise ValueError(f"top index/value shapes differ: {tuple(top_indices.shape)} vs {tuple(top_values.shape)}")
    decoder_t = decoder_weight.t().to(device=device, dtype=torch.float32)
    bias = decoder_bias.to(device=device, dtype=torch.float32)
    indices = top_indices.to(device=device, dtype=torch.long)
    values = top_values.to(device=device, dtype=torch.float32)
    out = bias.unsqueeze(0).expand(indices.shape[0], -1).clone()
    for rank in range(indices.shape[1]):
        rank_values = values[:, rank]
        if torch.count_nonzero(rank_values).item() == 0:
            continue
        rows = decoder_t.index_select(0, indices[:, rank])
        out.add_(rows * rank_values.unsqueeze(1))
    return out.to(dtype=dtype)


def component_prefix(*, out_dir: Path, component_id: str, task: str, component: str) -> Path:
    return out_dir / f"{component_id}_{task}_{component}"


def write_component(
    *,
    prefix: Path,
    tensor: torch.Tensor,
    reference_sidecar_file: Path,
    meta: dict[str, Any],
) -> None:
    save_file({"activations": tensor}, prefix.with_suffix(".safetensors"))
    shutil.copyfile(reference_sidecar_file, prefix.with_suffix(".example_ids.jsonl"))
    write_json(prefix.with_suffix(".meta.json"), meta)


def decode_one_feature_set(
    *,
    dictionary_tensors: dict[str, torch.Tensor],
    dictionary_meta: dict[str, Any],
    feature_prefix: Path,
    out_dir: Path,
    component_id: str,
    device: str,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    chunk_size: int,
) -> dict[str, Any]:
    started = time.time()
    feature_file = feature_prefix.with_suffix(".safetensors")
    feature_sidecar_file = feature_prefix.with_suffix(".example_ids.jsonl")
    feature_meta_file = feature_prefix.with_suffix(".meta.json")
    feature_meta = read_json(feature_meta_file)
    reference_rows = read_jsonl(feature_sidecar_file)
    target, target_files, target_metas = load_target_matrix(feature_meta, reference_rows)
    feature_tensors = load_file(feature_file)
    top_values = feature_tensors["top_values"]
    top_indices = feature_tensors["top_indices"]
    if top_values.shape[0] != target.shape[0]:
        raise ValueError(f"feature rows {top_values.shape[0]} != target rows {target.shape[0]}")

    architecture = dictionary_meta["architecture"]
    if architecture == "local_topk_transcoder":
        mean = dictionary_tensors["target_mean"].float()
        std = dictionary_tensors["target_std"].float()
    else:
        mean = dictionary_tensors["input_mean"].float()
        std = dictionary_tensors["input_std"].float()
    if int(mean.numel()) != int(target.shape[1]):
        raise ValueError(f"mean dim {mean.numel()} != target dim {target.shape[1]} for {feature_prefix}")

    decoder_weight = dictionary_tensors["decoder_weight"].float()
    decoder_bias = dictionary_tensors["decoder_bias"].float()
    if decoder_weight.shape[0] != target.shape[1]:
        raise ValueError(f"decoder output dim {decoder_weight.shape[0]} != target dim {target.shape[1]}")

    mean_device = mean.to(device)
    std_device = std.to(device)
    stats = ReconstructionStats()
    recon_chunks = []
    error_chunks = []
    for start in range(0, target.shape[0], chunk_size):
        end = min(start + chunk_size, target.shape[0])
        with torch.inference_mode():
            recon_standardized = decode_topk_local(
                decoder_weight=decoder_weight,
                decoder_bias=decoder_bias,
                top_indices=top_indices[start:end],
                top_values=top_values[start:end],
                device=device,
                dtype=dtype,
            ).float()
            recon = recon_standardized * std_device.unsqueeze(0) + mean_device.unsqueeze(0)
            raw = target[start:end].to(device=device, dtype=torch.float32)
            error = raw - recon
        stats.update(raw, recon)
        recon_chunks.append(recon.to(dtype=output_dtype).cpu())
        error_chunks.append(error.to(dtype=output_dtype).cpu())
        del raw, recon_standardized, recon, error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{feature_prefix.name}: decoded rows {end}/{target.shape[0]}", flush=True)

    target_tensor = target.to(dtype=output_dtype)
    recon_tensor = torch.cat(recon_chunks, dim=0)
    error_tensor = torch.cat(error_chunks, dim=0)
    task = str(target_metas[0].get("task") or feature_meta.get("source_activation_meta", {}).get("task"))
    source_file = str(target_metas[0].get("jsonl_path") or feature_meta.get("source_file"))
    common_meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_id": component_id,
        "task": task,
        "source_file": source_file,
        "target_activation_files": [display_path(path) for path in target_files],
        "target_activation_sha256": [sha256_file(path) for path in target_files],
        "target_activation_metas": target_metas,
        "feature_file": display_path(feature_file),
        "feature_sha256": sha256_file(feature_file),
        "feature_meta": feature_meta,
        "feature_sidecar_file": display_path(feature_sidecar_file),
        "feature_sidecar_sha256": sha256_file(feature_sidecar_file),
        "dictionary_meta": dictionary_meta,
        "shape": list(target_tensor.shape),
        "decode_dtype": str(dtype),
        "dtype": str(output_dtype),
        "device": device,
        "chunk_size": chunk_size,
        "top_k": int(top_values.shape[1]),
        "reconstruction_stats": stats.to_dict(),
        "elapsed_seconds": time.time() - started,
        "jsonl_path": source_file,
    }
    prefixes = {
        "target": component_prefix(out_dir=out_dir, component_id=component_id, task=task, component="target"),
        "reconstruction": component_prefix(out_dir=out_dir, component_id=component_id, task=task, component="reconstruction"),
        "error": component_prefix(out_dir=out_dir, component_id=component_id, task=task, component="error"),
    }
    for component, tensor in (
        ("target", target_tensor),
        ("reconstruction", recon_tensor),
        ("error", error_tensor),
    ):
        write_component(
            prefix=prefixes[component],
            tensor=tensor,
            reference_sidecar_file=feature_sidecar_file,
            meta={
                **common_meta,
                "kind": f"local_dictionary_{component}",
                "component": component,
                "activation_file": display_path(prefixes[component].with_suffix(".safetensors")),
                "sidecar_file": display_path(prefixes[component].with_suffix(".example_ids.jsonl")),
            },
        )

    return {
        "task": task,
        "feature_prefix": str(feature_prefix),
        "component_prefixes": {key: str(value) for key, value in prefixes.items()},
        "target_activation_files": [str(path) for path in target_files],
        "reconstruction_stats": stats.to_dict(),
        "elapsed_seconds": time.time() - started,
    }


def probe_components(
    *,
    component_prefixes: dict[str, str],
    source_file: str,
    splits_path: Path,
    split_family: str,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
    seed: int,
    skip_target_probe: bool,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    probes = {}
    components = ("reconstruction", "error") if skip_target_probe else ("target", "reconstruction", "error")
    for offset, component in enumerate(components):
        prefix = Path(component_prefixes[component])
        print(f"probing {prefix.name} split={split_family}", flush=True)
        probes[component] = run_raw_activation_probe(
            activation_path=prefix.with_suffix(".safetensors"),
            sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
            seed=seed + offset * 1009,
            drop_parse_failed=True,
            split_assignments=split_assignments,
            source_file=source_file,
            split_family=split_family,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
            bootstrap_samples=bootstrap_samples,
        )
    return strip_artifacts(probes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--dictionary-meta", type=Path, required=True)
    parser.add_argument("--feature-prefix", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--component-id", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dtype", default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--skip-target-probe", action="store_true")
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dictionary_tensors = load_file(args.dictionary)
    dictionary_meta = read_json(args.dictionary_meta)
    dtype = torch_dtype(args.dtype)
    output_dtype = torch_dtype(args.output_dtype)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dictionary": str(args.dictionary),
        "dictionary_meta": str(args.dictionary_meta),
        "dictionary_sha256": sha256_file(args.dictionary),
        "component_id": args.component_id,
        "feature_prefixes": [str(prefix) for prefix in args.feature_prefix],
        "out_dir": str(args.out_dir),
        "splits_path": str(args.splits),
        "split_family": args.split_family,
        "device": args.device,
        "dtype": str(dtype),
        "output_dtype": str(output_dtype),
        "chunk_size": args.chunk_size,
        "c_values": list(args.c_values),
        "max_iter": args.max_iter,
        "solver": args.solver,
        "bootstrap_samples": args.bootstrap_samples,
        "skip_target_probe": args.skip_target_probe,
        "seed": args.seed,
        "results": {},
    }
    for idx, feature_prefix in enumerate(args.feature_prefix):
        decoded = decode_one_feature_set(
            dictionary_tensors=dictionary_tensors,
            dictionary_meta=dictionary_meta,
            feature_prefix=feature_prefix,
            out_dir=args.out_dir,
            component_id=args.component_id,
            device=args.device,
            dtype=dtype,
            output_dtype=output_dtype,
            chunk_size=args.chunk_size,
        )
        first_component_meta = read_json(Path(decoded["component_prefixes"]["target"]).with_suffix(".meta.json"))
        source_file = str(first_component_meta["source_file"])
        probes = probe_components(
            component_prefixes=decoded["component_prefixes"],
            source_file=source_file,
            splits_path=args.splits,
            split_family=args.split_family,
            c_values=args.c_values,
            max_iter=args.max_iter,
            solver=args.solver,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + idx * 10007,
            skip_target_probe=args.skip_target_probe,
        )
        report["results"][decoded["task"]] = {
            **decoded,
            "source_file": source_file,
            "probes": probes,
        }
    write_json(args.output, strip_artifacts(report))
    print(args.output, flush=True)
    for task, result in report["results"].items():
        aucs = {
            component: probe.get("test_auc")
            for component, probe in result["probes"].items()
        }
        energy = result["reconstruction_stats"].get("energy_explained")
        print(f"{task}: aucs={aucs} energy={energy}", flush=True)


if __name__ == "__main__":
    main()
