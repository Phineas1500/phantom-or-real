#!/usr/bin/env python3
"""Probe skip-transcoder latent, affine-skip, full-output, and error components."""

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
from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    read_json,
    read_jsonl,
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


COMPONENTS = ("latent", "skip", "full", "error")
SKIP_ORIENTATIONS = ("sae_lens", "untransposed")


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def component_prefix(
    *,
    out_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    transcoder_id: str,
    top_k: int,
    component: str,
) -> Path:
    return out_dir / f"{model_key}_{task}_L{layer}_{transcoder_id}_top{top_k}_{component}"


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


def assert_aligned_sidecars(left_path: Path, right_path: Path) -> None:
    left_rows = read_jsonl(left_path)
    right_rows = read_jsonl(right_path)
    if len(left_rows) != len(right_rows):
        raise ValueError(f"{left_path} rows {len(left_rows)} != {right_path} rows {len(right_rows)}")
    for idx, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
        left_key = (left.get("row_index"), left.get("example_id"))
        right_key = (right.get("row_index"), right.get("example_id"))
        if left_key != right_key:
            raise ValueError(f"sidecar mismatch at row {idx}: {left_key} != {right_key}")


def apply_skip_connection(
    *,
    sae: Any,
    input_chunk: torch.Tensor,
    dtype: torch.dtype,
    skip_orientation: str,
) -> torch.Tensor:
    if skip_orientation == "sae_lens":
        return input_chunk @ sae.W_skip.T.to(device=input_chunk.device, dtype=dtype)
    if skip_orientation == "untransposed":
        return input_chunk @ sae.W_skip.to(device=input_chunk.device, dtype=dtype)
    raise ValueError(f"unknown skip orientation {skip_orientation!r}; expected one of {SKIP_ORIENTATIONS}")


def decode_components_one_dataset(
    *,
    sae: Any,
    input_prefix: Path,
    target_prefix: Path,
    feature_prefix: Path,
    out_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    transcoder_id: str,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    chunk_size: int,
    verify_dense_rows: int,
    skip_orientation: str,
) -> dict[str, Any]:
    started = time.time()
    input_file = input_prefix.with_suffix(".safetensors")
    input_sidecar_file = input_prefix.with_suffix(".example_ids.jsonl")
    input_meta_file = input_prefix.with_suffix(".meta.json")
    target_file = target_prefix.with_suffix(".safetensors")
    target_sidecar_file = target_prefix.with_suffix(".example_ids.jsonl")
    target_meta_file = target_prefix.with_suffix(".meta.json")
    feature_file = feature_prefix.with_suffix(".safetensors")
    feature_sidecar_file = feature_prefix.with_suffix(".example_ids.jsonl")
    feature_meta_file = feature_prefix.with_suffix(".meta.json")

    assert_aligned_sidecars(input_sidecar_file, target_sidecar_file)
    assert_aligned_sidecars(input_sidecar_file, feature_sidecar_file)

    input_activations = load_file(input_file)["activations"]
    target_activations = load_file(target_file)["activations"]
    feature_tensors = load_file(feature_file)
    top_values = feature_tensors["top_values"]
    top_indices = feature_tensors["top_indices"]

    if input_activations.shape != target_activations.shape:
        raise ValueError(f"input shape {tuple(input_activations.shape)} != target shape {tuple(target_activations.shape)}")
    if input_activations.shape[0] != top_values.shape[0]:
        raise ValueError(f"input rows {input_activations.shape[0]} != feature rows {top_values.shape[0]}")
    if input_activations.shape[1] != sae.cfg.d_in:
        raise ValueError(f"input dim {input_activations.shape[1]} != transcoder d_in {sae.cfg.d_in}")
    if target_activations.shape[1] != sae.cfg.d_out:
        raise ValueError(f"target dim {target_activations.shape[1]} != transcoder d_out {sae.cfg.d_out}")
    if not hasattr(sae, "W_skip"):
        raise ValueError(f"{transcoder_id} does not expose W_skip; expected an affine skip-transcoder")

    sparse_decode_check = verify_sparse_decode(
        sae=sae,
        top_indices=top_indices,
        top_values=top_values,
        rows=verify_dense_rows,
        dtype=dtype,
        device=device,
    )

    stats = {component: ReconstructionStats() for component in ("latent", "skip", "full")}
    chunks: dict[str, list[torch.Tensor]] = {component: [] for component in COMPONENTS}
    for start in range(0, input_activations.shape[0], chunk_size):
        end = min(start + chunk_size, input_activations.shape[0])
        input_chunk = input_activations[start:end].to(device=device, dtype=dtype)
        target_chunk = target_activations[start:end].to(device=device, dtype=dtype)
        indices_chunk = top_indices[start:end].to(device=device, dtype=torch.long)
        values_chunk = top_values[start:end].to(device=device, dtype=dtype)
        with torch.inference_mode():
            latent = decode_topk_linear(sae, indices_chunk, values_chunk, dtype=dtype)
            skip = apply_skip_connection(
                sae=sae,
                input_chunk=input_chunk,
                dtype=dtype,
                skip_orientation=skip_orientation,
            )
            full = latent + skip
            error = target_chunk - full
        for component, tensor in (("latent", latent), ("skip", skip), ("full", full)):
            stats[component].update(target_chunk, tensor)
        for component, tensor in (("latent", latent), ("skip", skip), ("full", full), ("error", error)):
            chunks[component].append(tensor.to(dtype=output_dtype).cpu())
        del input_chunk, target_chunk, indices_chunk, values_chunk, latent, skip, full, error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{feature_prefix.name}: component rows {end}/{input_activations.shape[0]}", flush=True)

    component_tensors = {component: torch.cat(component_chunks, dim=0) for component, component_chunks in chunks.items()}
    prefixes = {
        component: component_prefix(
            out_dir=out_dir,
            model_key=model_key,
            task=task,
            layer=layer,
            transcoder_id=transcoder_id,
            top_k=top_k,
            component=component,
        )
        for component in COMPONENTS
    }
    for component, tensor in component_tensors.items():
        save_file({"activations": tensor}, prefixes[component].with_suffix(".safetensors"))
        shutil.copyfile(feature_sidecar_file, prefixes[component].with_suffix(".example_ids.jsonl"))

    input_meta = read_json(input_meta_file)
    target_meta = read_json(target_meta_file)
    feature_meta = read_json(feature_meta_file)
    common_meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_activation_file": display_path(input_file),
        "input_activation_sha256": sha256_file(input_file),
        "input_sidecar_file": display_path(input_sidecar_file),
        "input_sidecar_sha256": sha256_file(input_sidecar_file),
        "input_activation_meta": input_meta,
        "target_activation_file": display_path(target_file),
        "target_activation_sha256": sha256_file(target_file),
        "target_sidecar_file": display_path(target_sidecar_file),
        "target_sidecar_sha256": sha256_file(target_sidecar_file),
        "target_activation_meta": target_meta,
        "feature_file": display_path(feature_file),
        "feature_sha256": sha256_file(feature_file),
        "feature_sidecar_file": display_path(feature_sidecar_file),
        "feature_sidecar_sha256": sha256_file(feature_sidecar_file),
        "feature_meta": feature_meta,
        "sae_release": feature_meta["sae_release"],
        "sae_id": feature_meta["sae_id"],
        "hf_snapshot_revision": feature_meta.get("hf_snapshot_revision"),
        "layer": layer,
        "task": task,
        "model_key": model_key,
        "shape": list(next(iter(component_tensors.values())).shape),
        "dtype": str(output_dtype),
        "decode_dtype": str(dtype),
        "device": device,
        "chunk_size": chunk_size,
        "top_k": int(top_values.shape[1]),
        "skip_orientation": skip_orientation,
        "sparse_decode_check": sparse_decode_check,
        "component_stats_vs_target": {component: summary.to_dict() for component, summary in stats.items()},
        "elapsed_seconds": time.time() - started,
        "jsonl_path": target_meta.get("jsonl_path"),
    }
    for component, prefix in prefixes.items():
        meta = {
            **common_meta,
            "kind": f"transcoder_{component}",
            "component": component,
            "activation_file": display_path(prefix.with_suffix(".safetensors")),
            "sidecar_file": display_path(prefix.with_suffix(".example_ids.jsonl")),
        }
        write_json(prefix.with_suffix(".meta.json"), meta)

    return {
        "input_prefix": str(input_prefix),
        "target_prefix": str(target_prefix),
        "feature_prefix": str(feature_prefix),
        "component_prefixes": {component: str(prefix) for component, prefix in prefixes.items()},
        "component_stats_vs_target": {component: summary.to_dict() for component, summary in stats.items()},
        "sparse_decode_check": sparse_decode_check,
        "elapsed_seconds": time.time() - started,
    }


def probe_components(
    *,
    component_prefixes: dict[str, str],
    target_prefix: Path,
    splits_path: Path,
    split_family: str,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
    bootstrap_samples: int,
    seed: int,
    layer: int,
    skip_target_probe: bool,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    target_meta = read_json(target_prefix.with_suffix(".meta.json"))
    source_file = target_meta.get("jsonl_path")
    probes = {}
    if not skip_target_probe:
        print(f"probing target {target_prefix.name} split={split_family}", flush=True)
        probes["target"] = run_raw_activation_probe(
            activation_path=target_prefix.with_suffix(".safetensors"),
            sidecar_path=target_prefix.with_suffix(".example_ids.jsonl"),
            seed=seed + layer,
            drop_parse_failed=True,
            split_assignments=split_assignments,
            source_file=source_file,
            split_family=split_family,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
            bootstrap_samples=bootstrap_samples,
        )
    for component, prefix_str in component_prefixes.items():
        prefix = Path(prefix_str)
        print(f"probing component {prefix.name} split={split_family}", flush=True)
        probes[component] = run_raw_activation_probe(
            activation_path=prefix.with_suffix(".safetensors"),
            sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
            seed=seed + layer,
            drop_parse_failed=True,
            split_assignments=split_assignments,
            source_file=source_file,
            split_family=split_family,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
            bootstrap_samples=bootstrap_samples,
        )
    return probes


def existing_component_prefixes(
    *,
    out_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    transcoder_id: str,
    top_k: int,
) -> dict[str, str]:
    return {
        component: str(
            component_prefix(
                out_dir=out_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                transcoder_id=transcoder_id,
                top_k=top_k,
                component=component,
            )
        )
        for component in COMPONENTS
    }


def run_diagnostics(
    *,
    input_activation_dir: Path,
    target_activation_dir: Path,
    feature_dir: Path,
    out_dir: Path,
    model_key: str,
    tasks: list[str],
    layer: int,
    input_activation_site: str,
    target_activation_site: str,
    transcoder_release: str,
    transcoder_id: str,
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
    reuse_existing: bool,
    skip_target_probe: bool,
    skip_orientation: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_activation_dir": str(input_activation_dir),
        "target_activation_dir": str(target_activation_dir),
        "feature_dir": str(feature_dir),
        "out_dir": str(out_dir),
        "model_key": model_key,
        "tasks": tasks,
        "layer": layer,
        "input_activation_site": input_activation_site,
        "target_activation_site": target_activation_site,
        "transcoder_release": transcoder_release,
        "transcoder_id": transcoder_id,
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
        "reuse_existing": reuse_existing,
        "skip_target_probe": skip_target_probe,
        "skip_orientation": skip_orientation,
        "results": {},
    }

    sae = None
    if not reuse_existing:
        from sae_lens import SAE

        print(f"Loading transcoder {transcoder_release} / {transcoder_id}", flush=True)
        sae, _cfg, _sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=transcoder_release,
            sae_id=transcoder_id,
            device=device,
            dtype=str(dtype).removeprefix("torch."),
        )
        sae.eval()

    try:
        for task in tasks:
            input_name = activation_stem(
                model_key=model_key,
                task=task,
                layer=layer,
                activation_site=input_activation_site,
            )
            target_name = activation_stem(
                model_key=model_key,
                task=task,
                layer=layer,
                activation_site=target_activation_site,
            )
            input_prefix = input_activation_dir / input_name
            target_prefix = target_activation_dir / target_name
            feature_prefix = feature_dir / f"{input_name}_{transcoder_id}_top{top_k}"

            if reuse_existing:
                prefixes = existing_component_prefixes(
                    out_dir=out_dir,
                    model_key=model_key,
                    task=task,
                    layer=layer,
                    transcoder_id=transcoder_id,
                    top_k=top_k,
                )
                first_meta = read_json(Path(prefixes["full"]).with_suffix(".meta.json"))
                decode_result = {
                    "input_prefix": str(input_prefix),
                    "target_prefix": str(target_prefix),
                    "feature_prefix": str(feature_prefix),
                    "component_prefixes": prefixes,
                    "component_stats_vs_target": first_meta.get("component_stats_vs_target"),
                    "sparse_decode_check": first_meta.get("sparse_decode_check"),
                    "elapsed_seconds": None,
                }
            else:
                assert sae is not None
                decode_result = decode_components_one_dataset(
                    sae=sae,
                    input_prefix=input_prefix,
                    target_prefix=target_prefix,
                    feature_prefix=feature_prefix,
                    out_dir=out_dir,
                    model_key=model_key,
                    task=task,
                    layer=layer,
                    transcoder_id=transcoder_id,
                    top_k=top_k,
                    device=device,
                    dtype=dtype,
                    output_dtype=output_dtype,
                    chunk_size=chunk_size,
                    verify_dense_rows=verify_dense_rows,
                    skip_orientation=skip_orientation,
                )

            probes = probe_components(
                component_prefixes=decode_result["component_prefixes"],
                target_prefix=target_prefix,
                splits_path=splits_path,
                split_family=split_family,
                c_values=c_values,
                max_iter=max_iter,
                solver=solver,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                layer=layer,
                skip_target_probe=skip_target_probe,
            )
            report["results"][task] = {
                **decode_result,
                "probes": probes,
            }
    finally:
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-activation-dir", type=Path, required=True)
    parser.add_argument("--target-activation-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--input-activation-site", default="mlp_in")
    parser.add_argument("--target-activation-site", default="mlp_out")
    parser.add_argument("--transcoder-release", required=True)
    parser.add_argument("--transcoder-id", required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--skip-target-probe",
        action="store_true",
        help="Skip re-probing the target activation matrix when an existing raw report is available.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dtype", default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--verify-dense-rows", type=int, default=4)
    parser.add_argument(
        "--skip-orientation",
        choices=SKIP_ORIENTATIONS,
        default="sae_lens",
        help=(
            "`sae_lens` uses x @ W_skip.T, matching SAE Lens SkipTranscoder.forward. "
            "`untransposed` uses x @ W_skip, which the L45 262K hook audit found "
            "to best match Gemma Scope's raw affine_skip_connection tensor."
        ),
    )
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args()

    report = run_diagnostics(
        input_activation_dir=args.input_activation_dir,
        target_activation_dir=args.target_activation_dir,
        feature_dir=args.feature_dir,
        out_dir=args.out_dir,
        model_key=args.model_key,
        tasks=args.tasks,
        layer=args.layer,
        input_activation_site=args.input_activation_site,
        target_activation_site=args.target_activation_site,
        transcoder_release=args.transcoder_release,
        transcoder_id=args.transcoder_id,
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
        reuse_existing=args.reuse_existing,
        skip_target_probe=args.skip_target_probe,
        skip_orientation=args.skip_orientation,
    )
    write_json(args.output, report)
    print(args.output)
    for task, result in report["results"].items():
        summary = {
            component: probe.get("test_auc")
            for component, probe in result["probes"].items()
        }
        energy = result["component_stats_vs_target"]["full"].get("energy_explained")
        print(f"{task}: aucs={summary} full_energy={energy}", flush=True)


if __name__ == "__main__":
    main()
