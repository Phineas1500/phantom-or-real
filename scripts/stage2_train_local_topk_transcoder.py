#!/usr/bin/env python3
"""Train a local top-k transcoder from one cached activation site to another."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import sha256_file  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    write_json,
)
from src.stage2_sae import display_path, write_jsonl  # noqa: E402


@dataclass(frozen=True)
class ActivationBundle:
    prefix: Path
    activation_file: Path
    sidecar_file: Path
    meta_file: Path
    meta: dict[str, Any]
    sidecar: list[dict[str, Any]]
    keep_indices: list[int]
    source_file: str


class TopKTranscoder(torch.nn.Module):
    def __init__(self, d_in: int, d_sae: int, d_out: int, latent_top_k: int) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.d_out = int(d_out)
        self.latent_top_k = int(latent_top_k)
        self.encoder = torch.nn.Linear(self.d_in, self.d_sae)
        self.decoder = torch.nn.Linear(self.d_sae, self.d_out)
        torch.nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        acts = torch.relu(self.encoder(x))
        k = min(self.latent_top_k, acts.shape[-1])
        values, indices = torch.topk(acts, k=k, dim=-1)
        sparse_acts = torch.zeros_like(acts)
        sparse_acts.scatter_(1, indices, values)
        return sparse_acts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_acts = self.encode_dense(x)
        return self.decoder(sparse_acts), sparse_acts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-activation-prefix", action="append", type=Path, required=True)
    parser.add_argument("--target-activation-prefix", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dictionary-dir", type=Path, default=Path("results/stage2/local_dictionaries"))
    parser.add_argument("--dictionary-id", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--latent-top-k", type=int, default=64)
    parser.add_argument("--feature-top-k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--drop-parse-failed", action="store_true", default=True)
    parser.add_argument("--keep-parse-failed", action="store_false", dest="drop_parse_failed")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-rows", type=int, default=4096)
    return parser.parse_args()


def load_bundle(prefix: Path, *, drop_parse_failed: bool) -> ActivationBundle:
    activation_file = prefix.with_suffix(".safetensors")
    sidecar_file = prefix.with_suffix(".example_ids.jsonl")
    meta_file = prefix.with_suffix(".meta.json")
    meta = read_json(meta_file)
    sidecar_all = read_jsonl(sidecar_file)
    keep_indices = [
        idx
        for idx, row in enumerate(sidecar_all)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    source_file = meta.get("jsonl_path")
    if not source_file:
        raise KeyError(f"{meta_file} is missing jsonl_path")
    return ActivationBundle(
        prefix=prefix,
        activation_file=activation_file,
        sidecar_file=sidecar_file,
        meta_file=meta_file,
        meta=meta,
        sidecar=[sidecar_all[idx] for idx in keep_indices],
        keep_indices=keep_indices,
        source_file=str(source_file),
    )


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("row_index"),
        row.get("example_id"),
        row.get("task"),
        row.get("height"),
        row.get("is_correct_strong"),
        row.get("parse_failed"),
    )


def assert_aligned(source: ActivationBundle, target: ActivationBundle) -> None:
    if source.source_file != target.source_file:
        raise ValueError(f"source files differ: {source.source_file} vs {target.source_file}")
    if len(source.sidecar) != len(target.sidecar):
        raise ValueError(f"sidecar lengths differ: {source.prefix} vs {target.prefix}")
    for idx, (left, right) in enumerate(zip(source.sidecar, target.sidecar, strict=True)):
        if row_key(left) != row_key(right):
            raise ValueError(f"source/target sidecar mismatch at row {idx}: {row_key(left)} != {row_key(right)}")


def load_kept_activations(bundle: ActivationBundle) -> torch.Tensor:
    tensor = load_file(bundle.activation_file)["activations"]
    if tensor.shape[0] < max(bundle.keep_indices, default=-1) + 1:
        raise ValueError(f"{bundle.activation_file} has fewer rows than its sidecar")
    return tensor[bundle.keep_indices].float().cpu()


def train_rows_for_pair(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    assignments: dict[tuple[str, int], dict[str, Any]],
    split_family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert_aligned(source, target)
    source_x = load_kept_activations(source)
    target_x = load_kept_activations(target)
    splits = split_indices_from_assignments(
        source.sidecar,
        assignments=assignments,
        source_file=source.source_file,
        split_field=f"{split_family}_split",
    )
    train_indices = splits["train"]
    if not train_indices:
        raise ValueError(f"no train rows for {source.prefix} / {split_family}")
    return source_x[train_indices], target_x[train_indices]


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def amp_dtype_for(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.float32)


def train_transcoder(
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    *,
    d_sae: int,
    latent_top_k: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    amp_dtype: str,
    log_every: int,
) -> tuple[TopKTranscoder, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    if source_x.ndim != 2 or target_x.ndim != 2:
        raise ValueError("source and target activations must be rank-2")
    if source_x.shape[0] != target_x.shape[0]:
        raise ValueError("source and target train row counts differ")
    source_mean = source_x.mean(dim=0)
    source_std = source_x.std(dim=0).clamp_min(1e-4)
    target_mean = target_x.mean(dim=0)
    target_std = target_x.std(dim=0).clamp_min(1e-4)
    source_mean_device = source_mean.to(device)
    source_std_device = source_std.to(device)
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)

    model = TopKTranscoder(
        d_in=int(source_x.shape[1]),
        d_sae=d_sae,
        d_out=int(target_x.shape[1]),
        latent_top_k=latent_top_k,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(seed)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    dtype = amp_dtype_for(amp_dtype)

    losses: list[dict[str, float]] = []
    started = time.time()
    for step in range(1, steps + 1):
        indices = torch.randint(0, source_x.shape[0], (batch_size,), generator=cpu_rng)
        source_batch = source_x[indices].to(device, non_blocking=True)
        target_batch = target_x[indices].to(device, non_blocking=True)
        source_batch = standardize(source_batch, source_mean_device, source_std_device)
        target_batch = standardize(target_batch, target_mean_device, target_std_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
            pred, sparse_acts = model(source_batch)
            loss = torch.mean((pred.float() - target_batch.float()) ** 2)
        loss.backward()
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            elapsed = time.time() - started
            record = {
                "step": float(step),
                "loss": float(loss.detach().cpu().item()),
                "mean_l0": float((sparse_acts.detach() > 0).sum(dim=1).float().mean().item()),
                "elapsed_seconds": float(elapsed),
            }
            losses.append(record)
            print(
                f"step {step}/{steps} loss={record['loss']:.6f} "
                f"mean_l0={record['mean_l0']:.2f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return model.eval(), source_mean, source_std, target_mean, target_std, losses


def reconstruction_report(
    model: TopKTranscoder,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    *,
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: str,
    eval_rows: int,
    amp_dtype: str,
) -> dict[str, float]:
    n = min(int(eval_rows), source_x.shape[0])
    if n <= 0:
        return {}
    source_batch = source_x[:n].to(device)
    target_batch = target_x[:n].to(device)
    source_mean_device = source_mean.to(device)
    source_std_device = source_std.to(device)
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    dtype = amp_dtype_for(amp_dtype)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
        source = standardize(source_batch, source_mean_device, source_std_device)
        target = standardize(target_batch, target_mean_device, target_std_device)
        pred, sparse_acts = model(source)
        mse = torch.mean((pred.float() - target.float()) ** 2)
        total = torch.mean(target.float() ** 2).clamp_min(1e-12)
    return {
        "eval_rows": float(n),
        "standardized_mse": float(mse.cpu().item()),
        "standardized_energy_explained": float((1.0 - mse / total).cpu().item()),
        "mean_l0": float((sparse_acts > 0).sum(dim=1).float().mean().cpu().item()),
    }


def feature_prefix_for(*, source_prefix: Path, out_dir: Path, dictionary_id: str, feature_top_k: int) -> Path:
    return out_dir / f"{source_prefix.name}_{dictionary_id}_top{feature_top_k}"


def encode_source(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    model: TopKTranscoder,
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    out_dir: Path,
    dictionary_id: str,
    feature_top_k: int,
    encode_batch_size: int,
    device: str,
    amp_dtype: str,
    dictionary_meta: dict[str, Any],
) -> Path:
    assert_aligned(source, target)
    source_x = load_kept_activations(source)
    output_prefix = feature_prefix_for(
        source_prefix=source.prefix,
        out_dir=out_dir,
        dictionary_id=dictionary_id,
        feature_top_k=feature_top_k,
    )
    feature_file = output_prefix.with_suffix(".safetensors")
    sidecar_file = output_prefix.with_suffix(".example_ids.jsonl")
    meta_file = output_prefix.with_suffix(".meta.json")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    source_mean_device = source_mean.to(device)
    source_std_device = source_std.to(device)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    dtype = amp_dtype_for(amp_dtype)
    top_values: list[torch.Tensor] = []
    top_indices: list[torch.Tensor] = []
    l0_counts: list[torch.Tensor] = []
    started = time.time()
    for start in range(0, source_x.shape[0], encode_batch_size):
        end = min(start + encode_batch_size, source_x.shape[0])
        batch = standardize(source_x[start:end].to(device), source_mean_device, source_std_device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
            sparse_acts = model.encode_dense(batch)
            k = min(feature_top_k, sparse_acts.shape[-1])
            values, indices = torch.topk(sparse_acts, k=k, dim=-1)
            l0 = (values > 0).sum(dim=-1).to(torch.int32)
        top_values.append(values.float().cpu())
        top_indices.append(indices.to(torch.int64).cpu())
        l0_counts.append(l0.cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"encoded {source.prefix.name} rows {end}/{source_x.shape[0]}", flush=True)

    top_values_tensor = torch.cat(top_values, dim=0)
    top_indices_tensor = torch.cat(top_indices, dim=0)
    l0_tensor = torch.cat(l0_counts, dim=0)
    save_file(
        {
            "top_values": top_values_tensor,
            "top_indices": top_indices_tensor,
            "l0": l0_tensor,
        },
        feature_file,
    )
    write_jsonl(sidecar_file, source.sidecar)
    elapsed = time.time() - started

    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_file": display_path(source.activation_file),
        "activation_sha256": sha256_file(source.activation_file),
        "source_activation_meta": source.meta,
        "target_activation_file": display_path(target.activation_file),
        "target_activation_sha256": sha256_file(target.activation_file),
        "target_activation_meta": target.meta,
        "source_sidecar_file": display_path(source.sidecar_file),
        "source_sidecar_sha256": sha256_file(source.sidecar_file),
        "feature_file": display_path(feature_file),
        "sidecar_file": display_path(sidecar_file),
        "row_count": int(top_values_tensor.shape[0]),
        "input_rows": int(source.meta.get("row_count", len(source.sidecar))),
        "drop_parse_failed": bool(dictionary_meta["drop_parse_failed"]),
        "top_k": int(top_values_tensor.shape[1]),
        "sae_release": "local_topk_transcoder",
        "sae_id": dictionary_id,
        "sae_cfg": {
            "architecture": "local_topk_transcoder",
            "d_in": int(model.d_in),
            "d_sae": int(model.d_sae),
            "d_out": int(model.d_out),
            "hook_name": source.meta.get("hook_name"),
            "hf_hook_name": source.meta.get("hook_name"),
            "hook_name_out": target.meta.get("hook_name"),
            "hf_hook_name_out": target.meta.get("hook_name"),
            "latent_top_k": int(model.latent_top_k),
            "standardized_source": True,
            "standardized_target": True,
        },
        "local_dictionary": dictionary_meta,
        "elapsed_seconds": elapsed,
        "rows_per_second": top_values_tensor.shape[0] / elapsed if elapsed > 0 else None,
        "l0_mean": float(l0_tensor.float().mean().item()) if l0_tensor.numel() else None,
        "top1_mean": float(top_values_tensor[:, 0].float().mean().item()) if top_values_tensor.numel() else None,
    }
    write_json(meta_file, meta)
    print(meta_file, flush=True)
    return output_prefix


def main() -> None:
    args = parse_args()
    if len(args.source_activation_prefix) != len(args.target_activation_prefix):
        raise ValueError("--source-activation-prefix and --target-activation-prefix counts must match")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    source_bundles = [
        load_bundle(prefix, drop_parse_failed=args.drop_parse_failed)
        for prefix in args.source_activation_prefix
    ]
    target_bundles = [
        load_bundle(prefix, drop_parse_failed=args.drop_parse_failed)
        for prefix in args.target_activation_prefix
    ]
    assignments = read_split_assignments(args.splits)
    source_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for source, target in zip(source_bundles, target_bundles, strict=True):
        source_train, target_train = train_rows_for_pair(
            source,
            target,
            assignments=assignments,
            split_family=args.split_family,
        )
        source_parts.append(source_train)
        target_parts.append(target_train)
    source_train_x = torch.cat(source_parts, dim=0)
    target_train_x = torch.cat(target_parts, dim=0)
    print(
        f"Training {args.dictionary_id}: rows={source_train_x.shape[0]} "
        f"d_in={source_train_x.shape[1]} d_out={target_train_x.shape[1]} "
        f"d_sae={args.d_sae} latent_top_k={args.latent_top_k} split={args.split_family}",
        flush=True,
    )

    model, source_mean, source_std, target_mean, target_std, losses = train_transcoder(
        source_train_x,
        target_train_x,
        d_sae=args.d_sae,
        latent_top_k=args.latent_top_k,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        amp_dtype=args.amp_dtype,
        log_every=args.log_every,
    )
    recon = reconstruction_report(
        model,
        source_train_x,
        target_train_x,
        source_mean=source_mean,
        source_std=source_std,
        target_mean=target_mean,
        target_std=target_std,
        device=args.device,
        eval_rows=args.eval_rows,
        amp_dtype=args.amp_dtype,
    )

    args.dictionary_dir.mkdir(parents=True, exist_ok=True)
    dictionary_file = args.dictionary_dir / f"{args.dictionary_id}.safetensors"
    dictionary_meta_file = args.dictionary_dir / f"{args.dictionary_id}.meta.json"
    save_file(
        {
            "encoder_weight": model.encoder.weight.detach().cpu(),
            "encoder_bias": model.encoder.bias.detach().cpu(),
            "decoder_weight": model.decoder.weight.detach().cpu(),
            "decoder_bias": model.decoder.bias.detach().cpu(),
            "source_mean": source_mean.cpu(),
            "source_std": source_std.cpu(),
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
        },
        dictionary_file,
    )
    dictionary_meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dictionary_id": args.dictionary_id,
        "dictionary_file": display_path(dictionary_file),
        "dictionary_sha256": sha256_file(dictionary_file),
        "architecture": "local_topk_transcoder",
        "d_in": int(source_train_x.shape[1]),
        "d_out": int(target_train_x.shape[1]),
        "d_sae": int(args.d_sae),
        "latent_top_k": int(args.latent_top_k),
        "feature_top_k": int(args.feature_top_k),
        "split_family": args.split_family,
        "splits_path": str(args.splits),
        "drop_parse_failed": bool(args.drop_parse_failed),
        "train_rows": int(source_train_x.shape[0]),
        "source_activation_prefixes": [str(bundle.prefix) for bundle in source_bundles],
        "target_activation_prefixes": [str(bundle.prefix) for bundle in target_bundles],
        "source_files": sorted({bundle.source_file for bundle in source_bundles}),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "loss_history": losses,
        "train_reconstruction": recon,
    }
    write_json(dictionary_meta_file, dictionary_meta)
    print(dictionary_meta_file, flush=True)

    encoded_prefixes = [
        encode_source(
            source,
            target,
            model=model,
            source_mean=source_mean,
            source_std=source_std,
            out_dir=args.out_dir,
            dictionary_id=args.dictionary_id,
            feature_top_k=args.feature_top_k,
            encode_batch_size=args.encode_batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
            dictionary_meta=dictionary_meta,
        )
        for source, target in zip(source_bundles, target_bundles, strict=True)
    ]
    print(json.dumps({"encoded_prefixes": [str(prefix) for prefix in encoded_prefixes]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
