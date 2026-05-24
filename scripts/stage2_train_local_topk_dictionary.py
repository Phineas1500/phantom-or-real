#!/usr/bin/env python3
"""Train a local top-k sparse dictionary on cached Stage 2 activations.

This is for Qwen-side coverage when a public sparse artifact family is missing.
The script trains only on the requested split's train rows, then encodes every
row in each activation file into the same top-k feature format used by the
Gemma/Qwen SAE probe scripts.
"""

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


class TopKAutoencoder(torch.nn.Module):
    def __init__(self, d_in: int, d_sae: int, latent_top_k: int) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.latent_top_k = int(latent_top_k)
        self.encoder = torch.nn.Linear(self.d_in, self.d_sae)
        self.decoder = torch.nn.Linear(self.d_sae, self.d_in)
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
    parser.add_argument("--activation-prefix", action="append", type=Path, required=True)
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


def load_kept_activations(bundle: ActivationBundle) -> torch.Tensor:
    tensor = load_file(bundle.activation_file)["activations"]
    if tensor.shape[0] < max(bundle.keep_indices, default=-1) + 1:
        raise ValueError(f"{bundle.activation_file} has fewer rows than its sidecar")
    return tensor[bundle.keep_indices].float().cpu()


def train_rows_for_bundle(
    bundle: ActivationBundle,
    *,
    assignments: dict[tuple[str, int], dict[str, Any]],
    split_family: str,
) -> torch.Tensor:
    x = load_kept_activations(bundle)
    splits = split_indices_from_assignments(
        bundle.sidecar,
        assignments=assignments,
        source_file=bundle.source_file,
        split_field=f"{split_family}_split",
    )
    train_indices = splits["train"]
    if not train_indices:
        raise ValueError(f"no train rows for {bundle.prefix} / {split_family}")
    return x[train_indices]


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def train_dictionary(
    train_x: torch.Tensor,
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
) -> tuple[TopKAutoencoder, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    if train_x.ndim != 2:
        raise ValueError(f"expected rank-2 training activations, got {tuple(train_x.shape)}")
    if train_x.shape[0] == 0:
        raise ValueError("cannot train on zero rows")
    if d_sae <= 0 or latent_top_k <= 0 or steps <= 0 or batch_size <= 0:
        raise ValueError("d_sae, latent_top_k, steps, and batch_size must be positive")

    d_in = int(train_x.shape[1])
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-4)
    mean_device = mean.to(device)
    std_device = std.to(device)

    model = TopKAutoencoder(d_in=d_in, d_sae=d_sae, latent_top_k=latent_top_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(seed)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    amp_torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(amp_dtype, torch.float32)

    losses: list[dict[str, float]] = []
    started = time.time()
    for step in range(1, steps + 1):
        indices = torch.randint(0, train_x.shape[0], (batch_size,), generator=cpu_rng)
        batch = train_x[indices].to(device, non_blocking=True)
        batch = standardize(batch, mean_device, std_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_torch_dtype, enabled=amp_enabled):
            recon, sparse_acts = model(batch)
            loss = torch.mean((recon.float() - batch.float()) ** 2)
        loss.backward()
        optimizer.step()

        if step == 1 or step == steps or step % log_every == 0:
            elapsed = time.time() - started
            mean_l0 = float((sparse_acts.detach() > 0).sum(dim=1).float().mean().item())
            record = {
                "step": float(step),
                "loss": float(loss.detach().cpu().item()),
                "mean_l0": mean_l0,
                "elapsed_seconds": float(elapsed),
            }
            losses.append(record)
            print(
                f"step {step}/{steps} loss={record['loss']:.6f} "
                f"mean_l0={record['mean_l0']:.2f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return model.eval(), mean, std, losses


def reconstruction_report(
    model: TopKAutoencoder,
    train_x: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    eval_rows: int,
    amp_dtype: str,
) -> dict[str, float]:
    n = min(int(eval_rows), train_x.shape[0])
    if n <= 0:
        return {}
    batch = train_x[:n].to(device)
    mean_device = mean.to(device)
    std_device = std.to(device)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    amp_torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(amp_dtype, torch.float32)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_torch_dtype, enabled=amp_enabled):
        x = standardize(batch, mean_device, std_device)
        recon, sparse_acts = model(x)
        mse = torch.mean((recon.float() - x.float()) ** 2)
        total = torch.mean(x.float() ** 2).clamp_min(1e-12)
        mean_l0 = (sparse_acts > 0).sum(dim=1).float().mean()
    return {
        "eval_rows": float(n),
        "standardized_mse": float(mse.cpu().item()),
        "standardized_energy_explained": float((1.0 - mse / total).cpu().item()),
        "mean_l0": float(mean_l0.cpu().item()),
    }


def feature_prefix_for(*, activation_prefix: Path, out_dir: Path, dictionary_id: str, feature_top_k: int) -> Path:
    return out_dir / f"{activation_prefix.name}_{dictionary_id}_top{feature_top_k}"


def encode_bundle(
    bundle: ActivationBundle,
    *,
    model: TopKAutoencoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_dir: Path,
    dictionary_id: str,
    feature_top_k: int,
    encode_batch_size: int,
    device: str,
    amp_dtype: str,
    dictionary_meta: dict[str, Any],
) -> Path:
    if feature_top_k <= 0 or encode_batch_size <= 0:
        raise ValueError("feature_top_k and encode_batch_size must be positive")
    x_all = load_kept_activations(bundle)
    output_prefix = feature_prefix_for(
        activation_prefix=bundle.prefix,
        out_dir=out_dir,
        dictionary_id=dictionary_id,
        feature_top_k=feature_top_k,
    )
    feature_file = output_prefix.with_suffix(".safetensors")
    sidecar_file = output_prefix.with_suffix(".example_ids.jsonl")
    meta_file = output_prefix.with_suffix(".meta.json")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    mean_device = mean.to(device)
    std_device = std.to(device)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    amp_torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(amp_dtype, torch.float32)
    top_values: list[torch.Tensor] = []
    top_indices: list[torch.Tensor] = []
    l0_counts: list[torch.Tensor] = []
    started = time.time()
    for start in range(0, x_all.shape[0], encode_batch_size):
        end = min(start + encode_batch_size, x_all.shape[0])
        batch = x_all[start:end].to(device)
        batch = standardize(batch, mean_device, std_device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_torch_dtype, enabled=amp_enabled):
            sparse_acts = model.encode_dense(batch)
            k = min(feature_top_k, sparse_acts.shape[-1])
            values, indices = torch.topk(sparse_acts, k=k, dim=-1)
            l0 = (values > 0).sum(dim=-1).to(torch.int32)
        top_values.append(values.float().cpu())
        top_indices.append(indices.to(torch.int64).cpu())
        l0_counts.append(l0.cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"encoded {bundle.prefix.name} rows {end}/{x_all.shape[0]}", flush=True)

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
    write_jsonl(sidecar_file, bundle.sidecar)

    elapsed = time.time() - started
    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_file": display_path(bundle.activation_file),
        "activation_sha256": sha256_file(bundle.activation_file),
        "source_activation_meta": bundle.meta,
        "source_sidecar_file": display_path(bundle.sidecar_file),
        "source_sidecar_sha256": sha256_file(bundle.sidecar_file),
        "feature_file": display_path(feature_file),
        "sidecar_file": display_path(sidecar_file),
        "row_count": int(top_values_tensor.shape[0]),
        "input_rows": int(bundle.meta.get("row_count", len(bundle.sidecar))),
        "drop_parse_failed": bool(dictionary_meta["drop_parse_failed"]),
        "top_k": int(top_values_tensor.shape[1]),
        "sae_release": "local_topk_autoencoder",
        "sae_id": dictionary_id,
        "sae_cfg": {
            "architecture": "local_topk_autoencoder",
            "d_in": int(model.d_in),
            "d_sae": int(model.d_sae),
            "d_out": int(model.d_in),
            "hook_name": bundle.meta.get("hook_name"),
            "hf_hook_name": bundle.meta.get("hook_name"),
            "hook_name_out": bundle.meta.get("hook_name"),
            "hf_hook_name_out": bundle.meta.get("hook_name"),
            "latent_top_k": int(model.latent_top_k),
            "standardized_input": True,
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    bundles = [load_bundle(prefix, drop_parse_failed=args.drop_parse_failed) for prefix in args.activation_prefix]
    if not bundles:
        raise ValueError("at least one activation prefix is required")
    assignments = read_split_assignments(args.splits)
    train_parts = [
        train_rows_for_bundle(bundle, assignments=assignments, split_family=args.split_family)
        for bundle in bundles
    ]
    d_in_values = {int(part.shape[1]) for part in train_parts}
    if len(d_in_values) != 1:
        raise ValueError(f"all activation prefixes must share width, got {sorted(d_in_values)}")
    train_x = torch.cat(train_parts, dim=0)
    print(
        f"Training {args.dictionary_id}: rows={train_x.shape[0]} d_in={train_x.shape[1]} "
        f"d_sae={args.d_sae} latent_top_k={args.latent_top_k} split={args.split_family}",
        flush=True,
    )

    model, mean, std, losses = train_dictionary(
        train_x,
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
        train_x,
        mean=mean,
        std=std,
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
            "input_mean": mean.cpu(),
            "input_std": std.cpu(),
        },
        dictionary_file,
    )
    dictionary_meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dictionary_id": args.dictionary_id,
        "dictionary_file": display_path(dictionary_file),
        "dictionary_sha256": sha256_file(dictionary_file),
        "architecture": "local_topk_autoencoder",
        "d_in": int(train_x.shape[1]),
        "d_sae": int(args.d_sae),
        "latent_top_k": int(args.latent_top_k),
        "feature_top_k": int(args.feature_top_k),
        "split_family": args.split_family,
        "splits_path": str(args.splits),
        "drop_parse_failed": bool(args.drop_parse_failed),
        "train_rows": int(train_x.shape[0]),
        "activation_prefixes": [str(bundle.prefix) for bundle in bundles],
        "source_files": sorted({bundle.source_file for bundle in bundles}),
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
        encode_bundle(
            bundle,
            model=model,
            mean=mean,
            std=std,
            out_dir=args.out_dir,
            dictionary_id=args.dictionary_id,
            feature_top_k=args.feature_top_k,
            encode_batch_size=args.encode_batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
            dictionary_meta=dictionary_meta,
        )
        for bundle in bundles
    ]
    print(json.dumps({"encoded_prefixes": [str(prefix) for prefix in encoded_prefixes]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
