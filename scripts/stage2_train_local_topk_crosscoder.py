#!/usr/bin/env python3
"""Train a local top-k multi-layer dictionary on concatenated activations."""

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

from src.activations import parse_int_list, sha256_file  # noqa: E402
from src.stage2_crosscoder import crosscoder_feature_prefix, verify_matching_sidecars  # noqa: E402
from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    write_json,
)
from src.stage2_sae import display_path, write_jsonl  # noqa: E402


@dataclass(frozen=True)
class MultiLayerTask:
    task: str
    prefixes: list[Path]
    activation_files: list[Path]
    sidecar_file: Path
    meta_files: list[Path]
    metas: list[dict[str, Any]]
    sidecar: list[dict[str, Any]]
    keep_indices: list[int]
    source_file: str


class TopKCrosscoder(torch.nn.Module):
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
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dictionary-dir", type=Path, default=Path("results/stage2/local_dictionaries"))
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--crosscoder-id", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--latent-top-k", type=int, default=64)
    parser.add_argument("--feature-top-k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--drop-parse-failed", action="store_true", default=True)
    parser.add_argument("--keep-parse-failed", action="store_false", dest="drop_parse_failed")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-rows", type=int, default=2048)
    return parser.parse_args()


def load_task(
    *,
    activation_dir: Path,
    model_key: str,
    task: str,
    layers: list[int],
    drop_parse_failed: bool,
) -> MultiLayerTask:
    prefixes = [
        activation_dir / activation_stem(model_key=model_key, task=task, layer=layer)
        for layer in layers
    ]
    sidecars = [read_jsonl(prefix.with_suffix(".example_ids.jsonl")) for prefix in prefixes]
    reference = sidecars[0]
    for rows in sidecars[1:]:
        verify_matching_sidecars(reference, rows)
    keep_indices = [
        idx
        for idx, row in enumerate(reference)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    metas = [read_json(prefix.with_suffix(".meta.json")) for prefix in prefixes]
    source_file = metas[0].get("jsonl_path")
    if not source_file:
        raise KeyError(f"{prefixes[0].with_suffix('.meta.json')} is missing jsonl_path")
    return MultiLayerTask(
        task=task,
        prefixes=prefixes,
        activation_files=[prefix.with_suffix(".safetensors") for prefix in prefixes],
        sidecar_file=prefixes[0].with_suffix(".example_ids.jsonl"),
        meta_files=[prefix.with_suffix(".meta.json") for prefix in prefixes],
        metas=metas,
        sidecar=[reference[idx] for idx in keep_indices],
        keep_indices=keep_indices,
        source_file=str(source_file),
    )


def load_concat_activations(task: MultiLayerTask) -> torch.Tensor:
    parts = []
    for activation_file in task.activation_files:
        tensor = load_file(activation_file)["activations"]
        if tensor.shape[0] < max(task.keep_indices, default=-1) + 1:
            raise ValueError(f"{activation_file} has fewer rows than its sidecar")
        parts.append(tensor[task.keep_indices].float().cpu())
    return torch.cat(parts, dim=1)


def train_rows_for_task(
    task: MultiLayerTask,
    *,
    assignments: dict[tuple[str, int], dict[str, Any]],
    split_family: str,
) -> torch.Tensor:
    x = load_concat_activations(task)
    splits = split_indices_from_assignments(
        task.sidecar,
        assignments=assignments,
        source_file=task.source_file,
        split_field=f"{split_family}_split",
    )
    train_indices = splits["train"]
    if not train_indices:
        raise ValueError(f"no train rows for {task.task} / {split_family}")
    return x[train_indices]


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def amp_dtype_for(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.float32)


def train_crosscoder(
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
) -> tuple[TopKCrosscoder, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-4)
    mean_device = mean.to(device)
    std_device = std.to(device)
    model = TopKCrosscoder(d_in=int(train_x.shape[1]), d_sae=d_sae, latent_top_k=latent_top_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(seed)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    dtype = amp_dtype_for(amp_dtype)
    losses: list[dict[str, float]] = []
    started = time.time()
    for step in range(1, steps + 1):
        indices = torch.randint(0, train_x.shape[0], (batch_size,), generator=cpu_rng)
        batch = standardize(train_x[indices].to(device, non_blocking=True), mean_device, std_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
            recon, sparse_acts = model(batch)
            loss = torch.mean((recon.float() - batch.float()) ** 2)
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
    return model.eval(), mean, std, losses


def reconstruction_report(
    model: TopKCrosscoder,
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
    mean_device = mean.to(device)
    std_device = std.to(device)
    batch = standardize(train_x[:n].to(device), mean_device, std_device)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    dtype = amp_dtype_for(amp_dtype)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
        recon, sparse_acts = model(batch)
        mse = torch.mean((recon.float() - batch.float()) ** 2)
        total = torch.mean(batch.float() ** 2).clamp_min(1e-12)
    return {
        "eval_rows": float(n),
        "standardized_mse": float(mse.cpu().item()),
        "standardized_energy_explained": float((1.0 - mse / total).cpu().item()),
        "mean_l0": float((sparse_acts > 0).sum(dim=1).float().mean().cpu().item()),
    }


def encode_task(
    task: MultiLayerTask,
    *,
    model: TopKCrosscoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_dir: Path,
    model_key: str,
    crosscoder_id: str,
    feature_top_k: int,
    encode_batch_size: int,
    device: str,
    amp_dtype: str,
    dictionary_meta: dict[str, Any],
) -> Path:
    x = load_concat_activations(task)
    output_prefix = crosscoder_feature_prefix(
        out_dir=out_dir,
        model_key=model_key,
        task=task.task,
        crosscoder_id=crosscoder_id,
        top_k=feature_top_k,
    )
    feature_file = output_prefix.with_suffix(".safetensors")
    sidecar_file = output_prefix.with_suffix(".example_ids.jsonl")
    meta_file = output_prefix.with_suffix(".meta.json")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    mean_device = mean.to(device)
    std_device = std.to(device)
    amp_enabled = device.startswith("cuda") and amp_dtype != "none"
    dtype = amp_dtype_for(amp_dtype)

    top_values: list[torch.Tensor] = []
    top_indices: list[torch.Tensor] = []
    l0_counts: list[torch.Tensor] = []
    started = time.time()
    for start in range(0, x.shape[0], encode_batch_size):
        end = min(start + encode_batch_size, x.shape[0])
        batch = standardize(x[start:end].to(device), mean_device, std_device)
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
        print(f"encoded {task.task} rows {end}/{x.shape[0]}", flush=True)

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
    write_jsonl(sidecar_file, task.sidecar)
    elapsed = time.time() - started
    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "crosscoder_id": crosscoder_id,
        "crosscoder_release": "local_topk_crosscoder",
        "source_file": task.source_file,
        "source_activation_files": [display_path(path) for path in task.activation_files],
        "source_activation_sha256": [sha256_file(path) for path in task.activation_files],
        "source_activation_metas": task.metas,
        "source_sidecar_file": display_path(task.sidecar_file),
        "source_sidecar_sha256": sha256_file(task.sidecar_file),
        "feature_file": display_path(feature_file),
        "sidecar_file": display_path(sidecar_file),
        "row_count": int(top_values_tensor.shape[0]),
        "input_rows": int(task.metas[0].get("row_count", len(task.sidecar))),
        "drop_parse_failed": bool(dictionary_meta["drop_parse_failed"]),
        "top_k": int(top_values_tensor.shape[1]),
        "crosscoder_cfg": {
            "architecture": "local_topk_crosscoder",
            "d_in": int(model.d_in),
            "d_sae": int(model.d_sae),
            "d_out": int(model.d_in),
            "layers": dictionary_meta["layers"],
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

    tasks = [
        load_task(
            activation_dir=args.activation_dir,
            model_key=args.model_key,
            task=task,
            layers=args.layers,
            drop_parse_failed=args.drop_parse_failed,
        )
        for task in args.tasks
    ]
    assignments = read_split_assignments(args.splits)
    train_parts = [
        train_rows_for_task(task, assignments=assignments, split_family=args.split_family)
        for task in tasks
    ]
    train_x = torch.cat(train_parts, dim=0)
    print(
        f"Training {args.crosscoder_id}: rows={train_x.shape[0]} d_in={train_x.shape[1]} "
        f"d_sae={args.d_sae} latent_top_k={args.latent_top_k} split={args.split_family}",
        flush=True,
    )
    model, mean, std, losses = train_crosscoder(
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
    dictionary_file = args.dictionary_dir / f"{args.crosscoder_id}.safetensors"
    dictionary_meta_file = args.dictionary_dir / f"{args.crosscoder_id}.meta.json"
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
        "dictionary_id": args.crosscoder_id,
        "dictionary_file": display_path(dictionary_file),
        "dictionary_sha256": sha256_file(dictionary_file),
        "architecture": "local_topk_crosscoder",
        "d_in": int(train_x.shape[1]),
        "d_sae": int(args.d_sae),
        "latent_top_k": int(args.latent_top_k),
        "feature_top_k": int(args.feature_top_k),
        "layers": [int(layer) for layer in args.layers],
        "split_family": args.split_family,
        "splits_path": str(args.splits),
        "drop_parse_failed": bool(args.drop_parse_failed),
        "train_rows": int(train_x.shape[0]),
        "tasks": [task.task for task in tasks],
        "source_files": sorted({task.source_file for task in tasks}),
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
        encode_task(
            task,
            model=model,
            mean=mean,
            std=std,
            out_dir=args.out_dir,
            model_key=args.model_key,
            crosscoder_id=args.crosscoder_id,
            feature_top_k=args.feature_top_k,
            encode_batch_size=args.encode_batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
            dictionary_meta=dictionary_meta,
        )
        for task in tasks
    ]
    print(json.dumps({"encoded_prefixes": [str(prefix) for prefix in encoded_prefixes]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
