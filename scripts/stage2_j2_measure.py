#!/usr/bin/env python3
"""Measure Gemma 3 27B Stage 2 headroom on Scholar J nodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from dotenv import load_dotenv
from safetensors.torch import save_file
from transformer_lens import HookedTransformer
from transformer_lens.utilities.multi_gpu import get_device_for_block_index


@dataclass(frozen=True)
class EncodedExample:
    row_index: int
    example_id: str | None
    height: int | None
    task: str | None
    token_ids: list[int]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def run(command: list[str]) -> dict:
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    print(f"exit={completed.returncode}", flush=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def synchronize_cuda() -> None:
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(idx)


def reset_cuda_peaks() -> None:
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)


def cuda_snapshot() -> list[dict]:
    if not torch.cuda.is_available():
        return []
    rows = []
    for idx in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(idx)
        rows.append(
            {
                "device": idx,
                "name": torch.cuda.get_device_properties(idx).name,
                "free_gib": free / 1024**3,
                "total_gib": total / 1024**3,
                "allocated_gib": torch.cuda.memory_allocated(idx) / 1024**3,
                "reserved_gib": torch.cuda.memory_reserved(idx) / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(idx) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(idx) / 1024**3,
            }
        )
    return rows


def print_cuda_snapshot(label: str) -> list[dict]:
    snapshot = cuda_snapshot()
    print(f"\nCUDA memory: {label}", flush=True)
    if not snapshot:
        print("  cuda unavailable", flush=True)
        return snapshot
    for row in snapshot:
        print(
            f"  cuda:{row['device']}: free={row['free_gib']:.2f} GiB "
            f"total={row['total_gib']:.2f} GiB "
            f"allocated={row['allocated_gib']:.2f} GiB "
            f"reserved={row['reserved_gib']:.2f} GiB "
            f"peak_allocated={row['peak_allocated_gib']:.2f} GiB "
            f"peak_reserved={row['peak_reserved_gib']:.2f} GiB",
            flush=True,
        )
    return snapshot


def first_parameter_device(module) -> str:
    for param in module.parameters():
        return str(param.device)
    for buffer in module.buffers():
        return str(buffer.device)
    return "no-params"


def force_pipeline_device_map(model: HookedTransformer) -> None:
    """Place whole TransformerLens modules on the devices used by forward()."""
    first_device = get_device_for_block_index(0, model.cfg)
    last_device = get_device_for_block_index(model.cfg.n_layers - 1, model.cfg)

    model.embed.to(first_device)
    model.hook_embed.to(first_device)
    if model.cfg.positional_embedding_type != "rotary":
        model.pos_embed.to(first_device)
        model.hook_pos_embed.to(first_device)

    for idx, block in enumerate(model.blocks):
        block.to(get_device_for_block_index(idx, model.cfg))

    if hasattr(model, "ln_final"):
        model.ln_final.to(last_device)
    model.unembed.to(last_device)


def print_block_devices(model: HookedTransformer) -> None:
    interesting = sorted(
        {
            0,
            1,
            model.cfg.n_layers // 2 - 1,
            model.cfg.n_layers // 2,
            model.cfg.n_layers - 2,
            model.cfg.n_layers - 1,
        }
    )
    print("module_device_summary:", flush=True)
    print(f"  embed={first_parameter_device(model.embed)}", flush=True)
    for idx in interesting:
        print(f"  block.{idx}={first_parameter_device(model.blocks[idx])}", flush=True)
    if hasattr(model, "ln_final"):
        print(f"  ln_final={first_parameter_device(model.ln_final)}", flush=True)
    print(f"  unembed={first_parameter_device(model.unembed)}", flush=True)


def read_rows(path: Path, *, height: int | None, limit: int, skip: int) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    matched = 0
    with path.open() as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if height is not None and row.get("height") != height:
                continue
            if matched < skip:
                matched += 1
                continue
            rows.append((row_index, row))
            if len(rows) >= limit:
                break
    return rows


def build_stage1_messages(system: str, user: str, model_name: str) -> list[dict[str, str]]:
    if "gemma" in model_name.lower():
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def encode_examples(rows: list[tuple[int, dict]], tokenizer, model_name: str) -> list[EncodedExample]:
    encoded_rows = []
    for row_index, row in rows:
        messages = build_stage1_messages(row["system_prompt"], row["prompt_text"], model_name)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        encoded_rows.append(
            EncodedExample(
                row_index=row_index,
                example_id=row.get("example_id"),
                height=row.get("height"),
                task=row.get("task"),
                token_ids=token_ids,
            )
        )
    return encoded_rows


def make_batch(examples: list[EncodedExample], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    max_len = max(len(example.token_ids) for example in examples)
    tokens = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    last_positions = []
    for idx, example in enumerate(examples):
        token_tensor = torch.tensor(example.token_ids, dtype=torch.long)
        tokens[idx, : token_tensor.numel()] = token_tensor
        attention_mask[idx, : token_tensor.numel()] = 1
        last_positions.append(token_tensor.numel() - 1)
    return tokens.to("cuda:0"), attention_mask.to("cuda:0"), last_positions


def benchmark_raw_batches(
    *,
    model: HookedTransformer,
    examples: list[EncodedExample],
    layer: int,
    batch_sizes: list[int],
) -> tuple[list[dict], torch.Tensor | None, list[dict]]:
    hook_name = f"blocks.{layer}.hook_resid_post"
    if hook_name not in model.hook_dict:
        raise ValueError(f"missing hook {hook_name}")

    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id or eos_token_id")

    token_lengths = [len(example.token_ids) for example in examples]
    sidecar_rows = [
        {
            "row_index": example.row_index,
            "example_id": example.example_id,
            "height": example.height,
            "task": example.task,
            "token_count": len(example.token_ids),
            "hook_name": hook_name,
        }
        for example in examples
    ]

    raw_results = []
    residual_cache: torch.Tensor | None = None

    for batch_size in batch_sizes:
        print(f"\nRaw extraction batch_size={batch_size}", flush=True)
        reset_cuda_peaks()
        torch.cuda.empty_cache()
        before = print_cuda_snapshot(f"raw batch_size={batch_size} before")
        started = time.time()
        status = "ok"
        error = None
        activations = []
        rows_done = 0
        batches_done = 0
        try:
            for start in range(0, len(examples), batch_size):
                chunk = examples[start : start + batch_size]
                tokens, attention_mask, last_positions = make_batch(chunk, pad_token_id)
                captured: dict[str, torch.Tensor] = {}

                def capture_last_position(act: torch.Tensor, hook) -> None:
                    positions = torch.tensor(last_positions, device=act.device)
                    batch_indices = torch.arange(len(last_positions), device=act.device)
                    captured["activation"] = act[batch_indices, positions, :].detach().to("cpu", dtype=torch.bfloat16)

                with torch.inference_mode():
                    model.run_with_hooks(
                        tokens,
                        return_type=None,
                        attention_mask=attention_mask,
                        fwd_hooks=[(hook_name, capture_last_position)],
                    )
                if "activation" not in captured:
                    raise RuntimeError("hook did not capture an activation")
                activation = captured["activation"]
                expected_shape = (len(chunk), model.cfg.d_model)
                if tuple(activation.shape) != expected_shape:
                    raise RuntimeError(f"unexpected activation shape {tuple(activation.shape)} != {expected_shape}")
                activations.append(activation)
                rows_done += len(chunk)
                batches_done += 1
                del tokens, attention_mask
            synchronize_cuda()
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            print(error, file=sys.stderr, flush=True)
            traceback.print_exc(limit=6)

        elapsed = time.time() - started
        stacked = None
        if status == "ok":
            stacked = torch.cat(activations, dim=0).contiguous()
            if residual_cache is None:
                residual_cache = stacked
        del activations
        torch.cuda.empty_cache()
        after = print_cuda_snapshot(f"raw batch_size={batch_size} after")
        result = {
            "batch_size": batch_size,
            "status": status,
            "error": error,
            "elapsed_seconds": elapsed,
            "rows_done": rows_done,
            "batches_done": batches_done,
            "tokens_total": sum(token_lengths[:rows_done]),
            "token_count_min": min(token_lengths),
            "token_count_max": max(token_lengths),
            "token_count_mean": sum(token_lengths) / len(token_lengths),
            "rows_per_second": rows_done / elapsed if elapsed > 0 else None,
            "tokens_per_second": sum(token_lengths[:rows_done]) / elapsed if elapsed > 0 else None,
            "memory_before": before,
            "memory_after": after,
            "memory_peak": cuda_snapshot(),
        }
        raw_results.append(result)
        print("RAW_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        if status != "ok":
            break
        del stacked

    return raw_results, residual_cache, sidecar_rows


def make_sae_input(residuals: torch.Tensor, batch_size: int, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    repeat_count = (batch_size + residuals.shape[0] - 1) // residuals.shape[0]
    tiled = residuals.repeat((repeat_count, 1))[:batch_size]
    return tiled.to(device=device, dtype=dtype, non_blocking=False)


def benchmark_sae_encoding(
    *,
    residuals: torch.Tensor,
    sae_release: str,
    sae_ids: list[str],
    sae_batch_sizes: list[int],
    sae_device: str,
    sae_dtype: str,
    top_k: int,
) -> list[dict]:
    from sae_lens import SAE

    dtype = getattr(torch, sae_dtype)
    sae_results = []
    for sae_id in sae_ids:
        print(f"\nSAE benchmark {sae_release} / {sae_id}", flush=True)
        reset_cuda_peaks()
        torch.cuda.empty_cache()
        before_load = print_cuda_snapshot(f"sae {sae_id} before load")
        started_load = time.time()
        try:
            sae, sae_cfg_dict, _ = SAE.from_pretrained_with_cfg_and_sparsity(
                release=sae_release,
                sae_id=sae_id,
                device=sae_device,
                dtype=sae_dtype,
            )
            sae.eval()
            synchronize_cuda()
            load_elapsed = time.time() - started_load
            after_load = print_cuda_snapshot(f"sae {sae_id} after load")
            sae_cfg = {
                "d_in": sae.cfg.d_in,
                "d_sae": sae.cfg.d_sae,
                "hook_name": sae_cfg_dict.get("hook_name"),
                "hf_hook_name": sae_cfg_dict.get("hf_hook_name"),
                "architecture": sae_cfg_dict.get("architecture"),
                "dtype": str(next(sae.parameters()).dtype),
                "device": str(next(sae.parameters()).device),
            }
        except Exception as exc:  # noqa: BLE001
            if "sae" in locals():
                del sae
            torch.cuda.empty_cache()
            result = {
                "sae_id": sae_id,
                "status": "load_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "load_elapsed_seconds": time.time() - started_load,
                "memory_before_load": before_load,
                "memory_after_load": cuda_snapshot(),
                "batch_results": [],
            }
            print("SAE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
            traceback.print_exc(limit=6)
            sae_results.append(result)
            continue

        batch_results = []
        for batch_size in sae_batch_sizes:
            print(f"SAE encode batch_size={batch_size}", flush=True)
            reset_cuda_peaks()
            torch.cuda.empty_cache()
            before_batch = print_cuda_snapshot(f"sae {sae_id} batch_size={batch_size} before")
            started_batch = time.time()
            status = "ok"
            error = None
            l0_mean = None
            try:
                sae_input = make_sae_input(residuals, batch_size, device=sae_device, dtype=dtype)
                with torch.inference_mode():
                    feature_acts = sae.encode(sae_input)
                    k = min(top_k, feature_acts.shape[-1])
                    top_values, top_indices = torch.topk(feature_acts, k=k, dim=-1)
                    l0_mean = float((feature_acts > 0).sum(dim=-1).float().mean().item())
                    checksum = float(top_values[:, 0].float().mean().item())
                synchronize_cuda()
                del sae_input, feature_acts, top_values, top_indices
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                checksum = None
                print(error, file=sys.stderr, flush=True)
                traceback.print_exc(limit=6)
            elapsed = time.time() - started_batch
            torch.cuda.empty_cache()
            after_batch = print_cuda_snapshot(f"sae {sae_id} batch_size={batch_size} after")
            batch_result = {
                "batch_size": batch_size,
                "status": status,
                "error": error,
                "elapsed_seconds": elapsed,
                "rows_per_second": batch_size / elapsed if elapsed > 0 else None,
                "l0_mean": l0_mean,
                "top1_mean_checksum": checksum,
                "memory_before": before_batch,
                "memory_after": after_batch,
                "memory_peak": cuda_snapshot(),
            }
            batch_results.append(batch_result)
            print("SAE_BATCH_RESULT " + json.dumps(batch_result, sort_keys=True), flush=True)
            if status != "ok":
                break

        result = {
            "sae_id": sae_id,
            "status": "ok",
            "error": None,
            "load_elapsed_seconds": load_elapsed,
            "sae_cfg": sae_cfg,
            "memory_before_load": before_load,
            "memory_after_load": after_load,
            "batch_results": batch_results,
        }
        print("SAE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        sae_results.append(result)
        del sae
        torch.cuda.empty_cache()

    return sae_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4])
    parser.add_argument("--sae-release", default="gemma-scope-2-27b-it-res-all")
    parser.add_argument("--sae-ids", default=None)
    parser.add_argument("--sae-batch-sizes", type=parse_int_list, default=[32, 128, 512])
    parser.add_argument("--sae-device", default="cuda:0")
    parser.add_argument("--sae-dtype", default="bfloat16")
    parser.add_argument("--sae-top-k", type=int, default=128)
    parser.add_argument("--skip-sae", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage2/pilots"))
    args = parser.parse_args()

    load_dotenv()
    torch.set_grad_enabled(False)

    sae_ids = (
        [item.strip() for item in args.sae_ids.split(",") if item.strip()]
        if args.sae_ids
        else [
            f"layer_{args.layer}_width_16k_l0_small",
            f"layer_{args.layer}_width_262k_l0_small",
        ]
    )

    print("Stage 2 J-node measurement", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"jsonl={args.jsonl}", flush=True)
    print(f"height={args.height}", flush=True)
    print(f"rows={args.rows}", flush=True)
    print(f"skip={args.skip}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"batch_sizes={args.batch_sizes}", flush=True)
    print(f"sae_release={args.sae_release}", flush=True)
    print(f"sae_ids={sae_ids}", flush=True)
    print(f"sae_batch_sizes={args.sae_batch_sizes}", flush=True)
    print(f"HF_HOME={os.environ.get('HF_HOME', '<unset>')}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"sae-lens={package_version('sae-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.n_devices:
        print("Not enough CUDA devices for requested n_devices.", file=sys.stderr, flush=True)
        return 2

    run(["nvidia-smi", "-L"])
    run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"])
    before_load = print_cuda_snapshot("before model load")

    started = time.time()
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        device="cuda",
        n_devices=args.n_devices,
        dtype=torch.bfloat16,
        n_ctx=args.n_ctx,
    )
    model.eval()
    if model.cfg.n_devices > 1:
        print("Applying deterministic pipeline device map for multi-GPU forward.", flush=True)
        force_pipeline_device_map(model)
        torch.cuda.empty_cache()
    model_load_elapsed = time.time() - started
    print(f"model_load_seconds={model_load_elapsed:.1f}", flush=True)
    print(f"cfg.n_layers={model.cfg.n_layers}", flush=True)
    print(f"cfg.d_model={model.cfg.d_model}", flush=True)
    print(f"cfg.n_devices={model.cfg.n_devices}", flush=True)
    print_block_devices(model)
    after_load = print_cuda_snapshot("after model load")

    rows = read_rows(args.jsonl, height=args.height, limit=args.rows, skip=args.skip)
    if len(rows) != args.rows:
        print(f"Requested {args.rows} rows but found {len(rows)}.", file=sys.stderr, flush=True)
        return 3
    examples = encode_examples(rows, model.tokenizer, args.model)
    print(
        "token_counts="
        + json.dumps(
            {
                "min": min(len(example.token_ids) for example in examples),
                "max": max(len(example.token_ids) for example in examples),
                "mean": sum(len(example.token_ids) for example in examples) / len(examples),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    raw_results, residuals, sidecar_rows = benchmark_raw_batches(
        model=model,
        examples=examples,
        layer=args.layer,
        batch_sizes=args.batch_sizes,
    )
    if residuals is None:
        print("No successful raw extraction; skipping SAE benchmark.", file=sys.stderr, flush=True)
        sae_results = []
    elif args.skip_sae:
        sae_results = []
    else:
        sae_results = benchmark_sae_encoding(
            residuals=residuals,
            sae_release=args.sae_release,
            sae_ids=sae_ids,
            sae_batch_sizes=args.sae_batch_sizes,
            sae_device=args.sae_device,
            sae_dtype=args.sae_dtype,
            top_k=args.sae_top_k,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.out_dir / f"j2_stage2_measure_L{args.layer}_h{args.height}_{timestamp}"
    if residuals is not None:
        save_file({"activations": residuals.contiguous()}, prefix.with_suffix(".safetensors"))
        with prefix.with_suffix(".example_ids.jsonl").open("w") as f:
            for row in sidecar_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    result = {
        "created_at_utc": timestamp,
        "model": args.model,
        "jsonl": str(args.jsonl),
        "height": args.height,
        "rows": args.rows,
        "skip": args.skip,
        "layer": args.layer,
        "n_devices": args.n_devices,
        "n_ctx": args.n_ctx,
        "transformer_lens_version": package_version("transformer-lens"),
        "sae_lens_version": package_version("sae-lens"),
        "torch_version": torch.__version__,
        "model_load_elapsed_seconds": model_load_elapsed,
        "memory_before_model_load": before_load,
        "memory_after_model_load": after_load,
        "raw_batch_sweep": raw_results,
        "sae_release": args.sae_release,
        "sae_benchmarks": sae_results,
        "residual_artifact": str(prefix.with_suffix(".safetensors")) if residuals is not None else None,
        "sidecar_artifact": str(prefix.with_suffix(".example_ids.jsonl")) if residuals is not None else None,
    }
    result_path = prefix.with_suffix(".json")
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"saved={result_path}", flush=True)
    if residuals is not None:
        print(f"saved={prefix.with_suffix('.safetensors')}", flush=True)
        print(f"saved={prefix.with_suffix('.example_ids.jsonl')}", flush=True)
    run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
