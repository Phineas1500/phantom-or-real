#!/usr/bin/env python3
"""Pilot TransformerLens activation extraction for Gemma 3 27B on Scholar J nodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from dotenv import load_dotenv
from safetensors.torch import save_file
from transformer_lens import HookedTransformer
from transformer_lens.utilities.multi_gpu import get_device_for_block_index


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    print(f"exit={completed.returncode}", flush=True)


def print_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: cuda unavailable", flush=True)
        return
    print(f"\nCUDA memory: {label}", flush=True)
    for idx in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(idx)
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        print(
            f"  cuda:{idx}: free={free / 1024**3:.2f} GiB "
            f"total={total / 1024**3:.2f} GiB "
            f"allocated={allocated / 1024**3:.2f} GiB "
            f"reserved={reserved / 1024**3:.2f} GiB",
            flush=True,
        )


def read_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def first_parameter_device(module) -> str:
    for param in module.parameters():
        return str(param.device)
    for buffer in module.buffers():
        return str(buffer.device)
    return "no-params"


def force_pipeline_device_map(model: HookedTransformer) -> None:
    """Place whole TransformerLens modules on the devices used by forward().

    TransformerLens 3.0's module mover uses a "best available GPU" heuristic,
    while HookedTransformer.forward still moves the residual stream with
    get_device_for_block_index(). On two equal A40s, the heuristic can split
    submodules in a block across devices. Re-placing full blocks by block index
    keeps each block internally coherent.
    """
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
    interesting = sorted({0, 1, model.cfg.n_layers // 2 - 1, model.cfg.n_layers // 2, model.cfg.n_layers - 2, model.cfg.n_layers - 1})
    print("module_device_summary:", flush=True)
    print(f"  embed={first_parameter_device(model.embed)}", flush=True)
    for idx in interesting:
        print(f"  block.{idx}={first_parameter_device(model.blocks[idx])}", flush=True)
    if hasattr(model, "ln_final"):
        print(f"  ln_final={first_parameter_device(model.ln_final)}", flush=True)
    print(f"  unembed={first_parameter_device(model.unembed)}", flush=True)


def gemma_chat_text(row: dict, tokenizer) -> str:
    content = f"{row['system_prompt']}\n\n{row['prompt_text']}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument(
        "--load-mode",
        choices=("no-processing", "default"),
        default="no-processing",
        help="Use no-processing by default to reduce CPU RAM during bf16 loads.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage2/pilots"))
    args = parser.parse_args()

    load_dotenv()
    torch.set_grad_enabled(False)

    print("Stage 2 TransformerLens 27B J-node pilot", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"jsonl={args.jsonl}", flush=True)
    print(f"rows={args.rows}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"n_devices={args.n_devices}", flush=True)
    print(f"n_ctx={args.n_ctx}", flush=True)
    print(f"load_mode={args.load_mode}", flush=True)
    print(f"HF_HOME={os.environ.get('HF_HOME', '<unset>')}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"sae-lens={package_version('sae-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"cuda:{idx} {props.name} {props.total_memory / 1024**3:.2f} GiB", flush=True)

    run(["nvidia-smi", "-L"])
    run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"])
    print_cuda_memory("before load")

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.n_devices:
        print("Not enough CUDA devices for requested n_devices.", file=sys.stderr, flush=True)
        return 2

    started = time.time()
    load_kwargs = {
        "device": "cuda",
        "n_devices": args.n_devices,
        "dtype": torch.bfloat16,
        "n_ctx": args.n_ctx,
    }
    if args.load_mode == "no-processing":
        model = HookedTransformer.from_pretrained_no_processing(args.model, **load_kwargs)
    else:
        model = HookedTransformer.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"model_load_seconds={time.time() - started:.1f}", flush=True)
    print(f"cfg.n_layers={model.cfg.n_layers}", flush=True)
    print(f"cfg.d_model={model.cfg.d_model}", flush=True)
    print(f"cfg.n_devices={model.cfg.n_devices}", flush=True)
    print(f"cfg.device={model.cfg.device}", flush=True)
    print_block_devices(model)
    if model.cfg.n_devices > 1:
        print("Applying deterministic pipeline device map for multi-GPU forward.", flush=True)
        force_pipeline_device_map(model)
        torch.cuda.empty_cache()
        print_block_devices(model)
    print_cuda_memory("after load")
    run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"])

    hook_name = f"blocks.{args.layer}.hook_resid_post"
    hook_names = sorted(model.hook_dict.keys())
    resid_post_hooks = [name for name in hook_names if "resid_post" in name]
    print(f"hook_count={len(hook_names)}", flush=True)
    print(f"resid_post_hook_count={len(resid_post_hooks)}", flush=True)
    print("first_resid_post_hooks=" + ", ".join(resid_post_hooks[:8]), flush=True)
    if hook_name not in model.hook_dict:
        print(f"Missing hook {hook_name}", file=sys.stderr, flush=True)
        print("Available resid_post hooks:", file=sys.stderr, flush=True)
        for name in resid_post_hooks[:80]:
            print(f"  {name}", file=sys.stderr, flush=True)
        return 3
    print(f"using_hook={hook_name}", flush=True)

    rows = read_rows(args.jsonl, args.rows)
    if len(rows) != args.rows:
        print(f"Requested {args.rows} rows but found {len(rows)}.", file=sys.stderr, flush=True)
        return 4

    args.out_dir.mkdir(parents=True, exist_ok=True)
    activations: list[torch.Tensor] = []
    sidecar_rows: list[dict] = []

    for row_idx, row in enumerate(rows):
        text = gemma_chat_text(row, model.tokenizer)
        encoded = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        tokens = encoded["input_ids"].to("cuda:0")
        token_count = int(tokens.shape[-1])
        captured: dict[str, torch.Tensor] = {}

        def capture_last_position(act: torch.Tensor, hook) -> None:
            captured["activation"] = act[:, -1, :].detach().to("cpu", dtype=torch.bfloat16)

        with torch.inference_mode():
            model.run_with_hooks(
                tokens,
                return_type=None,
                fwd_hooks=[(hook_name, capture_last_position)],
            )

        if "activation" not in captured:
            print(f"No activation captured for row {row_idx}", file=sys.stderr, flush=True)
            return 5
        activation = captured["activation"]
        if activation.shape != (1, model.cfg.d_model):
            print(f"Unexpected activation shape {tuple(activation.shape)}", file=sys.stderr, flush=True)
            return 6
        activations.append(activation)
        sidecar_rows.append(
            {
                "row_index": row_idx,
                "example_id": row.get("example_id"),
                "height": row.get("height"),
                "task": row.get("task"),
                "token_count": token_count,
                "hook_name": hook_name,
            }
        )
        print(
            f"captured row={row_idx} example_id={row.get('example_id')} "
            f"tokens={token_count} activation_shape={tuple(activation.shape)}",
            flush=True,
        )
        print_cuda_memory(f"after row {row_idx}")

    stacked = torch.cat(activations, dim=0).contiguous()
    out_prefix = args.out_dir / f"gemma3_27b_tl_j2_layer{args.layer}_rows{args.rows}"
    save_file({"activations": stacked}, out_prefix.with_suffix(".safetensors"))
    with out_prefix.with_suffix(".example_ids.jsonl").open("w") as f:
        for item in sidecar_rows:
            f.write(json.dumps(item, sort_keys=True) + "\n")

    print(f"saved={out_prefix.with_suffix('.safetensors')}", flush=True)
    print(f"saved={out_prefix.with_suffix('.example_ids.jsonl')}", flush=True)
    print(f"final_activation_shape={tuple(stacked.shape)} dtype={stacked.dtype}", flush=True)
    print_cuda_memory("final")
    run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
