#!/usr/bin/env python3
"""Extract Stage 2 residual activations with Hugging Face Transformers.

This path is intended for Qwen/Qwen-Scope, whose public SAE examples hook
``model.model.layers[LAYER]`` directly. It writes the same artifact schema as
``scripts/stage2_extract.py`` so the existing raw-probe code can be reused.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import (  # noqa: E402
    encode_stage1_rows,
    make_padded_batch,
    parse_int_list,
    read_stage1_rows,
    sha256_file,
    slugify_model_name,
    tokenizer_pad_token_id,
    write_activation_outputs,
)
from src.env_loader import load_env  # noqa: E402
from src.stage2_paths import DEFAULT_ACTIVATION_SITE, normalize_activation_site  # noqa: E402


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def first_parameter_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def layer_device_summary(model: torch.nn.Module, layers: list[int]) -> dict[str, str]:
    summary: dict[str, str] = {}
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        summary["embed"] = str(first_parameter_device(model.get_input_embeddings()))
    model_layers = model.model.layers
    interesting = sorted(set([0, *layers, max(0, len(model_layers) - 1)]))
    for layer in interesting:
        if 0 <= layer < len(model_layers):
            summary[f"layer.{layer}"] = str(first_parameter_device(model_layers[layer]))
    return summary


HF_SITE_HOOK_TEMPLATES = {
    DEFAULT_ACTIVATION_SITE: "model.model.layers.{layer}.output",
    "mlp_in": "model.model.layers.{layer}.mlp.input",
    "mlp_in_weighted": "model.model.layers.{layer}.mlp.input",
    "mlp_out": "model.model.layers.{layer}.mlp.output",
}


def parse_site_list(value: str) -> list[str]:
    sites = [normalize_activation_site(part) for part in value.split(",") if part.strip()]
    if not sites:
        raise argparse.ArgumentTypeError("at least one activation site is required")
    deduped: list[str] = []
    for site in sites:
        if site not in deduped:
            deduped.append(site)
    return deduped


def hf_hook_template_for_site(site: str) -> str:
    normalized = normalize_activation_site(site)
    try:
        return HF_SITE_HOOK_TEMPLATES[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(HF_SITE_HOOK_TEMPLATES))
        raise ValueError(f"unsupported HF activation site {site!r}; expected one of: {known}") from exc


def _slice_last_positions(
    hidden: torch.Tensor,
    *,
    last_positions: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.tensor(last_positions, device=hidden.device)
    batch_indices = torch.arange(len(last_positions), device=hidden.device)
    return hidden[batch_indices, positions, :].detach().to("cpu", dtype=output_dtype)


def _capture_hf_site_output(
    *,
    site: str,
    layer: int,
    captured: dict[tuple[str, int], torch.Tensor],
    last_positions: list[int],
    output_dtype: torch.dtype,
):
    def hook(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured[(site, layer)] = _slice_last_positions(
            hidden,
            last_positions=last_positions,
            output_dtype=output_dtype,
        )

    return hook


def _capture_hf_site_input(
    *,
    site: str,
    layer: int,
    captured: dict[tuple[str, int], torch.Tensor],
    last_positions: list[int],
    output_dtype: torch.dtype,
):
    def hook(_module, inputs) -> None:
        if not inputs:
            raise RuntimeError(f"site {site} layer {layer} pre-hook received no inputs")
        hidden = inputs[0]
        captured[(site, layer)] = _slice_last_positions(
            hidden,
            last_positions=last_positions,
            output_dtype=output_dtype,
        )

    return hook


def _register_hf_site_hook(
    *,
    model: torch.nn.Module,
    site: str,
    layer: int,
    captured: dict[tuple[str, int], torch.Tensor],
    last_positions: list[int],
    output_dtype: torch.dtype,
):
    if site == DEFAULT_ACTIVATION_SITE:
        return model.model.layers[layer].register_forward_hook(
            _capture_hf_site_output(
                site=site,
                layer=layer,
                captured=captured,
                last_positions=last_positions,
                output_dtype=output_dtype,
            )
        )
    if site in {"mlp_in", "mlp_in_weighted"}:
        return model.model.layers[layer].mlp.register_forward_pre_hook(
            _capture_hf_site_input(
                site=site,
                layer=layer,
                captured=captured,
                last_positions=last_positions,
                output_dtype=output_dtype,
            )
        )
    if site == "mlp_out":
        return model.model.layers[layer].mlp.register_forward_hook(
            _capture_hf_site_output(
                site=site,
                layer=layer,
                captured=captured,
                last_positions=last_positions,
                output_dtype=output_dtype,
            )
        )
    hf_hook_template_for_site(site)
    raise AssertionError(f"unreachable activation site {site!r}")


def validate_hf_layers(model: torch.nn.Module, layers: list[int]) -> None:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("expected model.model.layers on this Hugging Face causal LM")
    n_layers = len(model.model.layers)
    missing = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if missing:
        raise ValueError(f"layers {missing} outside available range 0..{n_layers - 1}")


def extract_hf_activations(
    model: torch.nn.Module,
    examples,
    *,
    layers: list[int],
    activation_sites: list[str],
    batch_size: int,
    output_dtype: torch.dtype,
) -> tuple[dict[str, dict[int, torch.Tensor]], list[dict[str, Any]], dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("no examples to extract")

    validate_hf_layers(model, layers)
    sites = [normalize_activation_site(site) for site in activation_sites]
    for site in sites:
        hf_hook_template_for_site(site)
    pad_token_id = tokenizer_pad_token_id(model.tokenizer)
    input_device = first_parameter_device(model.get_input_embeddings())
    hidden_size = int(model.config.hidden_size)
    token_counts = [example.token_count for example in examples]
    by_site_layer: dict[str, dict[int, list[torch.Tensor]]] = {
        site: {layer: [] for layer in layers} for site in sites
    }
    rows_done = 0
    started = time.monotonic()

    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            tokens, attention_mask, last_positions = make_padded_batch(
                chunk,
                pad_token_id=pad_token_id,
                device=input_device,
            )
            captured: dict[tuple[str, int], torch.Tensor] = {}
            handles = [
                _register_hf_site_hook(
                    model=model,
                    site=site,
                    layer=layer,
                    captured=captured,
                    last_positions=last_positions,
                    output_dtype=output_dtype,
                )
                for layer in layers
                for site in sites
            ]
            try:
                model(input_ids=tokens, attention_mask=attention_mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            for site in sites:
                for layer in layers:
                    activation = captured.get((site, layer))
                    if activation is None:
                        raise RuntimeError(f"site {site} layer {layer} hook did not capture")
                    expected = (len(chunk), hidden_size)
                    if tuple(activation.shape) != expected:
                        raise RuntimeError(
                            f"site {site} layer {layer} activation shape {tuple(activation.shape)} != {expected}"
                        )
                    by_site_layer[site][layer].append(activation.contiguous())
            rows_done += len(chunk)
            del tokens, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"extracted rows {rows_done}/{len(examples)}", flush=True)

    activations = {
        site: {layer: torch.cat(chunks, dim=0).contiguous() for layer, chunks in by_layer.items()}
        for site, by_layer in by_site_layer.items()
    }
    elapsed = time.monotonic() - started
    sidecar_rows = [
        {
            "row_index": example.row_index,
            "example_id": example.example_id,
            "height": example.height,
            "task": example.task,
            "model": example.model,
            "is_correct_strong": example.is_correct_strong,
            "parse_failed": example.parse_failed,
            "token_count": example.token_count,
            "last_token_position": example.token_count - 1,
        }
        for example in examples
    ]
    stats = {
        "elapsed_seconds": elapsed,
        "rows_done": rows_done,
        "rows_per_second": rows_done / elapsed if elapsed > 0 else None,
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_count_mean": sum(token_counts) / len(token_counts),
    }
    return activations, sidecar_rows, stats


def extract_hf_layer_activations(
    model: torch.nn.Module,
    examples,
    *,
    layers: list[int],
    batch_size: int,
    output_dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    activations_by_site, sidecar_rows, stats = extract_hf_activations(
        model,
        examples,
        layers=layers,
        activation_sites=[DEFAULT_ACTIVATION_SITE],
        batch_size=batch_size,
        output_dtype=output_dtype,
    )
    return activations_by_site[DEFAULT_ACTIVATION_SITE], sidecar_rows, stats


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--layers", required=True, type=parse_int_list)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-site", default=DEFAULT_ACTIVATION_SITE)
    parser.add_argument("--activation-sites", type=parse_site_list, default=None, help="Comma-separated HF sites to capture in one pass.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto", help='Transformers device_map, or "none" for a single device.')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=False to chat templates that support it.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--drop-parse-failed", action="store_true")
    args = parser.parse_args()

    rows = read_stage1_rows(
        args.jsonl,
        height=args.height,
        limit=args.limit,
        skip=args.skip,
        drop_parse_failed=args.drop_parse_failed,
    )
    if not rows:
        raise ValueError(f"no rows matched {args.jsonl}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    dtype = torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device_map == "none":
        model.to(args.device)
    model.eval()
    model.tokenizer = tokenizer

    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else None
    examples = encode_stage1_rows(
        rows,
        tokenizer=tokenizer,
        model_name=args.model,
        chat_template_kwargs=chat_template_kwargs,
    )
    too_long = [example.token_count for example in examples if example.token_count > args.n_ctx]
    if too_long:
        raise ValueError(f"{len(too_long)} prompts exceed --n-ctx={args.n_ctx}; max token count is {max(too_long)}")
    observed_tasks = sorted({example.task for example in examples if example.task is not None})
    task = args.task
    if task is None:
        if len(observed_tasks) != 1:
            raise ValueError(f"could not infer one task from rows: {observed_tasks}")
        task = observed_tasks[0]
    model_key = args.model_key or slugify_model_name(args.model)

    activation_sites = args.activation_sites or [normalize_activation_site(args.activation_site)]
    activations_by_site, sidecar_rows, stats = extract_hf_activations(
        model,
        examples,
        layers=args.layers,
        activation_sites=activation_sites,
        batch_size=args.batch_size,
        output_dtype=torch_dtype(args.output_dtype),
    )
    base_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_path": str(args.jsonl),
        "jsonl_sha256": sha256_file(args.jsonl),
        "model_name": args.model,
        "model_key": model_key,
        "task": task,
        "layers": args.layers,
        "activation_sites": activation_sites,
        "batch_size": args.batch_size,
        "n_ctx": args.n_ctx,
        "backend": "hf_transformers",
        "device_map": args.device_map,
        "chat_template_kwargs": chat_template_kwargs or {},
        "dtype": str(dtype),
        "output_dtype": str(torch_dtype(args.output_dtype)),
        "row_count": len(examples),
        "drop_parse_failed": args.drop_parse_failed,
        "height": args.height,
        "skip": args.skip,
        "limit": args.limit,
        "transformers_version": transformers.__version__,
        "hf_n_layers": len(model.model.layers),
        "hf_hidden_size": int(model.config.hidden_size),
        "hf_device_map": getattr(model, "hf_device_map", None),
        "module_devices": layer_device_summary(model, args.layers),
        "extraction_stats": stats,
    }
    for site in activation_sites:
        hook_template = hf_hook_template_for_site(site)
        metadata = {
            **base_metadata,
            "activation_site": site,
            "hook_template": hook_template,
        }
        written = write_activation_outputs(
            activations_by_site[site],
            sidecar_rows,
            out_dir=args.out_dir,
            model_key=model_key,
            task=task,
            metadata=metadata,
            activation_site=site,
            hook_template=hook_template,
        )
        for written_path in written:
            print(written_path)


if __name__ == "__main__":
    main()
