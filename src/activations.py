"""Stage 2 residual-stream activation extraction utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .messages import render_chat_text
from .stage2_paths import (
    DEFAULT_ACTIVATION_SITE,
    DEFAULT_HOOK_TEMPLATE,
    activation_stem,
    hook_name_for_layer,
    normalize_activation_site,
)


@dataclass(frozen=True)
class EncodedExample:
    row_index: int
    example_id: str | None
    height: int | None
    task: str | None
    model: str | None
    is_correct_strong: bool | None
    parse_failed: bool | None
    token_ids: list[int]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list for CLI layer arguments."""
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"expected comma-separated integers, got {value!r}") from exc
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def slugify_model_name(model_name: str) -> str:
    """Return the Stage 2 filename slug for a HF model or Stage 1 model key."""
    lowered = model_name.lower().replace("google/", "")
    if lowered.startswith("gemma-3-") and lowered.endswith("-it"):
        lowered = "gemma3-" + lowered.removeprefix("gemma-3-").removesuffix("-it")
    return lowered.replace("-", "_").replace("/", "_")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stage1_rows(
    path: Path,
    *,
    height: int | None = None,
    limit: int | None = None,
    skip: int = 0,
    drop_parse_failed: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    """Read Stage 1 JSONL rows with optional filtering and deterministic order."""
    rows: list[tuple[int, dict[str, Any]]] = []
    matched = 0
    with path.open() as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if height is not None and row.get("height") != height:
                continue
            if drop_parse_failed and row.get("parse_failed"):
                continue
            if matched < skip:
                matched += 1
                continue
            rows.append((row_index, row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode_stage1_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    tokenizer,
    model_name: str,
) -> list[EncodedExample]:
    """Apply the Stage 1 chat prompt reconstruction and tokenize each row."""
    encoded: list[EncodedExample] = []
    for row_index, row in rows:
        text = render_chat_text(
            tokenizer,
            system=row["system_prompt"],
            user=row["prompt_text"],
            model_name=model_name,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"row {row_index} produced no tokens")
        encoded.append(
            EncodedExample(
                row_index=row_index,
                example_id=row.get("example_id"),
                height=row.get("height"),
                task=row.get("task"),
                model=row.get("model"),
                is_correct_strong=row.get("is_correct_strong"),
                parse_failed=row.get("parse_failed"),
                token_ids=list(token_ids),
            )
        )
    return encoded


def make_padded_batch(
    examples: list[EncodedExample],
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Right-pad token IDs and return last-token positions before padding."""
    if not examples:
        raise ValueError("cannot batch zero examples")
    max_len = max(example.token_count for example in examples)
    tokens = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    last_positions: list[int] = []
    for idx, example in enumerate(examples):
        token_tensor = torch.tensor(example.token_ids, dtype=torch.long)
        length = token_tensor.numel()
        tokens[idx, :length] = token_tensor
        attention_mask[idx, :length] = 1
        last_positions.append(length - 1)
    return tokens.to(device), attention_mask.to(device), last_positions


def first_parameter_device(module) -> str:
    for param in module.parameters():
        return str(param.device)
    for buffer in module.buffers():
        return str(buffer.device)
    return "no-params"


def force_pipeline_device_map(model) -> None:
    """Place whole TransformerLens modules on devices used by multi-GPU forward."""
    from transformer_lens.utilities.multi_gpu import get_device_for_block_index

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


def module_device_summary(model) -> dict[str, str]:
    interesting = sorted(
        {
            0,
            1,
            max(0, model.cfg.n_layers // 2 - 1),
            model.cfg.n_layers // 2,
            max(0, model.cfg.n_layers - 2),
            model.cfg.n_layers - 1,
        }
    )
    summary = {"embed": first_parameter_device(model.embed)}
    for idx in interesting:
        summary[f"block.{idx}"] = first_parameter_device(model.blocks[idx])
    if hasattr(model, "ln_final"):
        summary["ln_final"] = first_parameter_device(model.ln_final)
    summary["unembed"] = first_parameter_device(model.unembed)
    return summary


def load_tl_model(
    model_name: str,
    *,
    n_devices: int = 1,
    n_ctx: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
    load_mode: str = "no-processing",
):
    """Load a TransformerLens model with the J-node-tested defaults."""
    from transformer_lens import HookedTransformer

    kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_devices": n_devices,
        "dtype": dtype,
        "n_ctx": n_ctx,
    }
    if load_mode == "no-processing":
        model = HookedTransformer.from_pretrained_no_processing(model_name, **kwargs)
    elif load_mode == "default":
        model = HookedTransformer.from_pretrained(model_name, **kwargs)
    else:
        raise ValueError(f"unknown load_mode {load_mode!r}")
    model.eval()
    if getattr(model.cfg, "n_devices", 1) > 1:
        force_pipeline_device_map(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model


def input_device_for_model(model) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device(getattr(model.cfg, "device", "cpu"))


def tokenizer_pad_token_id(tokenizer) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id or eos_token_id")
    return int(pad_token_id)


def validate_hooks(
    model,
    layers: list[int],
    *,
    hook_template: str = DEFAULT_HOOK_TEMPLATE,
) -> list[str]:
    hooks = [hook_name_for_layer(layer=layer, hook_template=hook_template) for layer in layers]
    missing = [hook for hook in hooks if hook not in model.hook_dict]
    if missing:
        sample = sorted(model.hook_dict.keys())[:20]
        raise ValueError(f"missing hooks {missing}; first available hooks: {sample}")
    return hooks


def _capture_hook(
    *,
    layer: int,
    captured: dict[int, torch.Tensor],
    last_positions: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
):
    def capture_last_position(act: torch.Tensor, hook) -> None:  # noqa: ARG001
        positions = torch.tensor(last_positions, device=act.device)
        batch_indices = torch.arange(len(last_positions), device=act.device)
        captured[layer] = act[batch_indices, positions, :].detach().to("cpu", dtype=output_dtype)

    return capture_last_position


def extract_residual_activations(
    model,
    examples: list[EncodedExample],
    *,
    layers: list[int],
    batch_size: int,
    hook_template: str = DEFAULT_HOOK_TEMPLATE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Extract last-pre-CoT activations for all requested layers."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("no examples to extract")

    hooks = validate_hooks(model, layers, hook_template=hook_template)
    pad_token_id = tokenizer_pad_token_id(model.tokenizer)
    input_device = input_device_for_model(model)
    token_counts = [example.token_count for example in examples]

    by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
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
            captured: dict[int, torch.Tensor] = {}
            fwd_hooks = [
                (
                    hook_name,
                    _capture_hook(
                        layer=layer,
                        captured=captured,
                        last_positions=last_positions,
                        output_dtype=output_dtype,
                    ),
                )
                for layer, hook_name in zip(layers, hooks, strict=True)
            ]
            model.run_with_hooks(
                tokens,
                return_type=None,
                attention_mask=attention_mask,
                fwd_hooks=fwd_hooks,
            )
            for layer in layers:
                activation = captured.get(layer)
                if activation is None:
                    raise RuntimeError(f"hook {hook_name_for_layer(layer=layer, hook_template=hook_template)} did not capture")
                expected = (len(chunk), model.cfg.d_model)
                if tuple(activation.shape) != expected:
                    raise RuntimeError(
                        f"layer {layer} activation shape {tuple(activation.shape)} != {expected}"
                    )
                by_layer[layer].append(activation.contiguous())
            rows_done += len(chunk)
            del tokens, attention_mask

    activations = {layer: torch.cat(chunks, dim=0).contiguous() for layer, chunks in by_layer.items()}
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


def write_activation_outputs(
    activations_by_layer: dict[int, torch.Tensor],
    sidecar_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    model_key: str,
    task: str,
    metadata: dict[str, Any],
    activation_site: str = DEFAULT_ACTIVATION_SITE,
    hook_template: str = DEFAULT_HOOK_TEMPLATE,
) -> list[Path]:
    """Write one safetensors file plus sidecars per layer."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    normalized_site = normalize_activation_site(activation_site)
    for layer, activations in activations_by_layer.items():
        prefix = out_dir / activation_stem(
            model_key=model_key,
            task=task,
            layer=layer,
            activation_site=normalized_site,
        )
        hook_name = hook_name_for_layer(layer=layer, hook_template=hook_template)
        save_file({"activations": activations.contiguous()}, prefix.with_suffix(".safetensors"))
        with prefix.with_suffix(".example_ids.jsonl").open("w") as f:
            for sidecar in sidecar_rows:
                row = dict(sidecar)
                row["hook_name"] = hook_name
                f.write(json.dumps(row, sort_keys=True) + "\n")
        meta = {
            **metadata,
            "layer": layer,
            "activation_site": normalized_site,
            "hook_name": hook_name,
            "hook_template": hook_template,
            "shape": list(activations.shape),
            "dtype": str(activations.dtype),
            "activation_file": str(prefix.with_suffix(".safetensors")),
            "sidecar_file": str(prefix.with_suffix(".example_ids.jsonl")),
        }
        with prefix.with_suffix(".meta.json").open("w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write("\n")
        written.extend(
            [
                prefix.with_suffix(".safetensors"),
                prefix.with_suffix(".example_ids.jsonl"),
                prefix.with_suffix(".meta.json"),
            ]
        )
    return written


def run_extraction(
    *,
    jsonl_path: Path,
    model_name: str,
    layers: list[int],
    batch_size: int,
    n_devices: int,
    n_ctx: int,
    out_dir: Path,
    model_key: str | None = None,
    task: str | None = None,
    height: int | None = None,
    limit: int | None = None,
    skip: int = 0,
    drop_parse_failed: bool = False,
    load_mode: str = "no-processing",
    activation_site: str = DEFAULT_ACTIVATION_SITE,
    hook_template: str = DEFAULT_HOOK_TEMPLATE,
    dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype = torch.bfloat16,
) -> list[Path]:
    """Load the TL model, extract activations, and write Stage 2 artifacts."""
    rows = read_stage1_rows(
        jsonl_path,
        height=height,
        limit=limit,
        skip=skip,
        drop_parse_failed=drop_parse_failed,
    )
    if not rows:
        raise ValueError(f"no rows matched {jsonl_path}")

    model = load_tl_model(
        model_name,
        n_devices=n_devices,
        n_ctx=n_ctx,
        dtype=dtype,
        load_mode=load_mode,
    )
    examples = encode_stage1_rows(rows, tokenizer=model.tokenizer, model_name=model_name)
    observed_tasks = sorted({example.task for example in examples if example.task is not None})
    if task is None:
        if len(observed_tasks) != 1:
            raise ValueError(f"could not infer one task from rows: {observed_tasks}")
        task = observed_tasks[0]
    if model_key is None:
        model_key = slugify_model_name(model_name)

    activations, sidecar_rows, stats = extract_residual_activations(
        model,
        examples,
        layers=layers,
        batch_size=batch_size,
        hook_template=hook_template,
        output_dtype=output_dtype,
    )
    normalized_site = normalize_activation_site(activation_site)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": sha256_file(jsonl_path),
        "model_name": model_name,
        "model_key": model_key,
        "task": task,
        "layers": layers,
        "activation_site": normalized_site,
        "hook_template": hook_template,
        "batch_size": batch_size,
        "n_devices": n_devices,
        "n_ctx": n_ctx,
        "load_mode": load_mode,
        "dtype": str(dtype),
        "output_dtype": str(output_dtype),
        "row_count": len(examples),
        "drop_parse_failed": drop_parse_failed,
        "height": height,
        "skip": skip,
        "limit": limit,
        "transformerlens_n_layers": model.cfg.n_layers,
        "transformerlens_d_model": model.cfg.d_model,
        "transformerlens_n_devices": getattr(model.cfg, "n_devices", None),
        "module_devices": module_device_summary(model),
        "extraction_stats": stats,
    }
    return write_activation_outputs(
        activations,
        sidecar_rows,
        out_dir=out_dir,
        model_key=model_key,
        task=task,
        metadata=metadata,
        activation_site=normalized_site,
        hook_template=hook_template,
    )
