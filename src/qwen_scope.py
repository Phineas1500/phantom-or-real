"""Qwen-Scope SAE helpers.

Qwen-Scope publishes one PyTorch dict per layer. Each dict stores a TopK SAE
with encoder weights shaped (d_sae, d_model) and decoder weights shaped
(d_model, d_sae), which differs from the SAE Lens objects used by Gemma Scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch


_L0_RE = re.compile(r"(?:^|[-_])L0[_-]?(\d+)(?:$|[-_])", re.IGNORECASE)


def qwen_scope_filename(layer: int) -> str:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    return f"layer{layer}.sae.pt"


def infer_qwen_scope_top_k(repo_id: str) -> int | None:
    match = _L0_RE.search(repo_id)
    if match is None:
        return None
    return int(match.group(1))


def qwen_scope_sae_id(repo_id: str) -> str:
    name = repo_id.rstrip("/").rsplit("/", 1)[-1]
    if name.startswith("SAE-Res-"):
        name = name.removeprefix("SAE-Res-")
    slug = name.lower().replace("qwen3.5", "qwen35")
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return f"qwenscope_{slug}"


@dataclass
class QwenScopeSAE:
    """In-memory Qwen-Scope SAE tensors for one layer."""

    W_enc: torch.Tensor | None
    W_dec: torch.Tensor | None
    b_enc: torch.Tensor | None
    b_dec: torch.Tensor | None
    repo_id: str
    layer: int
    checkpoint_path: Path
    top_k: int | None = None

    @property
    def d_model(self) -> int:
        if self.W_enc is not None:
            return int(self.W_enc.shape[1])
        if self.W_dec is not None:
            return int(self.W_dec.shape[0])
        raise ValueError("SAE has neither encoder nor decoder tensors")

    @property
    def d_sae(self) -> int:
        if self.W_enc is not None:
            return int(self.W_enc.shape[0])
        if self.W_dec is not None:
            return int(self.W_dec.shape[1])
        raise ValueError("SAE has neither encoder nor decoder tensors")

    @property
    def device(self) -> torch.device:
        tensor = self.W_enc if self.W_enc is not None else self.W_dec
        if tensor is None:
            raise ValueError("SAE has no tensors")
        return tensor.device


def _validate_qwen_scope_state(state: dict[str, torch.Tensor], *, path: Path) -> None:
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"{path} is missing Qwen-Scope tensors: {missing}")

    W_enc = state["W_enc"]
    W_dec = state["W_dec"]
    b_enc = state["b_enc"]
    b_dec = state["b_dec"]
    if W_enc.ndim != 2 or W_dec.ndim != 2:
        raise ValueError(f"{path} expected rank-2 W_enc/W_dec, got {W_enc.shape} and {W_dec.shape}")
    d_sae, d_model = W_enc.shape
    if tuple(W_dec.shape) != (d_model, d_sae):
        raise ValueError(f"{path} W_dec shape {tuple(W_dec.shape)} != {(d_model, d_sae)}")
    if tuple(b_enc.shape) != (d_sae,):
        raise ValueError(f"{path} b_enc shape {tuple(b_enc.shape)} != {(d_sae,)}")
    if tuple(b_dec.shape) != (d_model,):
        raise ValueError(f"{path} b_dec shape {tuple(b_dec.shape)} != {(d_model,)}")


def load_qwen_scope_sae(
    path: Path,
    *,
    repo_id: str,
    layer: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    load_encoder: bool = True,
    load_decoder: bool = True,
    top_k: int | None = None,
) -> QwenScopeSAE:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path} did not load as a tensor dict")
    _validate_qwen_scope_state(state, path=path)

    def maybe_tensor(name: str, keep: bool) -> torch.Tensor | None:
        if not keep:
            return None
        tensor = state[name]
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor.to(device=device)

    return QwenScopeSAE(
        W_enc=maybe_tensor("W_enc", load_encoder),
        W_dec=maybe_tensor("W_dec", load_decoder),
        b_enc=maybe_tensor("b_enc", load_encoder),
        b_dec=maybe_tensor("b_dec", load_decoder),
        repo_id=repo_id,
        layer=layer,
        checkpoint_path=path,
        top_k=top_k if top_k is not None else infer_qwen_scope_top_k(repo_id),
    )


def qwen_scope_sae_summary(sae: QwenScopeSAE) -> dict[str, object]:
    dtype_tensor = sae.W_enc if sae.W_enc is not None else sae.W_dec
    return {
        "architecture": "topk_sae",
        "format": "qwen_scope_pt",
        "d_in": sae.d_model,
        "d_sae": sae.d_sae,
        "d_out": sae.d_model,
        "device": str(sae.device),
        "dtype": str(dtype_tensor.dtype) if dtype_tensor is not None else None,
        "hook_name": f"model.model.layers.{sae.layer}.output",
        "hf_hook_name": f"model.model.layers.{sae.layer}.output",
        "top_k": sae.top_k,
        "repo_id": sae.repo_id,
        "layer": sae.layer,
    }


def encode_qwen_scope_topk(
    sae: QwenScopeSAE,
    residual: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sae.W_enc is None or sae.b_enc is None:
        raise ValueError("QwenScopeSAE was loaded without encoder tensors")
    if residual.shape[-1] != sae.d_model:
        raise ValueError(f"residual dim {residual.shape[-1]} != SAE d_model {sae.d_model}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(top_k, sae.d_sae)
    x = residual.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
    pre_acts = x @ sae.W_enc.T + sae.b_enc
    values, indices = torch.topk(pre_acts, k=k, dim=-1)
    return values, indices.to(torch.int64)


def decode_qwen_scope_topk(
    sae: QwenScopeSAE,
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if sae.W_dec is None or sae.b_dec is None:
        raise ValueError("QwenScopeSAE was loaded without decoder tensors")
    if top_indices.shape != top_values.shape:
        raise ValueError(f"top_indices shape {tuple(top_indices.shape)} != top_values shape {tuple(top_values.shape)}")
    if top_indices.ndim != 2:
        raise ValueError(f"expected rank-2 top-k tensors, got {tuple(top_indices.shape)}")

    target_dtype = dtype or sae.W_dec.dtype
    device = sae.W_dec.device
    indices = top_indices.to(device=device, dtype=torch.long)
    values = top_values.to(device=device, dtype=accumulation_dtype)
    decoder_rows = sae.W_dec.T
    out = (
        sae.b_dec.to(device=device, dtype=accumulation_dtype)
        .unsqueeze(0)
        .expand(indices.shape[0], -1)
        .clone()
    )
    for rank in range(indices.shape[1]):
        rank_values = values[:, rank]
        if torch.count_nonzero(rank_values).item() == 0:
            continue
        selected = decoder_rows.index_select(0, indices[:, rank]).to(dtype=accumulation_dtype)
        out.add_(selected * rank_values.unsqueeze(1))
    return out.to(dtype=target_dtype)
