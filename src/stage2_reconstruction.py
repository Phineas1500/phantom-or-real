"""Stage 2 SAE reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def decode_topk_linear(
    sae: Any,
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode sparse top-k SAE activations without materializing all features.

    This matches the linear decoder path used by Gemma Scope JumpReLU SAEs:
    `feature_acts @ W_dec + b_dec`, followed by SAE Lens' output transform.
    """

    if top_indices.shape != top_values.shape:
        raise ValueError(f"top_indices shape {tuple(top_indices.shape)} != top_values shape {tuple(top_values.shape)}")
    if top_indices.ndim != 2:
        raise ValueError(f"expected rank-2 top-k tensors, got {tuple(top_indices.shape)}")

    target_dtype = dtype or sae.W_dec.dtype
    device = sae.W_dec.device
    indices = top_indices.to(device=device, dtype=torch.long)
    values = top_values.to(device=device, dtype=accumulation_dtype)
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
        decoder_rows = sae.W_dec.index_select(0, indices[:, rank]).to(dtype=accumulation_dtype)
        out.add_(decoder_rows * rank_values.unsqueeze(1))

    out = sae.hook_sae_recons(out)
    out = sae.run_time_activation_norm_fn_out(out)
    out = sae.reshape_fn_out(out, sae.d_head)
    return out.to(dtype=target_dtype)


def dense_topk_features(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    *,
    d_sae: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if top_indices.shape != top_values.shape:
        raise ValueError(f"top_indices shape {tuple(top_indices.shape)} != top_values shape {tuple(top_values.shape)}")
    dense = torch.zeros((top_indices.shape[0], d_sae), dtype=dtype, device=device)
    dense.scatter_(1, top_indices.to(device=device, dtype=torch.long), top_values.to(device=device, dtype=dtype))
    return dense


@dataclass
class ReconstructionStats:
    rows: int = 0
    elements: int = 0
    raw_sq_sum: float = 0.0
    recon_sq_sum: float = 0.0
    error_sq_sum: float = 0.0
    dot_sum: float = 0.0
    cosine_sum: float = 0.0
    raw_l2_sum: float = 0.0
    recon_l2_sum: float = 0.0
    error_l2_sum: float = 0.0

    def update(self, raw: torch.Tensor, reconstruction: torch.Tensor) -> None:
        if raw.shape != reconstruction.shape:
            raise ValueError(f"raw shape {tuple(raw.shape)} != reconstruction shape {tuple(reconstruction.shape)}")
        raw_f = raw.float()
        recon_f = reconstruction.float()
        error = raw_f - recon_f
        raw_sq = raw_f.square().sum(dim=-1)
        recon_sq = recon_f.square().sum(dim=-1)
        error_sq = error.square().sum(dim=-1)
        dot = (raw_f * recon_f).sum(dim=-1)
        raw_norm = raw_sq.sqrt()
        recon_norm = recon_sq.sqrt()
        error_norm = error_sq.sqrt()
        cosine = dot / (raw_norm * recon_norm).clamp_min(1e-12)

        self.rows += int(raw_f.shape[0])
        self.elements += int(raw_f.numel())
        self.raw_sq_sum += float(raw_sq.sum().item())
        self.recon_sq_sum += float(recon_sq.sum().item())
        self.error_sq_sum += float(error_sq.sum().item())
        self.dot_sum += float(dot.sum().item())
        self.cosine_sum += float(cosine.sum().item())
        self.raw_l2_sum += float(raw_norm.sum().item())
        self.recon_l2_sum += float(recon_norm.sum().item())
        self.error_l2_sum += float(error_norm.sum().item())

    def to_dict(self) -> dict[str, float | int | None]:
        if self.rows == 0 or self.elements == 0:
            return {
                "rows": self.rows,
                "elements": self.elements,
                "mse": None,
                "rmse": None,
                "energy_explained": None,
                "relative_error_l2": None,
                "mean_row_cosine": None,
                "mean_raw_l2": None,
                "mean_reconstruction_l2": None,
                "mean_error_l2": None,
            }
        mse = self.error_sq_sum / self.elements
        return {
            "rows": self.rows,
            "elements": self.elements,
            "mse": mse,
            "rmse": mse**0.5,
            "energy_explained": 1.0 - (self.error_sq_sum / self.raw_sq_sum) if self.raw_sq_sum else None,
            "relative_error_l2": (self.error_sq_sum / self.raw_sq_sum) ** 0.5 if self.raw_sq_sum else None,
            "global_cosine": self.dot_sum / ((self.raw_sq_sum * self.recon_sq_sum) ** 0.5)
            if self.raw_sq_sum and self.recon_sq_sum
            else None,
            "mean_row_cosine": self.cosine_sum / self.rows,
            "mean_raw_l2": self.raw_l2_sum / self.rows,
            "mean_reconstruction_l2": self.recon_l2_sum / self.rows,
            "mean_error_l2": self.error_l2_sum / self.rows,
        }
