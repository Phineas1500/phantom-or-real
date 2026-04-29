"""Stage 2 helpers for Gemma Scope crosscoder pilot artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def crosscoder_feature_prefix(
    *,
    out_dir: Path,
    model_key: str,
    task: str,
    crosscoder_id: str,
    top_k: int,
    skip: int = 0,
    limit: int | None = None,
) -> Path:
    stem = f"{model_key}_{task}_crosscoder_{crosscoder_id}_top{top_k}"
    if skip:
        stem += f"_skip{skip}"
    if limit is not None:
        stem += f"_n{limit}"
    return out_dir / stem


def combine_shard_topk(
    shard_values: list[torch.Tensor],
    shard_indices: list[torch.Tensor],
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not shard_values:
        raise ValueError("at least one shard is required")
    if len(shard_values) != len(shard_indices):
        raise ValueError("shard_values and shard_indices must have the same length")
    for values, indices in zip(shard_values, shard_indices, strict=True):
        if values.shape != indices.shape:
            raise ValueError(f"values shape {values.shape} != indices shape {indices.shape}")
        if values.ndim != 2:
            raise ValueError(f"expected rank-2 shard tensors, got {values.shape}")
    values = torch.cat(shard_values, dim=1)
    indices = torch.cat(shard_indices, dim=1)
    k = min(top_k, values.shape[1])
    top_values, top_positions = torch.topk(values, k=k, dim=1)
    top_indices = torch.gather(indices, dim=1, index=top_positions)
    return top_values, top_indices


def verify_matching_sidecars(reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(f"sidecar length mismatch: {len(reference_rows)} != {len(candidate_rows)}")
    for idx, (reference, candidate) in enumerate(zip(reference_rows, candidate_rows, strict=True)):
        reference_key = (reference.get("row_index"), reference.get("example_id"))
        candidate_key = (candidate.get("row_index"), candidate.get("example_id"))
        if reference_key != candidate_key:
            raise ValueError(f"sidecar row {idx} mismatch: {reference_key} != {candidate_key}")
