"""Stage 2 SAE feature extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def slice_rows(rows: list[dict[str, Any]], *, skip: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    if skip < 0:
        raise ValueError("skip must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    end = None if limit is None else skip + limit
    return rows[skip:end]


def snapshot_revision_from_path(path: Path) -> str | None:
    # Do not resolve symlinks here: HF cache files under snapshots/<rev>/ often
    # point into blobs/, and resolving would discard the revision-bearing path.
    parts = path.parts
    for idx, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[idx + 1]
    return None


def sae_file_name(subfolder: str, sae_id: str, file_name: str) -> str:
    return f"{subfolder.strip('/')}/{sae_id}/{file_name}"


def summarize_sae_cfg(sae: Any, sae_cfg_dict: dict[str, Any]) -> dict[str, Any]:
    parameter = next(sae.parameters())
    return {
        "architecture": sae_cfg_dict.get("architecture"),
        "d_in": int(sae.cfg.d_in),
        "d_sae": int(sae.cfg.d_sae),
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "hf_hook_name": sae_cfg_dict.get("hf_hook_name"),
        "hook_name": sae_cfg_dict.get("hook_name"),
    }


def derive_sae_feature_prefix(
    *,
    activation_prefix: Path,
    out_dir: Path,
    sae_id: str,
    top_k: int,
    skip: int = 0,
    limit: int | None = None,
) -> Path:
    stem = f"{activation_prefix.name}_{sae_id}_top{top_k}"
    if skip:
        stem += f"_skip{skip}"
    if limit is not None:
        stem += f"_n{limit}"
    return out_dir / stem


def topk_tensors_to_csr(top_indices: Any, top_values: Any, *, d_sae: int):
    import numpy as np
    from scipy import sparse

    values = top_values.float().cpu().numpy()
    indices = top_indices.cpu().numpy()
    if values.shape != indices.shape:
        raise ValueError(f"top_values shape {values.shape} != top_indices shape {indices.shape}")
    if values.ndim != 2:
        raise ValueError(f"expected rank-2 top-k tensors, got shape {values.shape}")
    n_rows, top_k = values.shape
    indptr = np.arange(0, (n_rows + 1) * top_k, top_k, dtype=np.int64)
    return sparse.csr_matrix(
        (values.reshape(-1), indices.reshape(-1), indptr),
        shape=(n_rows, d_sae),
    )
