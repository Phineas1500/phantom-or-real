"""Dense active-feature probe helpers for sparse Stage 2 artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def train_active_feature_ids(x: Any, train_indices: Sequence[int]) -> np.ndarray:
    """Return feature columns active in the train split.

    Sparse SAE/crosscoder probes normally use CSR matrices and therefore cannot
    mean-center features. This helper selects only train-active columns before
    materializing them as dense arrays for a centered-scaling sanity check.
    """

    if not train_indices:
        raise ValueError("train_indices must not be empty")
    try:
        from scipy import sparse
    except Exception:  # noqa: BLE001
        sparse = None

    if sparse is not None and sparse.issparse(x):
        ids = np.unique(x[list(train_indices)].nonzero()[1])
    else:
        train = np.asarray(x)[list(train_indices)]
        ids = np.flatnonzero(np.any(train != 0, axis=0))
    return ids.astype(np.int64, copy=False)


def dense_active_matrix(x: Any, feature_ids: Sequence[int]) -> np.ndarray:
    """Materialize selected feature columns as a dense float64 matrix."""

    if len(feature_ids) == 0:
        raise ValueError("feature_ids must not be empty")
    columns = x[:, [int(feature) for feature in feature_ids]]
    if hasattr(columns, "toarray"):
        return columns.toarray().astype(np.float64, copy=False)
    return np.asarray(columns, dtype=np.float64)


def sparse_feature_source_file(meta: dict[str, Any]) -> str:
    """Return the Stage 1 source JSONL path from SAE/crosscoder metadata."""

    source_file = meta.get("source_file")
    if source_file:
        return str(source_file)
    source_activation_meta = meta.get("source_activation_meta") or {}
    source_file = source_activation_meta.get("jsonl_path")
    if source_file:
        return str(source_file)
    raise KeyError("could not infer source JSONL path from sparse feature metadata")


def sparse_feature_width(meta: dict[str, Any]) -> int:
    cfg = meta.get("sae_cfg") or meta.get("crosscoder_cfg")
    if not cfg or "d_sae" not in cfg:
        raise KeyError("could not infer sparse feature width from metadata")
    return int(cfg["d_sae"])
