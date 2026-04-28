"""Feature-weight stability helpers for Stage 2 SAE probes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_1d_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def coefficient_weights(
    standardized_coef: Any,
    *,
    scaler_scale: Any | None = None,
    top_n: int,
    mode: str = "abs",
) -> list[dict[str, Any]]:
    """Return top logistic coefficients with stable ordering.

    Coefficients are ranked in standardized feature space because that is the
    space used by the trained probe. If `scaler_scale` is provided, each record
    also includes the equivalent input-space logit coefficient.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    coef = _as_1d_array(standardized_coef)
    scale = None if scaler_scale is None else _as_1d_array(scaler_scale)
    if scale is not None and scale.shape != coef.shape:
        raise ValueError(f"scale shape {scale.shape} does not match coef shape {coef.shape}")

    if mode == "abs":
        candidates = np.flatnonzero(np.abs(coef) > 0)
        sort_values = -np.abs(coef)
    elif mode == "positive":
        candidates = np.flatnonzero(coef > 0)
        sort_values = -coef[candidates]
    elif mode == "negative":
        candidates = np.flatnonzero(coef < 0)
        sort_values = coef[candidates]
    else:
        raise ValueError(f"unknown coefficient mode: {mode}")

    if candidates.size == 0:
        return []
    if mode == "abs":
        order = np.lexsort((candidates, sort_values[candidates]))
    else:
        order = np.lexsort((candidates, sort_values))
    selected = candidates[order[:top_n]]

    rows = []
    for rank, feature in enumerate(selected, start=1):
        weight = float(coef[feature])
        input_weight = None
        if scale is not None:
            input_weight = float(weight / scale[feature]) if scale[feature] != 0 else None
        rows.append(
            {
                "rank": rank,
                "feature": int(feature),
                "weight": weight,
                "abs_weight": abs(weight),
                "input_weight": input_weight,
                "sign": "positive" if weight > 0 else "negative" if weight < 0 else "zero",
                "association": "correct" if weight > 0 else "incorrect" if weight < 0 else "neutral",
            }
        )
    return rows


def feature_overlap(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    top_n: int,
) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    left_rows = list(left[:top_n])
    right_rows = list(right[:top_n])
    left_by_feature = {int(row["feature"]): row for row in left_rows}
    right_by_feature = {int(row["feature"]): row for row in right_rows}
    left_features = set(left_by_feature)
    right_features = set(right_by_feature)
    overlap = sorted(left_features & right_features)
    union_n = len(left_features | right_features)
    signed_agreement = [
        feature
        for feature in overlap
        if left_by_feature[feature].get("sign") == right_by_feature[feature].get("sign")
    ]
    return {
        "top_n": top_n,
        "left_n": len(left_features),
        "right_n": len(right_features),
        "overlap_n": len(overlap),
        "jaccard": (len(overlap) / union_n) if union_n else None,
        "signed_agreement_n": len(signed_agreement),
        "signed_agreement_rate": (len(signed_agreement) / len(overlap)) if overlap else None,
        "overlap_features": overlap,
        "signed_agreement_features": signed_agreement,
    }


def feature_activation_stats(
    x: Any,
    feature_ids: Sequence[int],
    *,
    row_indices: Sequence[int] | None = None,
) -> dict[int, dict[str, Any]]:
    from scipy import sparse

    matrix = x if row_indices is None else x[list(row_indices)]
    n_rows = int(matrix.shape[0])
    stats = {}
    for feature in feature_ids:
        column = matrix[:, int(feature)]
        if sparse.issparse(column):
            column = column.tocsr(copy=True)
            column.eliminate_zeros()
            values = column.data.astype(np.float64, copy=False)
            nonzero_n = int(column.getnnz())
            total = float(column.sum())
            max_value = float(values.max()) if nonzero_n else 0.0
            mean_nonzero = float(values.mean()) if nonzero_n else None
        else:
            values = np.asarray(column, dtype=np.float64).reshape(-1)
            nonzero_values = values[values != 0]
            nonzero_n = int(nonzero_values.size)
            total = float(values.sum())
            max_value = float(nonzero_values.max()) if nonzero_n else 0.0
            mean_nonzero = float(nonzero_values.mean()) if nonzero_n else None
        stats[int(feature)] = {
            "nonzero_n": nonzero_n,
            "density": (nonzero_n / n_rows) if n_rows else None,
            "mean_all": (total / n_rows) if n_rows else None,
            "mean_nonzero": mean_nonzero,
            "max": max_value,
        }
    return stats


def dense_feature_columns(
    x: Any,
    feature_ids: Sequence[int],
    *,
    row_indices: Sequence[int] | None = None,
) -> np.ndarray:
    matrix = x if row_indices is None else x[list(row_indices)]
    columns = matrix[:, [int(feature) for feature in feature_ids]]
    if hasattr(columns, "toarray"):
        return columns.toarray().astype(np.float64, copy=False)
    return np.asarray(columns, dtype=np.float64)


def pairwise_column_correlations(left_columns: Any, right_columns: Any) -> np.ndarray:
    left = np.asarray(left_columns, dtype=np.float64)
    right = np.asarray(right_columns, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left_columns and right_columns must be rank-2")
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"row mismatch: {left.shape[0]} != {right.shape[0]}")

    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    left_norm = np.linalg.norm(left_centered, axis=0)
    right_norm = np.linalg.norm(right_centered, axis=0)
    left_norm[left_norm == 0] = np.nan
    right_norm[right_norm == 0] = np.nan
    corr = (left_centered.T @ right_centered) / np.outer(left_norm, right_norm)
    return np.nan_to_num(corr, nan=0.0)


def best_correlation_matches(
    correlations: Any,
    *,
    left_features: Sequence[int],
    right_features: Sequence[int],
) -> list[dict[str, Any]]:
    corr = np.asarray(correlations, dtype=np.float64)
    if corr.shape != (len(left_features), len(right_features)):
        raise ValueError(
            f"correlation shape {corr.shape} does not match "
            f"{len(left_features)}x{len(right_features)} features"
        )
    rows = []
    for left_idx, left_feature in enumerate(left_features):
        if len(right_features) == 0:
            break
        right_idx = int(np.argmax(np.abs(corr[left_idx])))
        value = float(corr[left_idx, right_idx])
        rows.append(
            {
                "source_feature": int(left_feature),
                "target_feature": int(right_features[right_idx]),
                "source_rank": left_idx + 1,
                "target_rank": right_idx + 1,
                "correlation": value,
                "abs_correlation": abs(value),
            }
        )
    rows.sort(key=lambda row: (-row["abs_correlation"], row["source_rank"]))
    return rows


def summarize_correlation_matches(
    matches: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7),
) -> dict[str, Any]:
    values = np.asarray([float(row["abs_correlation"]) for row in matches], dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0,
            "mean_abs_correlation": None,
            "median_abs_correlation": None,
            "max_abs_correlation": None,
            "threshold_counts": {str(threshold): 0 for threshold in thresholds},
        }
    return {
        "n": int(values.size),
        "mean_abs_correlation": float(values.mean()),
        "median_abs_correlation": float(np.median(values)),
        "max_abs_correlation": float(values.max()),
        "threshold_counts": {
            str(threshold): int(np.sum(values >= threshold))
            for threshold in thresholds
        },
    }
