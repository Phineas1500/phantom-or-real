from __future__ import annotations

import numpy as np

from scripts.stage2_rank_core_geometry import merge_top_abs, orthonormal_columns, subspace_fraction


def test_subspace_fraction_is_one_for_same_subspace() -> None:
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = orthonormal_columns(rows)
    assert np.isclose(subspace_fraction(q, q), 1.0)


def test_subspace_fraction_partial_overlap() -> None:
    target = orthonormal_columns(np.array([[1.0, 0.0, 0.0]]))
    reference = orthonormal_columns(np.array([[1.0, 1.0, 0.0]]))
    assert np.isclose(subspace_fraction(target, reference), 0.5)


def test_merge_top_abs_keeps_largest_abs_values() -> None:
    current = {"feature_index": [1], "signed_cosine": [-0.4], "abs_cosine": [0.4]}
    merged = merge_top_abs(current, np.array([0.2, -0.8, 0.6]), np.array([2, 3, 4]), top_n=2)
    assert merged["feature_index"] == [3, 4]
    assert merged["signed_cosine"] == [-0.8, 0.6]
    assert merged["abs_cosine"] == [0.8, 0.6]
