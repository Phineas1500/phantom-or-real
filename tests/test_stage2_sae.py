from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.stage2_feature_stability import (
    best_correlation_matches,
    coefficient_weights,
    feature_overlap,
    pairwise_column_correlations,
    summarize_correlation_matches,
)
from src.stage2_sae import (
    derive_sae_feature_prefix,
    sae_file_name,
    slice_rows,
    snapshot_revision_from_path,
    topk_tensors_to_csr,
)


def test_slice_rows_honors_skip_and_limit() -> None:
    rows = [{"row_index": idx} for idx in range(5)]

    assert slice_rows(rows, skip=1, limit=2) == [{"row_index": 1}, {"row_index": 2}]
    assert slice_rows(rows, skip=3, limit=None) == [{"row_index": 3}, {"row_index": 4}]


def test_snapshot_revision_from_hf_cache_path() -> None:
    path = Path("/cache/models--x/snapshots/abc123/resid_post_all/layer/config.json")

    assert snapshot_revision_from_path(path) == "abc123"


def test_sae_file_name_joins_subfolder_and_id() -> None:
    assert (
        sae_file_name("resid_post_all/", "layer_45_width_16k_l0_small", "params.safetensors")
        == "resid_post_all/layer_45_width_16k_l0_small/params.safetensors"
    )


def test_derive_sae_feature_prefix_includes_slice() -> None:
    prefix = derive_sae_feature_prefix(
        activation_prefix=Path("results/stage2/activations/gemma3_27b_infer_property_L45"),
        out_dir=Path("results/stage2/sae_features"),
        sae_id="layer_45_width_16k_l0_small",
        top_k=128,
        skip=10,
        limit=512,
    )

    assert prefix == Path(
        "results/stage2/sae_features/"
        "gemma3_27b_infer_property_L45_layer_45_width_16k_l0_small_top128_skip10_n512"
    )


def test_topk_tensors_to_csr_builds_sparse_feature_matrix() -> None:
    top_indices = torch.tensor([[1, 3], [0, 2]], dtype=torch.int64)
    top_values = torch.tensor([[0.5, 1.5], [2.0, 4.0]], dtype=torch.float32)

    matrix = topk_tensors_to_csr(top_indices, top_values, d_sae=5)

    assert matrix.shape == (2, 5)
    assert matrix[0, 1] == 0.5
    assert matrix[0, 3] == 1.5
    assert matrix[1, 0] == 2.0
    assert matrix[1, 2] == 4.0


def test_coefficient_weights_rank_by_standardized_abs_weight() -> None:
    rows = coefficient_weights([0.1, -2.0, 1.5], scaler_scale=[1.0, 4.0, 0.5], top_n=2)

    assert [row["feature"] for row in rows] == [1, 2]
    assert rows[0]["association"] == "incorrect"
    assert rows[0]["input_weight"] == -0.5
    assert rows[1]["association"] == "correct"


def test_feature_overlap_counts_signed_agreement() -> None:
    left = [
        {"feature": 1, "sign": "positive"},
        {"feature": 2, "sign": "negative"},
        {"feature": 3, "sign": "positive"},
    ]
    right = [
        {"feature": 2, "sign": "negative"},
        {"feature": 3, "sign": "negative"},
        {"feature": 4, "sign": "positive"},
    ]

    overlap = feature_overlap(left, right, top_n=3)

    assert overlap["overlap_n"] == 2
    assert overlap["signed_agreement_n"] == 1
    assert overlap["signed_agreement_features"] == [2]


def test_pairwise_column_correlations_and_best_matches() -> None:
    left = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, -1.0],
            [3.0, -2.0],
        ],
        dtype=torch.float32,
    ).numpy()
    right = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ],
        dtype=torch.float32,
    ).numpy()

    corr = pairwise_column_correlations(left, right)
    matches = best_correlation_matches(corr, left_features=[10, 11], right_features=[20, 21])
    summary = summarize_correlation_matches(matches, thresholds=(0.9,))

    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[1, 1] == pytest.approx(-1.0)
    assert matches[0]["abs_correlation"] == pytest.approx(1.0)
    assert summary["threshold_counts"]["0.9"] == 2
