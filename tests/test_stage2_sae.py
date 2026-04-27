from __future__ import annotations

from pathlib import Path

import torch

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
