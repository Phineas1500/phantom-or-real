from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.stage2_crosscoder import combine_shard_topk, crosscoder_feature_prefix, verify_matching_sidecars
from src.stage2_feature_stability import (
    best_correlation_matches,
    coefficient_weights,
    feature_overlap,
    pairwise_column_correlations,
    summarize_correlation_matches,
)
from src.stage2_reconstruction import ReconstructionStats, decode_topk_linear, dense_topk_features
from src.stage2_sae import (
    derive_sae_feature_prefix,
    sae_file_name,
    slice_rows,
    snapshot_revision_from_path,
    summarize_sae_cfg,
    topk_tensors_to_csr,
)
from src.stage2_paths import activation_stem, hook_name_for_layer, normalize_activation_site


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


def test_crosscoder_feature_prefix_includes_slice() -> None:
    prefix = crosscoder_feature_prefix(
        out_dir=Path("results/stage2/crosscoder_features"),
        model_key="gemma3_27b",
        task="infer_property",
        crosscoder_id="layer_16_31_40_53_width_65k_l0_medium",
        top_k=128,
        skip=10,
        limit=512,
    )

    assert prefix == Path(
        "results/stage2/crosscoder_features/"
        "gemma3_27b_infer_property_crosscoder_layer_16_31_40_53_width_65k_l0_medium_top128_skip10_n512"
    )


def test_summarize_sae_cfg_reads_transcoder_metadata_hooks() -> None:
    class _Cfg:
        d_in = 3
        d_sae = 5
        d_out = 7

    class _FakeSae(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.cfg = _Cfg()

    summary = summarize_sae_cfg(
        _FakeSae(),
        {
            "architecture": "jumprelu_skip_transcoder",
            "metadata": {
                "hook_name": "blocks.45.hook_mlp_in",
                "hf_hook_name": "model.layers.45.pre_feedforward_layernorm.output",
                "hook_name_out": "blocks.45.hook_mlp_out",
                "hf_hook_name_out": "model.layers.45.post_feedforward_layernorm.output",
            },
        },
    )

    assert summary["architecture"] == "jumprelu_skip_transcoder"
    assert summary["d_out"] == 7
    assert summary["hook_name"] == "blocks.45.hook_mlp_in"
    assert summary["hook_name_out"] == "blocks.45.hook_mlp_out"
    assert summary["hf_hook_name_out"] == "model.layers.45.post_feedforward_layernorm.output"


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


def test_activation_stem_preserves_residual_default_and_adds_site_suffix() -> None:
    assert (
        activation_stem(model_key="gemma3_27b", task="infer_property", layer=45)
        == "gemma3_27b_infer_property_L45"
    )
    assert (
        activation_stem(
            model_key="gemma3_27b",
            task="infer_property",
            layer=45,
            activation_site="mlp-out",
        )
        == "gemma3_27b_infer_property_L45_mlp_out"
    )
    assert normalize_activation_site("") == "resid_post"
    assert hook_name_for_layer(layer=45, hook_template="blocks.{layer}.ln2_post.hook_normalized") == (
        "blocks.45.ln2_post.hook_normalized"
    )
    with pytest.raises(ValueError, match="hook-template"):
        hook_name_for_layer(layer=45, hook_template="blocks.45.hook_mlp_out")


def test_topk_tensors_to_csr_builds_sparse_feature_matrix() -> None:
    top_indices = torch.tensor([[1, 3], [0, 2]], dtype=torch.int64)
    top_values = torch.tensor([[0.5, 1.5], [2.0, 4.0]], dtype=torch.float32)

    matrix = topk_tensors_to_csr(top_indices, top_values, d_sae=5)

    assert matrix.shape == (2, 5)
    assert matrix[0, 1] == 0.5
    assert matrix[0, 3] == 1.5
    assert matrix[1, 0] == 2.0
    assert matrix[1, 2] == 4.0


def test_combine_shard_topk_offsets_global_indices() -> None:
    values, indices = combine_shard_topk(
        [
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            torch.tensor([[0.7, 0.3], [0.4, 1.0]]),
        ],
        [
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([[2, 3], [2, 3]]),
        ],
        top_k=2,
    )

    assert torch.allclose(values, torch.tensor([[0.9, 0.7], [1.0, 0.8]]))
    assert indices.tolist() == [[1, 2], [3, 0]]


def test_verify_matching_sidecars_rejects_mismatched_rows() -> None:
    rows = [{"row_index": 0, "example_id": "a"}]

    verify_matching_sidecars(rows, [{"row_index": 0, "example_id": "a"}])
    with pytest.raises(ValueError, match="sidecar row"):
        verify_matching_sidecars(rows, [{"row_index": 1, "example_id": "a"}])


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


class _FakeLinearSae:
    def __init__(self) -> None:
        self.W_dec = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 2.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        self.b_dec = torch.tensor([0.5, -0.5, 1.0])
        self.d_head = None

    def hook_sae_recons(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def run_time_activation_norm_fn_out(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reshape_fn_out(self, x: torch.Tensor, d_head) -> torch.Tensor:
        return x


def test_decode_topk_linear_matches_dense_decode() -> None:
    sae = _FakeLinearSae()
    top_indices = torch.tensor([[0, 3], [2, 1]], dtype=torch.int64)
    top_values = torch.tensor([[2.0, 1.0], [4.0, 0.5]], dtype=torch.float32)

    sparse_decoded = decode_topk_linear(sae, top_indices, top_values, dtype=torch.float32)
    dense_features = dense_topk_features(
        top_indices,
        top_values,
        d_sae=4,
        dtype=torch.float32,
        device="cpu",
    )
    dense_decoded = dense_features @ sae.W_dec + sae.b_dec

    assert sparse_decoded == pytest.approx(dense_decoded)


def test_reconstruction_stats_reports_energy_explained() -> None:
    stats = ReconstructionStats()
    raw = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    reconstruction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    stats.update(raw, reconstruction)
    summary = stats.to_dict()

    assert summary["rows"] == 2
    assert summary["mse"] == pytest.approx(0.25)
    assert summary["energy_explained"] == pytest.approx(0.8)
    assert summary["relative_error_l2"] == pytest.approx((1.0 / 5.0) ** 0.5)
