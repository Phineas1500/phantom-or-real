from __future__ import annotations

import numpy as np
import pytest

from scripts.stage2_rank_k_guard import build_arms, fit_pca_basis, load_delta_by_row, parse_basis_modes, parse_rank_list


def test_parse_rank_list_rejects_nonpositive() -> None:
    assert parse_rank_list("1,2,4") == [1, 2, 4]
    with pytest.raises(ValueError):
        parse_rank_list("1,0")


def test_parse_basis_modes_rejects_unknown() -> None:
    assert parse_basis_modes("leave_one_row_out,in_sample") == ["leave_one_row_out", "in_sample"]
    with pytest.raises(ValueError):
        parse_basis_modes("bootstrap")


def test_build_arms_orders_basis_modes_then_ranks() -> None:
    arms = build_arms([1, 4], ["leave_one_row_out", "in_sample"])
    assert [arm.label for arm in arms] == [
        "unhinted_baseline",
        "rank1_loo_L30",
        "rank4_loo_L30",
        "rank1_in_sample_L30",
        "rank4_in_sample_L30",
    ]


def test_load_and_fit_pca_basis_excludes_heldout_row(tmp_path: Path) -> None:
    states = tmp_path / "states.npz"
    np.savez(
        states,
        L30_row10_concept_delta=np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        L30_row20_concept_delta=np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32),
        L40_row10_concept_delta=np.array([[99.0, 99.0]], dtype=np.float32),
    )

    by_row = load_delta_by_row(states, 30)
    basis = fit_pca_basis(by_row, rank_k=1, exclude_rows={10})

    assert sorted(by_row) == [10, 20]
    assert basis["source_rows"] == [20]
    assert basis["exclude_rows"] == [10]
    assert basis["components"].shape == (1, 2)
    assert basis["n_pooled_positions"] == 2
