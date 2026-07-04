import json
from pathlib import Path

import numpy as np
import torch

from scripts.stage2_rank_k_guard_v2 import (
    COMPOSITE_MANIFEST_ROWS,
    build_arms,
    build_specificity_arms,
    control_add_matrix,
    select_fresh_rows,
    shard_rows,
)
from scripts.stage2_subtype_discriminator import fit_pca_basis, rank_k_reconstruction


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_row(height: int, correct: bool = False, parse_failed: bool = False) -> dict:
    return {"height": height, "is_correct_strong": correct, "parse_failed": parse_failed}


def test_select_fresh_rows_filters_and_balances(tmp_path: Path) -> None:
    rows = []
    for _ in range(10):
        rows.append(make_row(3))
        rows.append(make_row(4))
    rows.append(make_row(3, correct=True))
    rows.append(make_row(4, parse_failed=True))
    rows.append(make_row(2))
    path = tmp_path / "stage1.jsonl"
    _write_jsonl(path, rows)

    selected = select_fresh_rows(path, exclude={0, 1}, heights=[3, 4], per_height=4, seed=7)
    assert len(selected) == 8
    assert sum(1 for row in selected if row["height"] == 3) == 4
    assert sum(1 for row in selected if row["height"] == 4) == 4
    assert all(not row["is_correct_strong"] and not row["parse_failed"] for row in selected)
    assert all(row["row_index"] not in {0, 1} for row in selected)

    again = select_fresh_rows(path, exclude={0, 1}, heights=[3, 4], per_height=4, seed=7)
    assert [row["row_index"] for row in again] == [row["row_index"] for row in selected]


def test_select_fresh_rows_raises_when_pool_too_small(tmp_path: Path) -> None:
    path = tmp_path / "stage1.jsonl"
    _write_jsonl(path, [make_row(3), make_row(4)])
    try:
        select_fresh_rows(path, exclude=set(), heights=[3, 4], per_height=2, seed=1)
    except ValueError as err:
        assert "eligible rows" in str(err)
    else:
        raise AssertionError("expected ValueError for undersized pool")


def test_shard_rows_interleaves_and_preserves_height_balance(tmp_path: Path) -> None:
    rows = [{"row_index": i, "height": 3 if i % 2 == 0 else 4} for i in range(16)]
    shard0 = shard_rows(rows, 0, 2)
    shard1 = shard_rows(rows, 1, 2)
    assert len(shard0) == len(shard1) == 8
    assert {row["row_index"] for row in shard0} | {row["row_index"] for row in shard1} == set(range(16))
    assert not {row["row_index"] for row in shard0} & {row["row_index"] for row in shard1}
    for shard in (shard0, shard1):
        assert sum(1 for row in shard if row["height"] == 3) == 4


def test_build_arms_shape() -> None:
    arms = build_arms([4, 8], 30)
    labels = [arm.label for arm in arms]
    assert labels == [
        "unhinted_baseline",
        "hinted_baseline",
        "L30_concept_replace",
        "L30_random_replace",
        "rank4_loo_add_L30",
        "rank8_loo_add_L30",
    ]
    assert all(arm.reference == "unhinted_baseline" for arm in arms[1:])
    assert arms[4].basis_mode == "leave_one_row_out"


def test_composite_manifest_rows_pinned() -> None:
    assert len(COMPOSITE_MANIFEST_ROWS) == 13
    assert COMPOSITE_MANIFEST_ROWS == sorted(COMPOSITE_MANIFEST_ROWS)


def test_build_specificity_arms_shape() -> None:
    arms = build_specificity_arms(30, 8, 2)
    labels = [arm.label for arm in arms]
    assert labels == [
        "unhinted_baseline",
        "hinted_baseline",
        "rank8_loo_add_L30",
        "mean_only_add_L30",
        "rand_subspace_add_L30_d1",
        "rand_subspace_add_L30_d2",
        "rand_norm_add_L30_d1",
        "rand_norm_add_L30_d2",
    ]
    assert all(arm.reference == "unhinted_baseline" for arm in arms[1:])
    assert arms[3].basis_mode == "leave_one_row_out"
    assert arms[4].basis_mode == "random_orthonormal"
    assert arms[6].basis_mode == "norm_matched_gaussian"


def test_control_add_matrix_norm_matching_and_determinism() -> None:
    rng = np.random.default_rng(3)
    delta_by_row = {i: rng.standard_normal((6, 32)).astype(np.float32) for i in range(4)}
    basis = fit_pca_basis(delta_by_row, 4, exclude_rows={0})
    delta = delta_by_row[0]
    recon = rank_k_reconstruction(torch.from_numpy(delta), basis).numpy().astype(np.float32)
    arms = build_specificity_arms(30, 4, 1)
    by_kind = {arm.kind: arm for arm in arms if arm.kind in {"mean_add", "rand_subspace_add", "rand_norm_add"}}
    q_cache: dict[int, np.ndarray] = {}
    kwargs = {"control_seed": 11, "shard_index": 0, "source_row_index": 0, "q_cache": q_cache}

    mean_vec = control_add_matrix(by_kind["mean_add"], delta, basis, recon, **kwargs)
    assert np.allclose(mean_vec, np.tile(basis["mean"], (6, 1)), atol=1e-5)

    sub = control_add_matrix(by_kind["rand_subspace_add"], delta, basis, recon, **kwargs)
    np.testing.assert_allclose(
        np.linalg.norm(sub - basis["mean"][None, :].astype(np.float32), axis=1),
        np.linalg.norm(recon - basis["mean"][None, :].astype(np.float32), axis=1),
        rtol=1e-3,
    )
    assert 1 in q_cache and q_cache[1].shape == (32, 4)

    noise = control_add_matrix(by_kind["rand_norm_add"], delta, basis, recon, **kwargs)
    np.testing.assert_allclose(np.linalg.norm(noise, axis=1), np.linalg.norm(recon, axis=1), rtol=1e-3)
    assert not np.allclose(noise, recon)

    again = control_add_matrix(by_kind["rand_norm_add"], delta, basis, recon, **kwargs)
    np.testing.assert_allclose(noise, again)


def test_build_predcoeff_arms_shape() -> None:
    from scripts.stage2_rank_k_guard_v2 import build_predcoeff_arms

    arms = build_predcoeff_arms(30, 8)
    assert [arm.label for arm in arms] == [
        "unhinted_baseline",
        "hinted_baseline",
        "rank8_dev_add_L30",
        "mean_only_dev_add_L30",
        "rank8_pred_add_L30",
        "rank8_shufpred_add_L30",
    ]
    assert [arm.kind for arm in arms] == [
        "none",
        "hinted_prompt",
        "rank_dev_add",
        "mean_dev_add",
        "pred_add",
        "shufpred_add",
    ]
    assert arms[4].basis_mode == "ridge_predicted"
    assert arms[5].basis_mode == "ridge_shuffled"


def test_load_dev_states_roundtrip(tmp_path: Path) -> None:
    from scripts.stage2_rank_k_guard_v2 import load_dev_states

    rng = np.random.default_rng(0)
    arrays = {}
    for row in (11, 22):
        arrays[f"L30_row{row}_concept_delta"] = rng.standard_normal((5, 16)).astype(np.float32)
        arrays[f"L30_row{row}_unhinted_concept_states"] = rng.standard_normal((5, 16)).astype(np.float32)
    path = tmp_path / "dev.npz"
    np.savez_compressed(path, **arrays)

    unhinted, delta = load_dev_states(path, 30)
    assert set(unhinted) == set(delta) == {11, 22}
    assert unhinted[11].shape == (5, 16)


def test_fit_coeff_predictor_recovers_linear_map_and_shuffle_breaks_it() -> None:
    from scripts.stage2_rank_k_guard_v2 import _ridge_predict, fit_coeff_predictor

    rng = np.random.default_rng(3)
    true_map = rng.standard_normal((32, 4))
    dev_unhinted = {}
    dev_coeffs = {}
    for row in range(8):
        x = rng.standard_normal((6, 32))
        dev_unhinted[row] = x
        dev_coeffs[row] = x @ true_map + 0.01 * rng.standard_normal((6, 4))

    fitted = fit_coeff_predictor(dev_unhinted, dev_coeffs, alphas=(1e-2, 1e0, 1e2))
    again = fit_coeff_predictor(dev_unhinted, dev_coeffs, alphas=(1e-2, 1e0, 1e2))
    assert fitted["alpha"] == again["alpha"]
    assert fitted["loo_cosine"] == again["loo_cosine"]
    assert fitted["loo_cosine"] > 0.8

    shuffled = fit_coeff_predictor(dev_unhinted, dev_coeffs, alphas=(1e-2, 1e0, 1e2), shuffle_seed=5)
    assert shuffled["loo_cosine"] < 0.3

    probe_x = rng.standard_normal((3, 32))
    pred = _ridge_predict(fitted["model"], probe_x)
    assert pred.shape == (3, 4)
    cos = (pred * (probe_x @ true_map)).sum(axis=1) / (
        np.linalg.norm(pred, axis=1) * np.linalg.norm(probe_x @ true_map, axis=1) + 1e-12
    )
    assert cos.mean() > 0.8
