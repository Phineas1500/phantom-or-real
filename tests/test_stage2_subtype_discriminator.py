from __future__ import annotations

import numpy as np

from scripts.stage2_subtype_discriminator import build_arms, parse_int_list, select_offtrio_layers, summarize_layer_deltas


def test_parse_int_list_rejects_empty() -> None:
    assert parse_int_list("15, 20,30") == [15, 20, 30]
    try:
        parse_int_list("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty list should fail")


def test_select_offtrio_layers_excludes_old_trio_and_ranks_by_metric() -> None:
    ladder = [
        {"layer": 30, "row_mean_delta_norm_mean": 100.0},
        {"layer": 20, "row_mean_delta_norm_mean": 10.0},
        {"layer": 50, "row_mean_delta_norm_mean": 30.0},
        {"layer": 35, "row_mean_delta_norm_mean": 30.0},
        {"layer": 45, "row_mean_delta_norm_mean": 200.0},
    ]

    assert select_offtrio_layers(ladder, [30, 40, 45], 3) == [35, 50, 20]


def test_build_arms_adds_old_trio_selected_controls_and_rank_rider() -> None:
    arms = build_arms([30, 40, 45], [35, 50], rank_k=4)

    assert [arm.label for arm in arms] == [
        "baseline",
        "old_trio_full_replace_L30_40_45",
        "L35_concept_replace",
        "L50_concept_replace",
        "L35_random_replace",
        "L50_random_replace",
        "L35_rank4_loo_add",
    ]
    assert arms[-1].basis_mode == "leave_one_row_out"
    assert arms[-1].reference == "baseline"


def test_summarize_layer_deltas_reports_row_weighted_and_position_stats() -> None:
    summary = summarize_layer_deltas(
        {
            1: np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32),
            2: np.array([[0.0, 12.0]], dtype=np.float32),
        }
    )

    assert summary["n_rows"] == 2
    assert summary["n_positions"] == 3
    assert summary["row_mean_delta_norm_mean"] == 7.25
    assert summary["position_delta_norm_max"] == 12.0
