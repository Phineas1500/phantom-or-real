from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from scripts.stage2_erasure_control_matching import (
    make_condition_plan,
    make_control_matching_hook,
    parse_condition_kinds,
    parse_float_list,
    summarize_hook_state,
    summarize_projection_telemetry,
)


def test_parse_control_matching_inputs_reject_bad_values() -> None:
    assert parse_float_list("0.25,0.5,1") == [0.25, 0.5, 1.0]
    assert parse_condition_kinds("baseline,erase_raw,erase_height") == [
        "baseline",
        "erase_raw",
        "erase_height",
    ]

    with pytest.raises(ValueError, match="positive"):
        parse_float_list("1,0")
    with pytest.raises(ValueError, match="unknown"):
        parse_condition_kinds("baseline,erase_magic")
    with pytest.raises(ValueError, match="baseline"):
        parse_condition_kinds("erase_raw")


def test_make_condition_plan_orders_controls_and_scales() -> None:
    plan = make_condition_plan(
        condition_kinds=parse_condition_kinds(
            "baseline,erase_raw,erase_height,erase_orthogonal,erase_gaussian"
        ),
        control_scales=[0.25, 0.5, 1.0],
        raw_scale=1.0,
        height_scale=1.0,
    )

    assert [condition.label for condition in plan] == [
        "baseline",
        "erase_raw_s1",
        "erase_height_s1",
        "erase_orthogonal_s0p25",
        "erase_orthogonal_s0p5",
        "erase_orthogonal_s1",
        "erase_gaussian_s0p25",
        "erase_gaussian_s0p5",
        "erase_gaussian_s1",
    ]
    assert [condition.vector_kind for condition in plan] == [
        None,
        "raw",
        "height",
        "orthogonal",
        "orthogonal",
        "orthogonal",
        "gaussian",
        "gaussian",
        "gaussian",
    ]


def test_control_matching_hook_applies_fractional_erasure_and_records_variance() -> None:
    hook, state = make_control_matching_hook(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        projection_mean=2.0,
        projection_std=2.0,
        scale=0.5,
    )
    hidden = torch.tensor([[[6.0, 1.0], [0.0, 3.0]]], dtype=torch.float32)

    patched = hook(hidden.clone(), None)

    assert_close(patched[:, :, 0], torch.tensor([[4.0, 1.0]]))
    assert_close(patched[:, :, 1], torch.tensor([[1.0, 3.0]]))

    summary = summarize_hook_state(state)
    assert summary["calls"] == 1
    assert summary["prompt_calls"] == 1
    assert summary["decode_calls"] == 0
    assert summary["positions"] == 2
    assert summary["mean_delta_sd"] == pytest.approx(0.5)
    assert summary["std_delta_sd"] == pytest.approx(1.5)
    assert summary["mean_abs_delta_sd"] == pytest.approx(1.5)
    assert summary["mean_within_call_var_sd2"] == pytest.approx(2.25)
    assert summary["position_weighted_within_call_var_sd2"] == pytest.approx(2.25)
    assert summary["max_abs_delta_sd"] == pytest.approx(2.0)
    assert summary["scale"] == pytest.approx(0.5)


def test_projection_telemetry_aggregates_hook_summaries() -> None:
    hook_a, state_a = make_control_matching_hook(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        projection_mean=2.0,
        projection_std=2.0,
        scale=1.0,
    )
    hook_a(torch.tensor([[[6.0, 1.0], [0.0, 3.0]]], dtype=torch.float32), None)

    hook_b, state_b = make_control_matching_hook(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        projection_mean=2.0,
        projection_std=2.0,
        scale=1.0,
    )
    hook_b(torch.tensor([[[4.0, 0.0]]], dtype=torch.float32), None)

    telemetry = summarize_projection_telemetry(
        [
            {"condition": "erase_raw_s1", "hook_summary": {"L30": summarize_hook_state(state_a)}},
            {"condition": "erase_raw_s1", "hook_summary": {"L30": summarize_hook_state(state_b)}},
        ]
    )

    layer = telemetry["erase_raw_s1"]["L30"]
    assert layer["generations"] == 2
    assert layer["calls"] == 2
    assert layer["prompt_calls"] == 1
    assert layer["decode_calls"] == 1
    assert layer["positions"] == 3
    assert layer["mean_abs_delta_sd"] == pytest.approx(4.0 / 3.0)
    assert layer["max_abs_delta_sd"] == pytest.approx(2.0)
