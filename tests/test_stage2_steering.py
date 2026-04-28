from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stage2_steering import (  # noqa: E402
    make_condition_plan,
    make_orthogonal_unit_direction,
    select_balanced_stage1_rows,
    summarize_steering_rows,
    train_raw_probe_direction,
)


def test_make_condition_plan_labels_strengths() -> None:
    plan = make_condition_plan(
        condition_kinds=["baseline", "raw", "orthogonal"],
        strengths=[-2.0, 2.0],
    )

    assert [condition.label for condition in plan] == [
        "baseline",
        "raw_neg2sd",
        "raw_pos2sd",
        "orthogonal_neg2sd",
        "orthogonal_pos2sd",
    ]


def test_make_orthogonal_unit_direction_is_orthogonal_and_unit() -> None:
    direction = torch.tensor([1.0, 2.0, -1.0]).numpy()

    orthogonal = make_orthogonal_unit_direction(direction, seed=123)
    unit = direction / torch.linalg.vector_norm(torch.tensor(direction)).item()

    assert float(torch.tensor(orthogonal).norm()) == pytest.approx(1.0)
    assert float(torch.dot(torch.tensor(orthogonal), torch.tensor(unit))) == pytest.approx(0.0, abs=1e-6)


def test_select_balanced_stage1_rows_uses_test_split(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "rows.jsonl"
    splits_path = tmp_path / "splits.jsonl"
    source_file = "results/full/with_errortype/gemma3_27b_infer_property.jsonl"
    rows = []
    assignments = []
    for idx in range(32):
        height = 3 if idx < 16 else 4
        label = bool((idx // 2) % 2)
        split = "test" if idx % 2 == 0 else "train"
        rows.append(
            {
                "example_id": f"ex{idx}",
                "height": height,
                "is_correct_strong": label,
                "parse_failed": False,
            }
        )
        assignments.append(
            {
                "source_file": source_file,
                "row_index": idx,
                "s1_split": split,
            }
        )
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with splits_path.open("w") as f:
        for row in assignments:
            f.write(json.dumps(row) + "\n")

    selected, summary = select_balanced_stage1_rows(
        jsonl_path=jsonl_path,
        splits_path=splits_path,
        source_file=source_file,
        split_family="s1",
        heights=[3, 4],
        per_height_label=2,
        seed=123,
    )

    assert len(selected) == 8
    assert {row["row_index"] % 2 for row in selected} == {0}
    assert summary["selected_counts"] == {
        "h3_incorrect": 2,
        "h3_correct": 2,
        "h4_incorrect": 2,
        "h4_correct": 2,
    }


def test_train_raw_probe_direction_recovers_input_space_direction(tmp_path: Path) -> None:
    activation_path = tmp_path / "acts.safetensors"
    sidecar_path = tmp_path / "acts.example_ids.jsonl"
    splits_path = tmp_path / "splits.jsonl"
    source_file = "results/full/with_errortype/gemma3_27b_infer_property.jsonl"
    labels = [0, 1] * 50
    activations = []
    sidecar_rows = []
    split_rows = []
    for idx, label in enumerate(labels):
        split = "train" if idx < 70 else "val" if idx < 84 else "test"
        x0 = float(label) * 2.0 - 1.0
        activations.append([x0, 0.1 * (idx % 3), -x0])
        sidecar_rows.append(
            {
                "row_index": idx,
                "example_id": f"ex{idx}",
                "height": 3,
                "task": "infer_property",
                "model": "gemma3-27b",
                "is_correct_strong": bool(label),
                "parse_failed": False,
            }
        )
        split_rows.append(
            {
                "source_file": source_file,
                "row_index": idx,
                "s1_split": split,
            }
        )
    save_file({"activations": torch.tensor(activations, dtype=torch.float32)}, activation_path)
    with sidecar_path.open("w") as f:
        for row in sidecar_rows:
            f.write(json.dumps(row) + "\n")
    with splits_path.open("w") as f:
        for row in split_rows:
            f.write(json.dumps(row) + "\n")

    direction = train_raw_probe_direction(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        splits_path=splits_path,
        source_file=source_file,
        split_family="s1",
        seed=123,
        c_values=(1.0,),
        max_iter=500,
    )

    assert direction["val_auc"] == 1.0
    assert direction["test_auc"] == 1.0
    assert direction["unit_direction"][0] > 0
    assert direction["unit_direction"][2] < 0
    assert float(torch.tensor(direction["unit_direction"]).norm()) == pytest.approx(1.0)


def test_summarize_steering_rows_counts_flips_vs_baseline() -> None:
    rows = [
        {
            "source_row_index": 1,
            "condition": "baseline",
            "height": 3,
            "original_is_correct_strong": False,
            "is_correct_strong": False,
            "is_correct_weak": False,
            "parse_failed": False,
            "quality_score": 0.0,
            "generated_token_count": 5,
            "model_output": "A is B",
        },
        {
            "source_row_index": 2,
            "condition": "baseline",
            "height": 3,
            "original_is_correct_strong": True,
            "is_correct_strong": True,
            "is_correct_weak": True,
            "parse_failed": False,
            "quality_score": 1.0,
            "generated_token_count": 5,
            "model_output": "A is B",
        },
        {
            "source_row_index": 1,
            "condition": "raw_pos2sd",
            "height": 3,
            "original_is_correct_strong": False,
            "is_correct_strong": True,
            "is_correct_weak": True,
            "parse_failed": False,
            "quality_score": 1.0,
            "generated_token_count": 7,
            "model_output": "A is B",
        },
        {
            "source_row_index": 2,
            "condition": "raw_pos2sd",
            "height": 3,
            "original_is_correct_strong": True,
            "is_correct_strong": True,
            "is_correct_weak": True,
            "parse_failed": False,
            "quality_score": 1.0,
            "generated_token_count": 6,
            "model_output": "A is B",
        },
    ]

    summary = summarize_steering_rows(rows)

    assert summary["by_condition"]["baseline"]["strong_accuracy"] == 0.5
    assert summary["by_condition"]["raw_pos2sd"]["strong_accuracy"] == 1.0
    assert summary["flips_vs_baseline"]["raw_pos2sd"] == {
        "paired_n": 2,
        "false_to_true": 1,
        "true_to_false": 0,
        "changed": 1,
        "net_accuracy_delta": 0.5,
    }
