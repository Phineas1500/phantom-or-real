from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stage2_probes import run_raw_activation_probe, stratified_split_indices  # noqa: E402


def test_stratified_split_indices_preserves_both_classes() -> None:
    labels = [0] * 20 + [1] * 20

    splits = stratified_split_indices(labels, seed=123)

    assert {labels[idx] for idx in splits["train"]} == {0, 1}
    assert {labels[idx] for idx in splits["val"]} == {0, 1}
    assert {labels[idx] for idx in splits["test"]} == {0, 1}


def test_run_raw_activation_probe_on_separable_synthetic_data(tmp_path: Path) -> None:
    activation_path = tmp_path / "acts.safetensors"
    sidecar_path = tmp_path / "acts.example_ids.jsonl"
    labels = [0, 1] * 40
    rows = []
    activations = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "row_index": idx,
                "example_id": f"ex{idx}",
                "height": 2,
                "task": "infer_property",
                "model": "gemma3-27b",
                "is_correct_strong": bool(label),
                "parse_failed": False,
            }
        )
        activations.append([float(label), float(label) * 2.0, 1.0 - float(label)])
    save_file({"activations": torch.tensor(activations, dtype=torch.float32)}, activation_path)
    with sidecar_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = run_raw_activation_probe(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        seed=123,
    )

    assert result["status"] == "ok"
    assert result["val_auc"] == 1.0
    assert result["test_auc"] == 1.0
    assert result["kept_rows"] == 80
