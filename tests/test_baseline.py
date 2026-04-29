"""Tests for src/baselines.py — focused on the namefreq no-leakage guarantee."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import _build_xy, _namefreq_lookup, train_and_evaluate


def _row(
    eid: str,
    height: int,
    task: str,
    concept: str,
    correct: bool,
    parse_failed: bool = False,
    system_prompt: str = "sys",
    prompt_text: str = "user",
) -> dict:
    return {
        "example_id": eid,
        "task": task,
        "height": height,
        "parse_failed": parse_failed,
        "is_correct_strong": correct,
        "system_prompt": system_prompt,
        "prompt_text": prompt_text,
        "model_output": "",
        "structural": {
            "target_concept": concept,
            "num_theories_axioms": 3,
            "num_observations": 3,
            "parent_salience": 3,
            "num_direct_paths": 1,
        },
    }


def test_namefreq_no_test_leakage() -> None:
    """Concept names that appear only in test rows must have namefreq == 0."""
    rows = [
        _row("train_0", height=1, task="infer_property", concept="phorpist", correct=True),
        _row("train_1", height=1, task="infer_property", concept="phorpist", correct=True),
        _row("train_2", height=1, task="infer_property", concept="phorpist", correct=False),
        # "glopist" appears ONLY in test:
        _row("test_0", height=1, task="infer_property", concept="glopist", correct=True),
        _row("test_1", height=1, task="infer_property", concept="glopist", correct=False),
    ]
    split_assignment = {
        "train_0": "train", "train_1": "train", "train_2": "train",
        "test_0": "test", "test_1": "test",
    }
    namefreq = _namefreq_lookup(rows, split_assignment)

    # "glopist" should NOT appear in the namefreq lookup (only in test rows).
    assert "glopist" not in namefreq, (
        "namefreq lookup includes a concept that only appears in test rows — leakage!"
    )
    # "phorpist" should appear with correct counts from train only.
    assert namefreq["phorpist"] == (2, 1), (
        f"expected (2, 1) for phorpist, got {namefreq['phorpist']}"
    )


def test_namefreq_features_are_zero_for_test_only_concepts() -> None:
    """b0_namefreq feature vector for a test row with an unseen concept = 0 freq."""
    rows = [
        _row("tr0", 1, "infer_property", "phorpist", True),
        _row("tr1", 1, "infer_property", "phorpist", False),
        _row("te0", 1, "infer_property", "unseen_concept", True),
    ]
    split_assignment = {"tr0": "train", "tr1": "train", "te0": "test"}
    # Fake token counts (not needed for correctness check here but required by the API).
    token_counts = {"tr0": 50, "tr1": 50, "te0": 50}

    namefreq = _namefreq_lookup(rows, split_assignment)
    X, y, ids = _build_xy(
        rows, "b0_namefreq",
        token_counts=token_counts,
        namefreq=namefreq,
    )

    te0_idx = ids.index("te0")
    # namefreq_pos and namefreq_neg are the last two features.
    assert X[te0_idx, -2] == 0.0, "namefreq_pos should be 0 for an unseen test concept"
    assert X[te0_idx, -1] == 0.0, "namefreq_neg should be 0 for an unseen test concept"


def test_b0_height_train_and_evaluate_returns_expected_keys() -> None:
    """train_and_evaluate returns a result dict with all required keys."""
    rng = np.random.default_rng(7)
    n = 120
    rows = []
    for i in range(n):
        h = int(rng.integers(1, 5))
        correct = bool(rng.integers(0, 2))
        part = "train" if i < 70 else ("val" if i < 85 else "test")
        rows.append(_row(f"e{i}", h, "infer_property", "someconcept", correct))

    split_assignment = {f"e{i}": ("train" if i < 70 else ("val" if i < 85 else "test"))
                        for i in range(n)}

    result = train_and_evaluate(rows, "b0_height", split_assignment)

    for key in ("variant", "feature_names", "best_C", "auc", "auc_ci_95",
                "balanced_accuracy", "per_height_auc", "n_train", "n_val", "n_test"):
        assert key in result, f"missing key {key!r} in result"

    assert result["variant"] == "b0_height"
    assert result["n_train"] == 70
    assert result["n_val"] == 15
    assert result["n_test"] == 35