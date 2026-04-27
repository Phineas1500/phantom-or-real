from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.splits import canonical_skeleton_key, make_s1_split, make_s2_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(eid: str, height: int, task: str, inheritance: dict, correct: bool = True,
         parse_failed: bool = False) -> dict:
    return {
        "example_id": eid,
        "task": task,
        "height": height,
        "parse_failed": parse_failed,
        "is_correct_strong": correct,
        "ontology_fol_structured": {"inheritance": inheritance},
    }


def _make_rows(n: int, height: int = 2, task: str = "infer_property",
               inheritance: dict | None = None) -> list[dict]:
    if inheritance is None:
        inheritance = {"child_a": ["root"], "child_b": ["root"]}
    rows = []
    for i in range(n):
        rows.append(_row(
            eid=f"{task}_h{height}_{i:05d}",
            height=height,
            task=task,
            inheritance=inheritance,
            correct=(i % 3 != 0),
        ))
    return rows


# ---------------------------------------------------------------------------
# Test: canonical_skeleton_key topology independence
# ---------------------------------------------------------------------------

def test_skeleton_key_same_topology_different_names() -> None:
    """Two trees with different concept names but identical topology → same key."""
    inh_a = {"cat": ["animal"], "dog": ["animal"]}
    inh_b = {"serpee": ["gergit"], "gorpee": ["gergit"]}
    row_a = _row("a1", height=2, task="infer_property", inheritance=inh_a)
    row_b = _row("b1", height=2, task="infer_property", inheritance=inh_b)
    assert canonical_skeleton_key(row_a) == canonical_skeleton_key(row_b)


def test_skeleton_key_different_topology() -> None:
    """Two trees with different topologies → different keys."""
    inh_chain = {"b": ["a"], "c": ["b"]}      # chain: a→b→c
    inh_star = {"b": ["a"], "c": ["a"]}        # star:  a→{b,c}
    row_chain = _row("c1", height=2, task="infer_property", inheritance=inh_chain)
    row_star = _row("s1", height=2, task="infer_property", inheritance=inh_star)
    assert canonical_skeleton_key(row_chain) != canonical_skeleton_key(row_star)


def test_skeleton_key_flat_h1() -> None:
    """Height-1 flat rows (no inheritance) all share the same key within a task."""
    row1 = _row("e1", height=1, task="infer_property", inheritance={})
    row2 = _row("e2", height=1, task="infer_property", inheritance={})
    assert canonical_skeleton_key(row1) == canonical_skeleton_key(row2)


# ---------------------------------------------------------------------------
# Test: S1 disjointness
# ---------------------------------------------------------------------------

def test_s1_no_example_in_both_train_and_test() -> None:
    rows = _make_rows(300, height=2)
    assignment = make_s1_split(rows, seed=42)
    train_ids = {eid for eid, p in assignment.items() if p == "train"}
    test_ids = {eid for eid, p in assignment.items() if p == "test"}
    assert len(train_ids & test_ids) == 0


def test_s1_parse_failed_excluded() -> None:
    rows = _make_rows(50)
    rows[0]["parse_failed"] = True
    assignment = make_s1_split(rows, seed=42)
    assert assignment[rows[0]["example_id"]] == "excluded"


# ---------------------------------------------------------------------------
# Test: S2 disjointness and group-disjointness
# ---------------------------------------------------------------------------

def test_s2_no_example_in_both_train_and_test() -> None:
    # Build a varied set of topologies so StratifiedGroupKFold has groups to work with.
    rows = []
    topologies = [
        {"b": ["a"], "c": ["a"]},           # star
        {"b": ["a"], "c": ["b"]},           # chain
        {"b": ["a"], "c": ["a"], "d": ["b"]},  # mixed
        {"b": ["a"]},                          # single child
        {"c": ["a"], "d": ["a"], "e": ["a"]},  # wide star
    ]
    for topo_i, inh in enumerate(topologies):
        for j in range(40):
            correct = j % 3 != 0
            rows.append(_row(
                eid=f"topo{topo_i}_h2_{j:04d}",
                height=2,
                task="infer_property",
                inheritance=inh,
                correct=correct,
            ))

    assignment, _ = make_s2_split(rows, seed=42)
    train_ids = {eid for eid, p in assignment.items() if p == "train"}
    test_ids = {eid for eid, p in assignment.items() if p == "test"}
    assert len(train_ids & test_ids) == 0


def test_s2_skeleton_group_disjoint_across_partitions() -> None:
    """No skeleton key should appear in both train and test of S2."""
    rows = []
    topologies = [{"b": ["a"], "c": ["a"]},
                  {"b": ["a"], "c": ["b"]},
                  {"d": ["a"], "e": ["a"], "f": ["a"]},
                  {"b": ["a"]},
                  {"c": ["a"], "d": ["a"], "e": ["a"]},
                  {"g": ["a"], "h": ["g"]}]
    for i, inh in enumerate(topologies):
        for j in range(30):
            rows.append(_row(f"g{i}_{j}", height=2, task="infer_property",
                             inheritance=inh, correct=(j % 2 == 0)))

    assignment, _ = make_s2_split(rows, seed=42)

    train_keys: set[str] = set()
    test_keys: set[str] = set()
    for row in rows:
        part = assignment[row["example_id"]]
        key = canonical_skeleton_key(row)
        if part == "train":
            train_keys.add(key)
        elif part == "test":
            test_keys.add(key)

    assert len(train_keys & test_keys) == 0