from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stage2_phase0 import (  # noqa: E402
    attach_splits,
    build_inventory,
    canonical_topology_hash,
    load_stage1_records,
    make_split_assignments,
    row_names,
    summarize_split_assignments,
    write_jsonl,
)


def _row(
    *,
    idx: int,
    model: str = "gemma3-4b",
    task: str = "infer_property",
    height: int = 2,
    correct: bool = False,
    parse_failed: bool = False,
    child: str = "dax",
    parent: str = "wug",
) -> dict:
    return {
        "example_id": f"ex{idx}",
        "task": task,
        "height": height,
        "model": model,
        "prompt_text": f"Q: Alice is a {child}. {child}s are {parent}s.",
        "system_prompt": "system",
        "ground_truth": f"{parent}s are quiet",
        "model_output": "",
        "is_correct_strong": correct,
        "is_correct_weak": correct,
        "quality_score": 1.0 if correct else 0.0,
        "parse_failed": parse_failed,
        "failure_mode": "parse_error" if parse_failed else None,
        "error_type": None,
        "structural": {
            "target_concept": parent,
            "has_direct_member": True,
            "num_direct_paths": 1,
            "parent_salience": 4,
            "num_theories_axioms": 2,
            "num_observations": 1,
            "tree_height": height,
        },
        "ontology_fol_structured": {
            "membership": {"alice": [child]},
            "inheritance": {child: [parent]},
            "properties": {},
            "negated_properties": {},
            "hypothesis": {"type": "rule", "subject": parent, "predicate": "quiet", "negated": False},
        },
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_canonical_topology_hash_ignores_names() -> None:
    row_a = _row(idx=0, child="dax", parent="wug")
    row_b = _row(idx=1, child="fep", parent="zorp")

    assert canonical_topology_hash(row_a) == canonical_topology_hash(row_b)


def test_row_names_collects_symbols() -> None:
    names = row_names(_row(idx=0, child="dax", parent="wug"))

    assert {"alice", "dax", "wug", "quiet"} <= names


def test_inventory_flags_low_classes_and_parse_warnings(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [_row(idx=i, correct=(i == 0), parse_failed=(i in {2, 3})) for i in range(10)]
    _write_rows(path, rows)

    inventory = build_inventory([path], low_class_threshold=3)

    warnings = inventory["warnings"]
    assert any(warning["type"] == "high_parse_failure_rate" for warning in warnings)
    assert any(
        warning["type"] == "low_non_parse_class_count" and warning["label"] == "positive"
        for warning in warnings
    )


def test_split_assignments_cover_every_row_and_attach(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        _row(idx=i, correct=(i % 3 == 0), child=f"dax{i}", parent=f"wug{i // 2}")
        for i in range(30)
    ]
    _write_rows(path, rows)
    records = load_stage1_records([path])

    assignments = make_split_assignments(records, seed=123)
    summary = summarize_split_assignments(assignments)

    assert len(assignments) == len(rows)
    assert summary["total_rows"] == len(rows)
    assert {row["s1_split"] for row in assignments} == {"train", "val", "test"}
    assert {row["s2_split"] for row in assignments} <= {"train", "val", "test"}
    assert summary["families"]["s1"]["gemma3-4b__infer_property"]["h2"]["is_evaluable"]
    assert not summary["families"]["s2"]["gemma3-4b__infer_property"]["h2"]["is_evaluable"]
    assert any(
        warning["type"] == "split_not_evaluable" and warning["family"] == "s2"
        for warning in summary["warnings"]
    )

    split_path = tmp_path / "splits.jsonl"
    write_jsonl(split_path, assignments)
    attach_splits(records, {(row["source_file"], row["row_index"]): row for row in assignments})
    assert all(record["s1_split"] for record in records)
