import json
from pathlib import Path

from scripts.stage2_rank_k_guard_v2 import (
    COMPOSITE_MANIFEST_ROWS,
    build_arms,
    select_fresh_rows,
    shard_rows,
)


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
