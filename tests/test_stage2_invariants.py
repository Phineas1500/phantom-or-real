from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.stage2_write_invariants import (  # noqa: E402
    build_invariants,
    discover_cached_snapshot,
    hf_cache_dir_for_transformers,
    sha256_file,
    sha256_text,
)


def test_hash_helpers(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("abc")
    assert sha256_text("abc") == sha256_file(path)


def test_discover_cached_snapshot_prefers_refs_main(tmp_path: Path) -> None:
    cache = tmp_path / "hf-cache"
    repo = cache / "hub" / "models--google--gemma-3-27b-it"
    snapshots = repo / "snapshots"
    commit_a = "a" * 40
    commit_b = "b" * 40
    (snapshots / commit_a).mkdir(parents=True)
    (snapshots / commit_b).mkdir()
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit_a)

    snapshot = discover_cached_snapshot("google/gemma-3-27b-it", cache)

    assert snapshot.refs_main == commit_a
    assert snapshot.selected_commit == commit_a
    assert snapshot.available_snapshots == [commit_a, commit_b]


def test_transformers_cache_dir_points_at_hub(tmp_path: Path) -> None:
    cache = tmp_path / "hf-cache"
    assert hf_cache_dir_for_transformers(cache) == str(cache / "hub")
    assert hf_cache_dir_for_transformers(cache / "hub") == str(cache / "hub")


def test_discover_cached_snapshot_falls_back_to_newest_snapshot(tmp_path: Path) -> None:
    cache = tmp_path / "hf-cache"
    repo = cache / "hub" / "models--google--gemma-3-4b-it"
    snapshots = repo / "snapshots"
    commit_a = "a" * 40
    commit_b = "b" * 40
    path_a = snapshots / commit_a
    path_b = snapshots / commit_b
    path_a.mkdir(parents=True)
    path_b.mkdir()
    os.utime(path_a, (1, 1))
    os.utime(path_b, (2, 2))

    snapshot = discover_cached_snapshot("google/gemma-3-4b-it", cache)

    assert snapshot.refs_main is None
    assert snapshot.selected_commit == commit_b


def test_build_invariants_hashes_jsonls_without_models(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonls"
    jsonl_dir.mkdir()
    row_path = jsonl_dir / "gemma3_27b_infer_property.jsonl"
    row_path.write_text(json.dumps({"example_id": "x"}) + "\n")
    output = tmp_path / "stage2_invariants.json"

    invariants = build_invariants(
        jsonl_dir=jsonl_dir,
        models=[],
        output=output,
        hf_cache=tmp_path / "hf-cache",
        local_files_only=True,
        judge_snapshot=None,
    )

    stage1 = invariants["stage1_jsonls"]
    assert len(stage1) == 1
    only = next(iter(stage1.values()))
    assert only["rows"] == 1
    assert only["sha256"] == sha256_file(row_path)
    assert invariants["chat_template_probe"]["message_builder"] == "src.messages.build_messages"
