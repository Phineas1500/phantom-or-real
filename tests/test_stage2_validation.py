from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.activations import EncodedExample, sha256_file  # noqa: E402
from src.stage2_validation import (  # noqa: E402
    compare_sidecar_rows,
    validate_activation_artifact,
    validate_stage1_jsonl,
)


def _report() -> dict:
    return {"checks": [], "errors": [], "warnings": []}


def test_validate_stage1_jsonl_checks_hash_and_rows(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "rows.jsonl"
    rows = [{"a": 1}, {"b": 2}]
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    invariants = {
        "stage1_jsonls": {
            str(jsonl_path.resolve()): {
                "sha256": sha256_file(jsonl_path),
                "rows": 2,
            }
        }
    }

    report = _report()
    validate_stage1_jsonl(report, jsonl_path=jsonl_path, invariants=invariants)

    assert report["checks"][0]["status"] == "ok"
    assert not report["errors"]


def test_compare_sidecar_rows_detects_token_count_mismatch() -> None:
    example = EncodedExample(0, "ex0", 2, "infer_property", "gemma3-27b", True, False, [10, 11])
    sidecar = [
        {
            "row_index": 0,
            "example_id": "ex0",
            "height": 2,
            "task": "infer_property",
            "model": "gemma3-27b",
            "is_correct_strong": True,
            "parse_failed": False,
            "token_count": 99,
            "last_token_position": 1,
            "hook_name": "blocks.30.hook_resid_post",
        }
    ]

    comparison = compare_sidecar_rows(sidecar_rows=sidecar, examples=[example], layer=30)

    assert comparison["mismatch_count_sampled"] == 1
    assert comparison["mismatches"][0]["field"] == "token_count"


def test_validate_activation_artifact_checks_shape_meta_and_sidecar(tmp_path: Path) -> None:
    prefix = tmp_path / "gemma3_27b_infer_property_L30"
    jsonl_sha = "abc123"
    examples = [
        EncodedExample(0, "ex0", 1, "infer_property", "gemma3-27b", True, False, [10, 11]),
        EncodedExample(1, "ex1", 1, "infer_property", "gemma3-27b", False, False, [12]),
    ]
    activation = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)
    save_file({"activations": activation}, prefix.with_suffix(".safetensors"))
    with prefix.with_suffix(".example_ids.jsonl").open("w") as f:
        for example in examples:
            f.write(
                json.dumps(
                    {
                        "row_index": example.row_index,
                        "example_id": example.example_id,
                        "height": example.height,
                        "task": example.task,
                        "model": example.model,
                        "is_correct_strong": example.is_correct_strong,
                        "parse_failed": example.parse_failed,
                        "token_count": example.token_count,
                        "last_token_position": example.token_count - 1,
                        "hook_name": "blocks.30.hook_resid_post",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    with prefix.with_suffix(".meta.json").open("w") as f:
        json.dump(
            {
                "shape": [2, 3],
                "dtype": "torch.bfloat16",
                "row_count": 2,
                "jsonl_sha256": jsonl_sha,
                "hook_name": "blocks.30.hook_resid_post",
            },
            f,
        )

    report = _report()
    validate_activation_artifact(
        report,
        prefix=prefix,
        layer=30,
        jsonl_sha256=jsonl_sha,
        examples=examples,
    )

    assert report["checks"][0]["status"] == "ok"
    assert not report["errors"]
