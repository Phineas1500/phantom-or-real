from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.activations import (  # noqa: E402
    EncodedExample,
    hook_name_for_layer,
    make_padded_batch,
    parse_int_list,
    read_stage1_rows,
    slugify_model_name,
    write_activation_outputs,
)
from src.messages import build_messages  # noqa: E402


def test_build_messages_matches_gemma_stage1_contract() -> None:
    assert build_messages("sys", "user", "google/gemma-3-27b-it") == [
        {"role": "user", "content": "sys\n\nuser"}
    ]
    assert build_messages("sys", "user", "other-model") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_make_padded_batch_tracks_last_unpadded_position() -> None:
    examples = [
        EncodedExample(0, "a", 1, "infer_property", "m", True, False, [10, 11]),
        EncodedExample(1, "b", 2, "infer_property", "m", False, False, [12]),
    ]

    tokens, attention_mask, last_positions = make_padded_batch(
        examples,
        pad_token_id=0,
        device="cpu",
    )

    assert tokens.tolist() == [[10, 11], [12, 0]]
    assert attention_mask.tolist() == [[1, 1], [1, 0]]
    assert last_positions == [1, 0]


def test_read_stage1_rows_filters_height_parse_fail_skip_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {"height": 1, "parse_failed": False, "example_id": "keep0"},
        {"height": 2, "parse_failed": False, "example_id": "wrong_height"},
        {"height": 1, "parse_failed": True, "example_id": "parse_failed"},
        {"height": 1, "parse_failed": False, "example_id": "skip_me"},
        {"height": 1, "parse_failed": False, "example_id": "keep1"},
    ]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    selected = read_stage1_rows(path, height=1, drop_parse_failed=True, skip=1, limit=1)

    assert selected == [(3, {"height": 1, "parse_failed": False, "example_id": "skip_me"})]


def test_write_activation_outputs_creates_safetensors_sidecar_and_meta(tmp_path: Path) -> None:
    activations = {30: torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)}
    sidecar_rows = [
        {"row_index": 0, "example_id": "ex0", "token_count": 5},
        {"row_index": 1, "example_id": "ex1", "token_count": 6},
    ]

    written = write_activation_outputs(
        activations,
        sidecar_rows,
        out_dir=tmp_path,
        model_key="gemma3_27b",
        task="infer_property",
        metadata={"row_count": 2},
    )

    prefix = tmp_path / "gemma3_27b_infer_property_L30"
    assert set(written) == {
        prefix.with_suffix(".safetensors"),
        prefix.with_suffix(".example_ids.jsonl"),
        prefix.with_suffix(".meta.json"),
    }
    loaded = load_file(prefix.with_suffix(".safetensors"))["activations"]
    assert tuple(loaded.shape) == (2, 3)
    assert loaded.dtype == torch.bfloat16

    sidecar = [json.loads(line) for line in prefix.with_suffix(".example_ids.jsonl").read_text().splitlines()]
    assert [row["example_id"] for row in sidecar] == ["ex0", "ex1"]
    assert {row["hook_name"] for row in sidecar} == {"blocks.30.hook_resid_post"}

    meta = json.loads(prefix.with_suffix(".meta.json").read_text())
    assert meta["shape"] == [2, 3]
    assert meta["hook_name"] == "blocks.30.hook_resid_post"


def test_write_activation_outputs_can_use_named_non_residual_site(tmp_path: Path) -> None:
    activations = {45: torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)}
    sidecar_rows = [
        {"row_index": 0, "example_id": "ex0", "token_count": 5},
        {"row_index": 1, "example_id": "ex1", "token_count": 6},
    ]

    written = write_activation_outputs(
        activations,
        sidecar_rows,
        out_dir=tmp_path,
        model_key="gemma3_27b",
        task="infer_property",
        metadata={"row_count": 2},
        activation_site="mlp-out",
        hook_template="blocks.{layer}.ln2_post.hook_normalized",
    )

    prefix = tmp_path / "gemma3_27b_infer_property_L45_mlp_out"
    assert set(written) == {
        prefix.with_suffix(".safetensors"),
        prefix.with_suffix(".example_ids.jsonl"),
        prefix.with_suffix(".meta.json"),
    }

    sidecar = [json.loads(line) for line in prefix.with_suffix(".example_ids.jsonl").read_text().splitlines()]
    assert {row["hook_name"] for row in sidecar} == {"blocks.45.ln2_post.hook_normalized"}

    meta = json.loads(prefix.with_suffix(".meta.json").read_text())
    assert meta["activation_site"] == "mlp_out"
    assert meta["hook_name"] == "blocks.45.ln2_post.hook_normalized"
    assert meta["hook_template"] == "blocks.{layer}.ln2_post.hook_normalized"


def test_parse_layers_and_model_slug() -> None:
    assert parse_int_list("4, 12,30") == [4, 12, 30]
    assert slugify_model_name("google/gemma-3-27b-it") == "gemma3_27b"


def test_hook_template_requires_layer_placeholder() -> None:
    assert hook_name_for_layer(layer=45, hook_template="blocks.{layer}.hook_mlp_out") == "blocks.45.hook_mlp_out"
