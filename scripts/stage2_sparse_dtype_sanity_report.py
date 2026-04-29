#!/usr/bin/env python3
"""Compare cached bfloat16 sparse encodings against float32 sample encodings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class FeaturePair:
    name: str
    method: str
    task: str
    bf16_prefix: Path
    fp32_prefix: Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_meta(prefix: Path) -> dict:
    path = prefix.with_suffix(".meta.json")
    with path.open() as f:
        return json.load(f)


def active_ids(indices: torch.Tensor, values: torch.Tensor) -> list[set[int]]:
    rows: list[set[int]] = []
    for idx_row, val_row in zip(indices, values, strict=True):
        mask = val_row > 0
        rows.append(set(int(v) for v in idx_row[mask].tolist()))
    return rows


def feature_prefixes(root: Path, dtype_root: Path) -> list[FeaturePair]:
    crosscoder_id = "layer_16_31_40_53_width_65k_l0_medium"
    specs = [
        (
            "residual_sae_16k",
            "residual_sae",
            "infer_property",
            root / "sae_features/gemma3_27b_infer_property_L45_layer_45_width_16k_l0_small_top128",
            dtype_root / "gemma3_27b_infer_property_L45_layer_45_width_16k_l0_small_top128_n512",
        ),
        (
            "residual_sae_16k",
            "residual_sae",
            "infer_subtype",
            root / "sae_features/gemma3_27b_infer_subtype_L45_layer_45_width_16k_l0_small_top128",
            dtype_root / "gemma3_27b_infer_subtype_L45_layer_45_width_16k_l0_small_top128_n512",
        ),
        (
            "mlp_out_sae_16k",
            "mlp_out_sae",
            "infer_property",
            root / "sae_features/gemma3_27b_infer_property_L45_mlp_out_layer_45_width_16k_l0_small_top128",
            dtype_root / "gemma3_27b_infer_property_L45_mlp_out_layer_45_width_16k_l0_small_top128_n512",
        ),
        (
            "mlp_out_sae_16k",
            "mlp_out_sae",
            "infer_subtype",
            root / "sae_features/gemma3_27b_infer_subtype_L45_mlp_out_layer_45_width_16k_l0_small_top128",
            dtype_root / "gemma3_27b_infer_subtype_L45_mlp_out_layer_45_width_16k_l0_small_top128_n512",
        ),
        (
            "skip_transcoder_16k_affine",
            "skip_transcoder",
            "infer_property",
            root
            / "sae_features/gemma3_27b_infer_property_L45_mlp_in_layer_45_width_16k_l0_small_affine_top128",
            dtype_root
            / "gemma3_27b_infer_property_L45_mlp_in_layer_45_width_16k_l0_small_affine_top128_n512",
        ),
        (
            "skip_transcoder_16k_affine",
            "skip_transcoder",
            "infer_subtype",
            root
            / "sae_features/gemma3_27b_infer_subtype_L45_mlp_in_layer_45_width_16k_l0_small_affine_top128",
            dtype_root
            / "gemma3_27b_infer_subtype_L45_mlp_in_layer_45_width_16k_l0_small_affine_top128_n512",
        ),
        (
            "crosscoder_65k",
            "crosscoder",
            "infer_property",
            root / f"crosscoder_features/gemma3_27b_infer_property_crosscoder_{crosscoder_id}_top128",
            dtype_root / f"gemma3_27b_infer_property_crosscoder_{crosscoder_id}_top128_n512",
        ),
        (
            "crosscoder_65k",
            "crosscoder",
            "infer_subtype",
            root / f"crosscoder_features/gemma3_27b_infer_subtype_crosscoder_{crosscoder_id}_top128",
            dtype_root / f"gemma3_27b_infer_subtype_crosscoder_{crosscoder_id}_top128_n512",
        ),
    ]
    return [FeaturePair(*spec) for spec in specs]


def summarize_pair(pair: FeaturePair) -> dict:
    bf16 = load_file(pair.bf16_prefix.with_suffix(".safetensors"))
    fp32 = load_file(pair.fp32_prefix.with_suffix(".safetensors"))
    row_count = int(fp32["top_values"].shape[0])
    if row_count <= 0:
        raise ValueError(f"{pair.fp32_prefix} has no rows")

    bf16_values = bf16["top_values"][:row_count]
    bf16_indices = bf16["top_indices"][:row_count]
    bf16_l0 = bf16["l0"][:row_count].to(torch.int32)
    fp32_values = fp32["top_values"]
    fp32_indices = fp32["top_indices"]
    fp32_l0 = fp32["l0"].to(torch.int32)

    if bf16_values.shape != fp32_values.shape:
        raise ValueError(f"{pair.name}/{pair.task} top-value shape mismatch: {bf16_values.shape} vs {fp32_values.shape}")
    if bf16_indices.shape != fp32_indices.shape:
        raise ValueError(f"{pair.name}/{pair.task} top-index shape mismatch: {bf16_indices.shape} vs {fp32_indices.shape}")

    bf16_rows = read_jsonl(pair.bf16_prefix.with_suffix(".example_ids.jsonl"))[:row_count]
    fp32_rows = read_jsonl(pair.fp32_prefix.with_suffix(".example_ids.jsonl"))
    for row_idx, (left, right) in enumerate(zip(bf16_rows, fp32_rows, strict=True)):
        if left.get("example_id") != right.get("example_id") or left.get("task") != right.get("task"):
            raise ValueError(f"{pair.name}/{pair.task} sidecar mismatch at row {row_idx}")

    bf16_active = active_ids(bf16_indices, bf16_values)
    fp32_active = active_ids(fp32_indices, fp32_values)
    jaccards = []
    recall_vs_bf16 = []
    recall_vs_fp32 = []
    for left, right in zip(bf16_active, fp32_active, strict=True):
        union = left | right
        intersection = left & right
        jaccards.append(1.0 if not union else len(intersection) / len(union))
        recall_vs_bf16.append(1.0 if not left else len(intersection) / len(left))
        recall_vs_fp32.append(1.0 if not right else len(intersection) / len(right))

    l0_diff = (bf16_l0 - fp32_l0).abs().float()
    return {
        "name": pair.name,
        "method": pair.method,
        "task": pair.task,
        "rows": row_count,
        "bf16_feature_file": str(pair.bf16_prefix.with_suffix(".safetensors")),
        "fp32_feature_file": str(pair.fp32_prefix.with_suffix(".safetensors")),
        "bf16_dtype": load_meta(pair.bf16_prefix).get("dtype"),
        "fp32_dtype": load_meta(pair.fp32_prefix).get("dtype"),
        "bf16_l0_mean": float(bf16_l0.float().mean().item()),
        "fp32_l0_mean": float(fp32_l0.float().mean().item()),
        "bf16_l0_max": int(bf16_l0.max().item()),
        "fp32_l0_max": int(fp32_l0.max().item()),
        "l0_abs_diff_mean": float(l0_diff.mean().item()),
        "l0_diff_row_fraction": float((l0_diff > 0).float().mean().item()),
        "active_jaccard_mean": float(torch.tensor(jaccards).mean().item()),
        "active_recall_vs_bf16_mean": float(torch.tensor(recall_vs_bf16).mean().item()),
        "active_recall_vs_fp32_mean": float(torch.tensor(recall_vs_fp32).mean().item()),
        "top1_index_match_rate": float((bf16_indices[:, 0] == fp32_indices[:, 0]).float().mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, default=Path("results/stage2"))
    parser.add_argument("--dtype-root", type=Path, default=Path("results/stage2/dtype_sanity_features"))
    parser.add_argument("--output", type=Path, default=Path("docs/sparse_dtype_sanity_27b.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [summarize_pair(pair) for pair in feature_prefixes(args.stage2_root, args.dtype_root)]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Compares existing bfloat16 sparse encodings with float32 re-encodings of the first 512 rows.",
        "results": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
