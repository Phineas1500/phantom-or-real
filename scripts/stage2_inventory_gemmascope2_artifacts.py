#!/usr/bin/env python3
"""Inventory Gemma Scope 2 27B sparse-dictionary artifacts on Hugging Face."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import list_repo_files


TARGET_FAMILIES = {
    "resid_post_all",
    "mlp_out_all",
    "transcoder",
    "transcoder_all",
    "crosscoder",
}


def width_to_int(width: str | None) -> int | None:
    if width is None:
        return None
    suffix = width[-1].lower()
    if suffix == "k":
        return int(float(width[:-1]) * 1_000)
    if suffix == "m":
        return int(float(width[:-1]) * 1_000_000)
    if width.isdigit():
        return int(width)
    return None


def parse_artifact_id(artifact_id: str) -> dict[str, Any]:
    affine = artifact_id.endswith("_affine")
    base = artifact_id[: -len("_affine")] if affine else artifact_id
    parts = base.split("_")

    parsed: dict[str, Any] = {
        "layers": None,
        "width": None,
        "width_int": None,
        "l0": None,
        "affine": affine,
    }
    if not parts or parts[0] != "layer" or "width" not in parts or "l0" not in parts:
        return parsed

    width_idx = parts.index("width")
    l0_idx = parts.index("l0")
    if width_idx <= 1 or l0_idx <= width_idx + 1:
        return parsed

    layer_tokens = parts[1:width_idx]
    if all(token.isdigit() for token in layer_tokens):
        parsed["layers"] = [int(token) for token in layer_tokens]
    parsed["width"] = parts[width_idx + 1]
    parsed["width_int"] = width_to_int(parsed["width"])
    parsed["l0"] = "_".join(parts[l0_idx + 1 :]) or None
    return parsed


def local_snapshot_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    if (path / "snapshots").is_dir():
        snapshots = sorted((path / "snapshots").iterdir())
        if snapshots:
            return snapshots[-1]
    return path


def artifact_record(family: str, artifact_id: str, files: list[str], snapshot: Path | None) -> dict[str, Any]:
    parsed = parse_artifact_id(artifact_id)
    local_path = snapshot / family / artifact_id if snapshot is not None else None
    return {
        "family": family,
        "artifact_id": artifact_id,
        **parsed,
        "file_count": len(files),
        "files": sorted(files),
        "has_local_cache": bool(local_path and local_path.exists()),
    }


def artifact_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    layers = record.get("layers")
    first_layer = layers[0] if layers else 10_000
    return (
        record["family"],
        first_layer,
        len(layers or []),
        record.get("width_int") or 0,
        str(record.get("l0") or ""),
        bool(record.get("affine")),
        record["artifact_id"],
    )


def recommendation(record: dict[str, Any], *, already_run: set[str]) -> str:
    artifact_id = record["artifact_id"]
    family = record["family"]
    layers = record.get("layers") or []
    l0 = record.get("l0")
    affine = bool(record.get("affine"))
    width_int = record.get("width_int") or 0

    if artifact_id in already_run:
        return "already run"
    if family in {"transcoder", "transcoder_all"} and layers == [45] and affine:
        if l0 != "small" or width_int > 262_000:
            return "highest-priority exact transcoder candidate"
        return "clean exact transcoder, but not denser than completed runs"
    if family == "transcoder" and layers in ([40], [53]) and affine:
        if l0 in {"medium", "big"}:
            return "selected-layer higher-L0 exact transcoder candidate"
        return "selected-layer exact transcoder candidate"
    if family == "crosscoder":
        return "optional multi-layer sparse candidate; compare to raw concat"
    if family == "mlp_out_all" and layers == [45]:
        return "optional MLP-output SAE follow-up"
    return ""


def build_inventory(repo_id: str, *, local_cache_root: Path | None) -> dict[str, Any]:
    files = list_repo_files(repo_id)
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    other_transcoder_paths = []

    for file_path in files:
        parts = file_path.split("/")
        if len(parts) < 2:
            continue
        family, artifact_id = parts[0], parts[1]
        if family in TARGET_FAMILIES:
            grouped[family][artifact_id].append("/".join(parts[2:]))
        elif "transcoder" in file_path or "crosscoder" in file_path:
            other_transcoder_paths.append(file_path)

    snapshot = local_snapshot_root(local_cache_root)
    families: dict[str, Any] = {}
    all_records = []
    for family in sorted(grouped):
        records = [
            artifact_record(family, artifact_id, files, snapshot)
            for artifact_id, files in grouped[family].items()
        ]
        records.sort(key=artifact_sort_key)
        families[family] = {
            "artifact_count": len(records),
            "artifacts": records,
        }
        all_records.extend(records)

    already_run = {
        "layer_45_width_16k_l0_small_affine",
        "layer_45_width_262k_l0_small_affine",
    }
    candidates = []
    for record in sorted(all_records, key=artifact_sort_key):
        note = recommendation(record, already_run=already_run)
        if note:
            candidates.append({**record, "recommendation": note})

    l45_affine_transcoders = [
        record
        for record in all_records
        if record["family"] in {"transcoder", "transcoder_all"}
        and record.get("layers") == [45]
        and record.get("affine")
    ]
    unexplored_l45_affine = [
        record
        for record in l45_affine_transcoders
        if record["artifact_id"] not in already_run
        and ((record.get("l0") != "small") or ((record.get("width_int") or 0) > 262_000))
    ]

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "local_cache_root": str(local_cache_root) if local_cache_root else None,
        "local_snapshot_root": str(snapshot) if snapshot else None,
        "total_repo_files": len(files),
        "families": families,
        "candidates": candidates,
        "summary": {
            "l45_affine_transcoder_count": len(l45_affine_transcoders),
            "unexplored_higher_l0_or_denser_l45_affine_transcoder_count": len(unexplored_l45_affine),
            "other_transcoder_path_count": len(other_transcoder_paths),
        },
        "other_transcoder_paths": sorted(other_transcoder_paths),
    }


def table_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"


def write_markdown(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Gemma Scope 2 27B Artifact Inventory",
        "",
        f"- Repository: `{inventory['repo_id']}`",
        f"- Generated: `{inventory['created_at_utc']}`",
        f"- Total repo files listed: `{inventory['total_repo_files']}`",
        "",
        "## Family Counts",
        "",
        "| Family | Artifacts | Locally cached |",
        "| --- | ---: | ---: |",
    ]
    for family, payload in inventory["families"].items():
        cached = sum(1 for record in payload["artifacts"] if record["has_local_cache"])
        lines.append(table_row([f"`{family}`", str(payload["artifact_count"]), str(cached)]))

    lines.extend(
        [
            "",
            "## Candidate Artifacts",
            "",
            "| Family | Artifact | Layers | Width | L0 | Affine | Cached | Recommendation |",
            "| --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for record in inventory["candidates"]:
        lines.append(
            table_row(
                [
                    f"`{record['family']}`",
                    f"`{record['artifact_id']}`",
                    ",".join(str(layer) for layer in (record.get("layers") or [])),
                    str(record.get("width") or ""),
                    str(record.get("l0") or ""),
                    "yes" if record.get("affine") else "no",
                    "yes" if record.get("has_local_cache") else "no",
                    record["recommendation"],
                ]
            )
        )

    unexplored_count = inventory["summary"][
        "unexplored_higher_l0_or_denser_l45_affine_transcoder_count"
    ]
    lines.extend(["", "## Interpretation", ""])
    if unexplored_count == 0:
        lines.extend(
            [
                "- The Hub listing exposes no unrun higher-L0 or denser single-layer affine L45 transcoder beyond the completed `16k_l0_small_affine` and `262k_l0_small_affine` runs.",
                "- The clean L45 exact-transcoder branch should therefore stop here unless a different layer/site is chosen deliberately.",
                "- If we still want one more learned-dictionary AUC attempt, the only catalog-visible sparse extension is the multi-layer crosscoder `l0_big`, which should be compared to the existing raw-concat baseline rather than treated as a direct L45 transcoder replacement.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Found `{unexplored_count}` unrun higher-L0 or denser single-layer affine L45 transcoder candidate(s).",
                "- Run the smallest such exact candidate first, with the existing hook-audit check before probing.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="google/gemma-scope-2-27b-it")
    parser.add_argument(
        "--local-cache-root",
        type=Path,
        default=Path("/scratch/scholar/skiron/hf-cache/hub/models--google--gemma-scope-2-27b-it"),
    )
    parser.add_argument("--output-json", type=Path, default=Path("docs/gemmascope2_artifact_inventory_27b.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/gemmascope2_artifact_inventory_27b.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_inventory(args.repo_id, local_cache_root=args.local_cache_root)
    write_json(args.output_json, inventory)
    write_markdown(args.output_md, inventory)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
