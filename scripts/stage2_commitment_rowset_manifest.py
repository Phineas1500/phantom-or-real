#!/usr/bin/env python3
"""Build the canonical row-set manifest for commitment/recognition work."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GEMMA_SOURCE = "results/full/with_errortype/gemma3_27b_infer_property.jsonl"
QWEN_SOURCE = "results/full/with_errortype/qwen35_27b_infer_subtype.jsonl"

SOURCE_SPECS = {
    GEMMA_SOURCE: {"model_key": "gemma3_27b", "task": "infer_property"},
    QWEN_SOURCE: {"model_key": "qwen35_27b", "task": "infer_subtype"},
}

RECOGNITION_SPECS = [
    {
        "artifact_id": "gemma_hardfoil_forced_choice",
        "artifact": "results/stage2/steering/answer_property_margins_27b_l45_polarity_hardfoil.jsonl",
        "model_key": "gemma3_27b",
        "task": "infer_property",
        "foil_type": "stage1_model_output_hard_foil",
        "row_selection": "S1 test, parsed, free-form strong-incorrect, balanced by height and answer polarity",
        "condition": "baseline",
    },
    {
        "artifact_id": "qwen_h4_hardfoil_forced_choice",
        "artifact": "results/stage2/qwen_causal/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.jsonl",
        "model_key": "qwen35_27b",
        "task": "infer_subtype",
        "foil_type": "stage1_model_output_hard_foil",
        "row_selection": "S1 test, parsed, free-form strong-incorrect, first 64 by seed",
        "condition": None,
    },
]

DECODE_TRACE_SPECS = [
    {
        "artifact_id": "gemma_decode_trace_l45_l53_property_pilot",
        "artifact": "results/stage2/decode_time/decode_projection_trace_27b_l45_l53_property_pilot.jsonl",
        "model_key": "gemma3_27b",
        "task": "infer_property",
    },
]

PATCH_SPECS = [
    {
        "artifact_id": "gemma_forward_patching",
        "artifact": "results/stage2/patching/clean_to_corrupt_27b_property_margin_pilot.jsonl",
        "model_key": "gemma3_27b",
        "task": "infer_property",
        "source_file": GEMMA_SOURCE,
        "patch_direction": "h1_to_h4",
    },
    {
        "artifact_id": "gemma_reverse_patching",
        "artifact": "results/stage2/patching/corrupt_to_clean_27b_property_margin_pilot.jsonl",
        "model_key": "gemma3_27b",
        "task": "infer_property",
        "source_file": GEMMA_SOURCE,
        "patch_direction": "h4_to_h1",
    },
    {
        "artifact_id": "qwen_forward_patching",
        "artifact": "results/stage2/patching/qwen35_27b_infer_subtype_clean_to_corrupt_margin_pilot.jsonl",
        "model_key": "qwen35_27b",
        "task": "infer_subtype",
        "source_file": QWEN_SOURCE,
        "patch_direction": "h1_to_h4",
    },
    {
        "artifact_id": "qwen_reverse_patching",
        "artifact": "results/stage2/patching/qwen35_27b_infer_subtype_corrupt_to_clean_margin_pilot.jsonl",
        "model_key": "qwen35_27b",
        "task": "infer_subtype",
        "source_file": QWEN_SOURCE,
        "patch_direction": "h4_to_h1",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_source_rows(repo_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    out = {}
    for source_file in SOURCE_SPECS:
        source_path = repo_root / source_file
        rows = {}
        with source_path.open() as f:
            for idx, line in enumerate(f):
                if line.strip():
                    rows[idx] = json.loads(line)
        out[source_file] = rows
    return out


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def manifest_key(model_key: str, task: str, source_file: str, source_row_index: int) -> tuple[str, str, str, int]:
    return (model_key, task, source_file, source_row_index)


def row_source_summary(source_row: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
    row = source_row or {}
    return {
        "example_id": row.get("example_id", fallback.get("example_id")),
        "height": row.get("height", fallback.get("height")),
        "free_form": {
            "is_correct_strong": row.get("is_correct_strong", fallback.get("original_is_correct_strong")),
            "is_correct_weak": row.get("is_correct_weak", fallback.get("original_is_correct_weak")),
            "parse_failed": row.get("parse_failed", fallback.get("original_parse_failed")),
            "quality_score": row.get("quality_score"),
            "failure_mode": row.get("failure_mode"),
            "error_type": row.get("error_type"),
        },
    }


def get_entry(
    rows: dict[tuple[str, str, str, int], dict[str, Any]],
    source_rows: dict[str, dict[int, dict[str, Any]]],
    *,
    model_key: str,
    task: str,
    source_file: str,
    source_row_index: int,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    key = manifest_key(model_key, task, source_file, source_row_index)
    if key not in rows:
        source_row = source_rows.get(source_file, {}).get(source_row_index)
        source = row_source_summary(source_row, fallback)
        rows[key] = {
            "manifest_id": f"{model_key}:{task}:{source_row_index}",
            "model": model_key,
            "task": task,
            "target_variable": "commitment_state",
            "split": "S1",
            "site_or_layer": "mixed",
            "method": "canonical_commitment_rowset_manifest",
            "representation_type": "mixed",
            "controls": [],
            "n": 1,
            "baseline_metrics": source["free_form"],
            "intervention_metrics": {},
            "paired_flips": {},
            "parse_fail_rate": None,
            "matched_noise_summary": {},
            "causal_abstraction_claim": "row-set manifest only; no new causal claim",
            "source_file": source_file,
            "source_row_index": source_row_index,
            "example_id": source["example_id"],
            "height": source["height"],
            "free_form": source["free_form"],
            "recognition": None,
            "decode_trace": None,
            "patch_membership": [],
            "coverage_flags": {
                "has_recognition": False,
                "has_decode_trace": False,
                "has_patch_pair": False,
                "has_all_three": False,
            },
            "recommended_use": [],
        }
    return rows[key]


def add_unique_recommended_use(entry: dict[str, Any], use: str) -> None:
    if use not in entry["recommended_use"]:
        entry["recommended_use"].append(use)


def add_recognition(
    repo_root: Path,
    rows: dict[tuple[str, str, str, int], dict[str, Any]],
    source_rows: dict[str, dict[int, dict[str, Any]]],
) -> None:
    for spec in RECOGNITION_SPECS:
        for row in read_jsonl(repo_root / spec["artifact"]):
            if spec["condition"] is not None and row.get("condition") != spec["condition"]:
                continue
            source_file = row["source_file"]
            source_row_index = int(row["source_row_index"])
            entry = get_entry(
                rows,
                source_rows,
                model_key=spec["model_key"],
                task=spec["task"],
                source_file=source_file,
                source_row_index=source_row_index,
                fallback=row,
            )
            entry["recognition"] = {
                "available": True,
                "artifact_id": spec["artifact_id"],
                "artifact": spec["artifact"],
                "foil_type": spec["foil_type"],
                "foil_source": row.get("foil_source"),
                "row_selection": spec["row_selection"],
                "height": row.get("height"),
                "gold_hypothesis": row.get("gold_hypothesis"),
                "foil_hypothesis": row.get("foil_hypothesis"),
                "original_margin_gold_minus_foil": row.get("original_margin_gold_minus_foil"),
                "mcq_is_correct_choice": row.get("mcq_is_correct_choice"),
                "mcq_choice_parse_failed": row.get("mcq_choice_parse_failed"),
                "mcq_margin_gold_minus_foil": row.get("mcq_margin_gold_minus_foil"),
            }
            entry["coverage_flags"]["has_recognition"] = True
            if entry["free_form"].get("is_correct_strong") is False and row.get("mcq_is_correct_choice") is True:
                add_unique_recommended_use(entry, "recognition_generation_gap")


def trace_layer_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    prefill = [r["projection_z"] for r in trace if r.get("phase") == "prefill" and "projection_z" in r]
    decode = [r["projection_z"] for r in trace if r.get("phase") != "prefill" and "projection_z" in r]
    return {
        "n_steps": len(trace),
        "prefill_z_mean": mean(prefill),
        "decode_z_mean": mean(decode),
        "decode_z_lt0_fraction": (sum(1 for x in decode if x < 0) / len(decode)) if decode else None,
    }


def add_decode_traces(
    repo_root: Path,
    rows: dict[tuple[str, str, str, int], dict[str, Any]],
    source_rows: dict[str, dict[int, dict[str, Any]]],
) -> None:
    for spec in DECODE_TRACE_SPECS:
        for row in read_jsonl(repo_root / spec["artifact"]):
            source_file = row["source_file"]
            source_row_index = int(row["source_row_index"])
            entry = get_entry(
                rows,
                source_rows,
                model_key=spec["model_key"],
                task=spec["task"],
                source_file=source_file,
                source_row_index=source_row_index,
                fallback=row,
            )
            prompt_sidecar = row.get("prompt_projection_sidecar") or {}
            projection_traces = row.get("projection_traces") or {}
            entry["decode_trace"] = {
                "available": True,
                "artifact_id": spec["artifact_id"],
                "artifact": spec["artifact"],
                "condition": row.get("condition"),
                "layers": row.get("layers"),
                "generated_is_correct_strong": row.get("is_correct_strong"),
                "generated_is_correct_weak": row.get("is_correct_weak"),
                "generated_parse_failed": row.get("parse_failed"),
                "generated_quality_score": row.get("quality_score"),
                "prompt_projection_z": {
                    layer: sidecar.get("direction_projection_z") for layer, sidecar in prompt_sidecar.items()
                },
                "trace_summary": {
                    layer: trace_layer_summary(trace) for layer, trace in projection_traces.items()
                },
            }
            entry["coverage_flags"]["has_decode_trace"] = True
            add_unique_recommended_use(entry, "decode_trace_existing")


def compact_patch_sites(rows: list[dict[str, Any]]) -> dict[str, list[int] | list[str]]:
    layers = sorted({int(row["layer"]) for row in rows if "layer" in row})
    landmarks = sorted({row["landmark"] for row in rows if row.get("landmark")})
    modes = sorted({row["patch_mode"] for row in rows if row.get("patch_mode")})
    return {"layers": layers, "landmarks": landmarks, "patch_modes": modes}


def add_patch_membership(
    repo_root: Path,
    rows: dict[tuple[str, str, str, int], dict[str, Any]],
    source_rows: dict[str, dict[int, dict[str, Any]]],
) -> None:
    for spec in PATCH_SPECS:
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(repo_root / spec["artifact"]):
            by_pair[str(row["pair_id"])].append(row)
        for pair_id, pair_rows in by_pair.items():
            first = pair_rows[0]
            site_summary = compact_patch_sites(pair_rows)
            for role, row_index_key in (("clean", "clean_row_index"), ("corrupt", "corrupt_row_index")):
                source_row_index = int(first[row_index_key])
                entry = get_entry(
                    rows,
                    source_rows,
                    model_key=spec["model_key"],
                    task=spec["task"],
                    source_file=spec["source_file"],
                    source_row_index=source_row_index,
                    fallback=first,
                )
                membership = {
                    "artifact_id": spec["artifact_id"],
                    "artifact": spec["artifact"],
                    "patch_direction": spec["patch_direction"],
                    "pair_id": pair_id,
                    "pair_role": role,
                    "clean_row_index": int(first["clean_row_index"]),
                    "corrupt_row_index": int(first["corrupt_row_index"]),
                    "gold_hypothesis": first.get("gold_hypothesis"),
                    "foil_hypothesis": first.get("foil_hypothesis"),
                    **site_summary,
                }
                if membership not in entry["patch_membership"]:
                    entry["patch_membership"].append(membership)
                entry["coverage_flags"]["has_patch_pair"] = True
                add_unique_recommended_use(entry, f"patch_{role}_candidate")


def finalize_rows(rows: dict[tuple[str, str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for entry in rows.values():
        flags = entry["coverage_flags"]
        flags["has_all_three"] = flags["has_recognition"] and flags["has_decode_trace"] and flags["has_patch_pair"]
        if flags["has_recognition"] and not flags["has_decode_trace"]:
            add_unique_recommended_use(entry, "decode_trace_candidate")
        entry["patch_membership"] = sorted(
            entry["patch_membership"],
            key=lambda x: (x["artifact_id"], x["pair_id"], x["pair_role"]),
        )
        entry["recommended_use"] = sorted(entry["recommended_use"])
        out.append(entry)
    return sorted(out, key=lambda x: (x["model"], x["task"], x["source_row_index"]))


def coverage_summary(canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        grouped[(row["model"], row["task"])].append(row)

    out = []
    for (model, task), group in sorted(grouped.items()):
        flags = [row["coverage_flags"] for row in group]
        recog = sum(1 for flag in flags if flag["has_recognition"])
        decode = sum(1 for flag in flags if flag["has_decode_trace"])
        patch = sum(1 for flag in flags if flag["has_patch_pair"])
        recog_decode = sum(1 for flag in flags if flag["has_recognition"] and flag["has_decode_trace"])
        recog_patch = sum(1 for flag in flags if flag["has_recognition"] and flag["has_patch_pair"])
        decode_patch = sum(1 for flag in flags if flag["has_decode_trace"] and flag["has_patch_pair"])
        all_three = sum(1 for flag in flags if flag["has_all_three"])
        out.append(
            {
                "model": model,
                "task": task,
                "canonical_rows": len(group),
                "recognition_rows": recog,
                "decode_trace_rows": decode,
                "patch_rows": patch,
                "recognition_decode_overlap": recog_decode,
                "recognition_patch_overlap": recog_patch,
                "decode_patch_overlap": decode_patch,
                "all_three_overlap": all_three,
            }
        )
    return out


def recognition_summary(canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        if row["recognition"]:
            rec = row["recognition"]
            grouped[(row["model"], row["task"], rec["row_selection"])].append(row)
    for (model, task, selection), group in sorted(grouped.items()):
        mcq_correct = [r["recognition"]["mcq_is_correct_choice"] for r in group]
        parse_fail = [r["recognition"]["mcq_choice_parse_failed"] for r in group]
        original_margins = [r["recognition"]["original_margin_gold_minus_foil"] for r in group]
        mcq_margins = [r["recognition"]["mcq_margin_gold_minus_foil"] for r in group]
        heights = sorted({str(r["height"]) for r in group})
        out.append(
            {
                "model": model,
                "task": task,
                "heights": heights,
                "row_selection": selection,
                "n": len(group),
                "mcq_correct": sum(1 for x in mcq_correct if x is True),
                "mcq_accuracy": sum(1 for x in mcq_correct if x is True) / len(group),
                "mcq_parse_fail_rate": sum(1 for x in parse_fail if x is True) / len(group),
                "mean_original_margin": mean([x for x in original_margins if x is not None]),
                "mean_mcq_margin": mean([x for x in mcq_margins if x is not None]),
            }
        )
    return out


def patch_pair_summary(canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_roles: dict[tuple[str, str], dict[str, Any]] = {}
    row_by_index = {(r["model"], r["task"], r["source_row_index"]): r for r in canonical_rows}
    for row in canonical_rows:
        for membership in row["patch_membership"]:
            key = (membership["artifact_id"], membership["pair_id"])
            if key not in pair_roles:
                pair_roles[key] = {
                    "artifact_id": membership["artifact_id"],
                    "patch_direction": membership["patch_direction"],
                    "pair_id": membership["pair_id"],
                    "model": row["model"],
                    "task": row["task"],
                    "clean_row_index": membership["clean_row_index"],
                    "corrupt_row_index": membership["corrupt_row_index"],
                    "layers": membership["layers"],
                    "landmarks": membership["landmarks"],
                    "patch_modes": membership["patch_modes"],
                }
    out = []
    for pair in pair_roles.values():
        clean = row_by_index.get((pair["model"], pair["task"], pair["clean_row_index"]))
        corrupt = row_by_index.get((pair["model"], pair["task"], pair["corrupt_row_index"]))
        out.append(
            {
                **pair,
                "clean_height": clean["height"] if clean else None,
                "corrupt_height": corrupt["height"] if corrupt else None,
                "clean_has_recognition": clean["coverage_flags"]["has_recognition"] if clean else False,
                "corrupt_has_recognition": corrupt["coverage_flags"]["has_recognition"] if corrupt else False,
                "clean_has_decode_trace": clean["coverage_flags"]["has_decode_trace"] if clean else False,
                "corrupt_has_decode_trace": corrupt["coverage_flags"]["has_decode_trace"] if corrupt else False,
            }
        )
    return sorted(out, key=lambda x: (x["model"], x["artifact_id"], x["pair_id"]))


def candidate_rows(canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in canonical_rows:
        rec = row["recognition"]
        if not rec:
            continue
        if row["coverage_flags"]["has_decode_trace"]:
            continue
        if row["free_form"].get("is_correct_strong") is not False:
            continue
        if rec.get("mcq_is_correct_choice") is not True:
            continue
        candidates.append(
            {
                "model": row["model"],
                "task": row["task"],
                "source_row_index": row["source_row_index"],
                "example_id": row["example_id"],
                "height": row["height"],
                "mcq_margin_gold_minus_foil": rec.get("mcq_margin_gold_minus_foil"),
                "original_margin_gold_minus_foil": rec.get("original_margin_gold_minus_foil"),
                "has_patch_pair": row["coverage_flags"]["has_patch_pair"],
            }
        )
    return sorted(candidates, key=lambda x: (x["model"], x["task"], x["height"], x["source_row_index"]))


def build_report(repo_root: Path) -> dict[str, Any]:
    source_rows = load_source_rows(repo_root)
    indexed_rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    add_recognition(repo_root, indexed_rows, source_rows)
    add_decode_traces(repo_root, indexed_rows, source_rows)
    add_patch_membership(repo_root, indexed_rows, source_rows)
    canonical_rows = finalize_rows(indexed_rows)

    coverage = coverage_summary(canonical_rows)
    recognition = recognition_summary(canonical_rows)
    patch_pairs = patch_pair_summary(canonical_rows)
    decode_candidates = candidate_rows(canonical_rows)
    use_counts = Counter(use for row in canonical_rows for use in row["recommended_use"])

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "mixed",
        "task": "mixed",
        "target_variable": "commitment_state",
        "split": "S1",
        "site_or_layer": "mixed",
        "method": "canonical_commitment_rowset_manifest",
        "representation_type": "mixed",
        "controls": [],
        "n": len(canonical_rows),
        "baseline_metrics": {
            "coverage": coverage,
            "recognition": recognition,
            "decode_trace_candidate_rows": len(decode_candidates),
        },
        "intervention_metrics": {},
        "paired_flips": {},
        "parse_fail_rate": None,
        "matched_noise_summary": {},
        "causal_abstraction_claim": (
            "Diagnostic row manifest only. It links free-form source rows, hard-foil recognition, "
            "decode trace status, and patch-pair membership before new commitment-state jobs."
        ),
        "source_artifacts": {
            "sources": SOURCE_SPECS,
            "recognition": RECOGNITION_SPECS,
            "decode_traces": DECODE_TRACE_SPECS,
            "patching": PATCH_SPECS,
        },
        "summary": {
            "coverage": coverage,
            "recognition": recognition,
            "patch_pair_count": len(patch_pairs),
            "decode_trace_candidates": decode_candidates,
            "recommended_use_counts": dict(sorted(use_counts.items())),
        },
        "patch_pairs": patch_pairs,
        "rows": canonical_rows,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Commitment Row-Set Manifest",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Purpose: canonicalize the rows used by the commitment/recognition track before launching new GPU jobs. The manifest links source rows, hard-foil forced-choice recognition, decode-trace coverage, and natural patch-pair membership.",
        "",
        "## Current Interpretation",
        "",
        "The existing row sets are mostly disjoint. Recognition-vs-generation is well supported, but the current artifacts do not yet identify a shared row set where recognition, decode trajectory, and patch-pair evidence can be interpreted together. The next GPU job should extend decode trajectory measurement on manifest-selected recognition rows before more intervention scans.",
        "",
        "## Coverage Summary",
        "",
        "| model | task | rows | recognition | decode trace | patch rows | recog+decode | recog+patch | decode+patch | all three |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["summary"]["coverage"]:
        lines.append(
            "| {model} | {task} | {canonical_rows} | {recognition_rows} | {decode_trace_rows} | {patch_rows} | {recognition_decode_overlap} | {recognition_patch_overlap} | {decode_patch_overlap} | {all_three_overlap} |".format(
                **row
            )
        )

    lines += [
        "",
        "## Recognition Rows",
        "",
        "| model | task | heights | n | MCQ correct | MCQ acc. | parse fail | orig margin | MCQ margin |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["summary"]["recognition"]:
        lines.append(
            f"| {row['model']} | {row['task']} | {','.join(row['heights'])} | {row['n']} | {row['mcq_correct']} | {pct(row['mcq_accuracy'])} | {pct(row['mcq_parse_fail_rate'])} | {fmt(row['mean_original_margin'])} | {fmt(row['mean_mcq_margin'])} |"
        )
    lines += [
        "",
        "Gemma and Qwen support the same recognition-vs-generation theme, but remain non-matched: Gemma is property h3/h4 with balanced polarity, while Qwen is subtype h4.",
        "",
        "## Decode Trace Candidates",
        "",
        "Rows below are free-form strong-incorrect, hard-foil MCQ-correct, and not already covered by the decode-trace pilot. They are the preferred next decode-trajectory batch.",
        "",
        "| model | task | row | example | h | orig margin | MCQ margin | patch pair? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["summary"]["decode_trace_candidates"][:40]:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['source_row_index']} | {row['example_id']} | {row['height']} | {fmt(row['original_margin_gold_minus_foil'])} | {fmt(row['mcq_margin_gold_minus_foil'])} | {row['has_patch_pair']} |"
        )
    if len(report["summary"]["decode_trace_candidates"]) > 40:
        lines.append(
            f"| ... | ... | ... | ... | ... | ... | ... | {len(report['summary']['decode_trace_candidates']) - 40} additional candidates omitted from Markdown; see JSON. |"
        )

    lines += [
        "",
        "## Patch Pair Coverage",
        "",
        "| model | task | artifact | direction | pair | clean row | clean h | corrupt row | corrupt h | clean recog/trace | corrupt recog/trace |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in report["patch_pairs"]:
        clean_cov = f"{pair['clean_has_recognition']}/{pair['clean_has_decode_trace']}"
        corrupt_cov = f"{pair['corrupt_has_recognition']}/{pair['corrupt_has_decode_trace']}"
        lines.append(
            f"| {pair['model']} | {pair['task']} | {pair['artifact_id']} | {pair['patch_direction']} | {pair['pair_id']} | {pair['clean_row_index']} | {pair['clean_height']} | {pair['corrupt_row_index']} | {pair['corrupt_height']} | {clean_cov} | {corrupt_cov} |"
        )

    lines += [
        "",
        "## Recommended Next Job",
        "",
        "1. Run a Gemma decode-trajectory margin job over the 14 Gemma recognition-gap candidates in this manifest, tracking `gold_vs_foil_margin`, `selected_hypothesis`, and the existing prompt-trained correctness projection at L45/L53.",
        "2. If the Gemma measurement separates regenerated-correct from regenerated-wrong trajectories, run the same measurement on a balanced Qwen subset drawn from the 43 Qwen recognition-gap candidates.",
        "3. Keep patching jobs paused until decode trajectories identify a candidate commitment transition or until the row set is expanded so recognition and patch-pair rows overlap.",
        "",
        "## Causal-Abstraction Claim",
        "",
        report["causal_abstraction_claim"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json-out", type=Path, default=Path("docs/commitment_rowset_manifest.json"))
    parser.add_argument("--md-out", type=Path, default=Path("docs/commitment_rowset_manifest.md"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    report = build_report(repo_root)

    json_out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
    md_out = args.md_out if args.md_out.is_absolute() else repo_root / args.md_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_out.write_text(render_md(report) + "\n")

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    print(f"Canonical rows: {report['n']}")
    print(f"Decode trace candidates: {len(report['summary']['decode_trace_candidates'])}")


if __name__ == "__main__":
    main()
