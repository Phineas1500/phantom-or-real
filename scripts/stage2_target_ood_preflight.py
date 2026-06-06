#!/usr/bin/env python3
"""Audit target and OOD feasibility for the next-paper Stage 2 program."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import read_json, read_jsonl, read_split_assignments, write_json  # noqa: E402


TASKS = ("infer_property", "infer_subtype")
SPLIT_FAMILIES = ("s1", "s3")

MODEL_CONFIGS = (
    {
        "model_key": "gemma3_27b",
        "model_label": "Gemma 3 27B",
        "main_layer": 45,
        "source_prefix": "gemma3_27b",
    },
    {
        "model_key": "qwen35_27b",
        "model_label": "Qwen3.5 27B",
        "main_layer": 53,
        "source_prefix": "qwen35_27b",
    },
)


def read_full_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_row_index"] = row_index
            rows.append(row)
    return rows


def target_value(row: dict[str, Any], target: str) -> bool:
    if target == "is_correct_strong":
        return bool(row.get("is_correct_strong"))
    if target == "is_correct_weak":
        return bool(row.get("is_correct_weak"))
    if target == "quality_score_perfect":
        return row.get("quality_score") == 1.0
    if target == "weak_not_strong":
        return bool(row.get("is_correct_weak")) and not bool(row.get("is_correct_strong"))
    raise ValueError(f"unknown target: {target}")


TARGETS = (
    "is_correct_strong",
    "is_correct_weak",
    "quality_score_perfect",
    "weak_not_strong",
)


def class_counts(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    positives = sum(1 for row in rows if target_value(row, target))
    return {
        "n": len(rows),
        "positive_n": int(positives),
        "negative_n": int(len(rows) - positives),
        "positive_rate": float(positives / len(rows)) if rows else None,
        "has_two_classes": positives > 0 and positives < len(rows),
    }


def quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["quality_score"]) for row in rows if row.get("quality_score") is not None]
    counts = Counter(values)
    if not values:
        return {"n": 0, "unique_count": 0}
    return {
        "n": len(values),
        "unique_count": len(counts),
        "min": min(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": max(values),
        "top_values": [
            {"value": value, "n": count}
            for value, count in counts.most_common(10)
        ],
    }


def rows_by_height(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[f"h{int(row['height'])}"].append(row)
    return dict(sorted(out.items()))


def target_by_height(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for height_key, height_rows in rows_by_height(rows).items():
        out[height_key] = {
            target: class_counts(height_rows, target)
            for target in TARGETS
        }
    return out


def disagreement_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "strong_vs_weak": sum(
            bool(row.get("is_correct_strong")) != bool(row.get("is_correct_weak"))
            for row in rows
        ),
        "strong_vs_quality_score_perfect": sum(
            bool(row.get("is_correct_strong")) != (row.get("quality_score") == 1.0)
            for row in rows
        ),
        "weak_vs_quality_score_perfect": sum(
            bool(row.get("is_correct_weak")) != (row.get("quality_score") == 1.0)
            for row in rows
        ),
    }


def split_counts(
    *,
    rows: list[dict[str, Any]],
    assignments: dict[tuple[str, int], dict[str, Any]],
    source_file: str,
    split_family: str,
) -> dict[str, Any]:
    field = f"{split_family}_split"
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    missing = []
    for row in rows:
        key = (source_file, int(row["_row_index"]))
        assignment = assignments.get(key)
        if assignment is None:
            missing.append(key)
            continue
        split = assignment[field]
        split_rows[split].append(row)

    return {
        "missing_assignment_count": len(missing),
        "targets": {
            target: {
                split: class_counts(split_rows[split], target)
                for split in ("train", "val", "test")
            }
            for target in TARGETS
        },
    }


def height_extrapolation_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in rows if int(row["height"]) in (1, 2)]
    test_rows = [row for row in rows if int(row["height"]) in (3, 4)]
    out = {
        "train_heights": [1, 2],
        "test_heights": [3, 4],
        "targets": {},
    }
    for target in TARGETS:
        train_counts = class_counts(train_rows, target)
        test_counts = class_counts(test_rows, target)
        out["targets"][target] = {
            "train": train_counts,
            "test": test_counts,
            "runnable": bool(train_counts["has_two_classes"] and test_counts["has_two_classes"]),
        }
    return out


def structural_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "tree_height",
        "num_direct_paths",
        "num_observations",
        "num_theories_axioms",
        "parent_salience",
    )
    out: dict[str, Any] = {}
    for field in numeric_fields:
        values = []
        for row in rows:
            structural = row.get("structural") or {}
            value = structural.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            out[field] = {
                "n": len(values),
                "unique_count": len(set(values)),
                "min": min(values),
                "mean": float(statistics.fmean(values)),
                "max": max(values),
            }
    return out


def source_report(
    *,
    source_path: Path,
    rows: list[dict[str, Any]],
    assignments: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    parsed_rows = [row for row in rows if not row.get("parse_failed")]
    report = {
        "source_file": str(source_path),
        "n": len(rows),
        "parsed_n": len(parsed_rows),
        "parse_failed_n": len(rows) - len(parsed_rows),
        "targets_all_rows": {target: class_counts(rows, target) for target in TARGETS},
        "targets_parsed_rows": {target: class_counts(parsed_rows, target) for target in TARGETS},
        "target_disagreements_all_rows": disagreement_counts(rows),
        "target_disagreements_parsed_rows": disagreement_counts(parsed_rows),
        "quality_score_all_rows": quality_summary(rows),
        "quality_score_parsed_rows": quality_summary(parsed_rows),
        "targets_by_height_parsed_rows": target_by_height(parsed_rows),
        "splits_parsed_rows": {
            split_family: split_counts(
                rows=parsed_rows,
                assignments=assignments,
                source_file=str(source_path),
                split_family=split_family,
            )
            for split_family in SPLIT_FAMILIES
        },
        "height_extrapolation_parsed_rows": height_extrapolation_counts(parsed_rows),
        "structural_parsed_rows": structural_summary(parsed_rows),
    }
    return report


def parse_activation_inventory(activation_dir: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for meta_path in sorted(activation_dir.glob("*.meta.json")):
        try:
            meta = read_json(meta_path)
        except Exception:  # noqa: BLE001
            continue
        model_key = str(meta.get("model_key"))
        task = str(meta.get("task"))
        if model_key not in {"gemma3_27b", "qwen35_27b"} or task not in TASKS:
            continue
        key = f"{model_key}/{task}"
        inventory.setdefault(key, [])
        inventory[key].append(
            {
                "meta_file": str(meta_path),
                "activation_file": meta.get("activation_file"),
                "sidecar_file": meta.get("sidecar_file"),
                "layer": meta.get("layer"),
                "hook_name": meta.get("hook_name"),
                "shape": meta.get("shape"),
                "row_count": meta.get("row_count"),
                "jsonl_path": meta.get("jsonl_path"),
            }
        )
    return inventory


def activation_alignment_report(
    *,
    activation_dir: Path,
    source_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for config in MODEL_CONFIGS:
        model_key = config["model_key"]
        layer = int(config["main_layer"])
        for task in TASKS:
            source_file = f"results/full/with_errortype/{config['source_prefix']}_{task}.jsonl"
            source_by_index = {
                int(row["_row_index"]): row
                for row in source_rows.get(source_file, [])
            }
            stem = activation_stem(model_key=model_key, task=task, layer=layer)
            prefix = activation_dir / stem
            meta_path = prefix.with_suffix(".meta.json")
            sidecar_path = prefix.with_suffix(".example_ids.jsonl")
            activation_path = prefix.with_suffix(".safetensors")
            key = f"{model_key}/{task}/L{layer}/resid_post"
            if not meta_path.exists():
                out[key] = {"available": False, "meta_file": str(meta_path)}
                continue
            meta = read_json(meta_path)
            sidecar_rows = read_jsonl(sidecar_path) if sidecar_path.exists() else []
            missing_source = 0
            strong_mismatch = 0
            for sidecar_row in sidecar_rows:
                source_row = source_by_index.get(int(sidecar_row["row_index"]))
                if source_row is None:
                    missing_source += 1
                    continue
                if bool(sidecar_row.get("is_correct_strong")) != bool(source_row.get("is_correct_strong")):
                    strong_mismatch += 1
            out[key] = {
                "available": True,
                "activation_file": str(activation_path),
                "activation_exists": activation_path.exists(),
                "meta_file": str(meta_path),
                "sidecar_file": str(sidecar_path),
                "sidecar_exists": sidecar_path.exists(),
                "meta_jsonl_path": meta.get("jsonl_path"),
                "expected_source_file": source_file,
                "source_file_matches": meta.get("jsonl_path") == source_file,
                "shape": meta.get("shape"),
                "row_count": meta.get("row_count"),
                "sidecar_row_count": len(sidecar_rows),
                "sidecar_count_matches_meta": len(sidecar_rows) == int(meta.get("row_count", -1)),
                "missing_source_row_count": missing_source,
                "strong_label_mismatch_count": strong_mismatch,
                "target_join_required_for": ["is_correct_weak", "quality_score", "structural"],
            }
    return out


def name_scramble_report(docs_dir: Path, repo_root: Path) -> dict[str, Any]:
    report_path = docs_dir / "namescramble_27b_l45_raw_probe_s1.json"
    out: dict[str, Any] = {
        "gemma3_27b": {
            "available": report_path.exists(),
            "raw_probe_report": str(report_path),
        },
        "qwen35_27b": {
            "available": False,
            "note": "No Qwen name-scramble activations were found in the current repo.",
        },
    }
    if report_path.exists():
        data = read_json(report_path)
        task_summary = {}
        for task, task_data in data.get("results", {}).items():
            task_summary[task] = {}
            for condition, condition_data in task_data.get("conditions", {}).items():
                scrambled = condition_data.get("scrambled", {})
                task_summary[task][condition] = {
                    "n": scrambled.get("n"),
                    "auc": scrambled.get("auc"),
                    "strong_accuracy": scrambled.get("strong_accuracy"),
                    "auc_drop_vs_matched_original": condition_data.get("auc_drop_vs_matched_original"),
                    "activation_prefix": condition_data.get("activation_prefix"),
                    "infer_jsonl": condition_data.get("infer_jsonl"),
                }
        out["gemma3_27b"].update(
            {
                "conditions": data.get("conditions"),
                "split_family": data.get("split_family"),
                "task_summary": task_summary,
            }
        )

    qwen_files = list((repo_root / "results/stage2").glob("*qwen*scrambl*"))
    if qwen_files:
        out["qwen35_27b"] = {
            "available": True,
            "note": "Potential Qwen name-scramble artifacts found; inspect before treating as complete.",
            "files": [str(path) for path in qwen_files],
        }
    return out


def recommended_next_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for source_key, source in report["source_reports"].items():
        split_reports = source["splits_parsed_rows"]
        for split_family in SPLIT_FAMILIES:
            weak = split_reports[split_family]["targets"]["is_correct_weak"]
            if all(weak[split]["has_two_classes"] for split in ("train", "val", "test")):
                runs.append(
                    {
                        "priority": "high",
                        "source": source_key,
                        "split_family": split_family,
                        "target": "is_correct_weak",
                        "reason": "Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe.",
                    }
                )
        strong_h = source["height_extrapolation_parsed_rows"]["targets"]["is_correct_strong"]
        weak_h = source["height_extrapolation_parsed_rows"]["targets"]["is_correct_weak"]
        if strong_h["runnable"]:
            runs.append(
                {
                    "priority": "medium",
                    "source": source_key,
                    "split_family": "height_h12_to_h34",
                    "target": "is_correct_strong",
                    "reason": "Height extrapolation is class-balanced and directly tests whether the correctness direction is a depth/difficulty proxy.",
                }
            )
        if weak_h["runnable"]:
            runs.append(
                {
                    "priority": "medium",
                    "source": source_key,
                    "split_family": "height_h12_to_h34",
                    "target": "is_correct_weak",
                    "reason": "Weak-correctness height extrapolation checks whether relaxed validity behaves like strong correctness across depth.",
                }
            )
    return runs


def source_path_for(model_config: dict[str, Any], task: str) -> Path:
    return Path("results/full/with_errortype") / f"{model_config['source_prefix']}_{task}.jsonl"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    assignments = read_split_assignments(args.splits)
    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_reports: dict[str, Any] = {}
    for config in MODEL_CONFIGS:
        for task in TASKS:
            source_path = source_path_for(config, task)
            rows = read_full_rows(source_path)
            source_rows[str(source_path)] = rows
            source_reports[f"{config['model_key']}/{task}"] = {
                "model_key": config["model_key"],
                "model_label": config["model_label"],
                "task": task,
                **source_report(source_path=source_path, rows=rows, assignments=assignments),
            }

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal_abstraction_track": "target_ood_preflight",
        "target_variables": list(TARGETS),
        "graded_quality_note": (
            "quality_score is audited as a continuous/graded field; quality_score_perfect "
            "is the binary perfect-quality target used for initial feasibility only."
        ),
        "source_reports": source_reports,
        "activation_inventory": parse_activation_inventory(args.activation_dir),
        "main_activation_alignment": activation_alignment_report(
            activation_dir=args.activation_dir,
            source_rows=source_rows,
        ),
        "name_scramble": name_scramble_report(args.docs_dir, args.repo_root),
        "assumptions": {
            "drop_parse_failed_for_probe_feasibility": True,
            "height_extrapolation_train_heights": [1, 2],
            "height_extrapolation_test_heights": [3, 4],
            "main_gemma_site": "Gemma 3 27B L45 residual post",
            "main_qwen_site": "Qwen3.5 27B L53 residual post",
        },
    }
    report["recommended_next_runs"] = recommended_next_runs(report)
    return report


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Target/OOD Preflight",
        "",
        f"Generated: `{report['created_at_utc']}`",
        "",
        "This preflight audits the low-cost target and OOD extensions for the causal-abstraction program. It uses parsed rows for probe feasibility and does not load large activation tensors.",
        "",
        "## Label Feasibility",
        "",
    ]
    rows = []
    for source_key, source in report["source_reports"].items():
        parsed_n = source["parsed_n"]
        strong = source["targets_parsed_rows"]["is_correct_strong"]
        weak = source["targets_parsed_rows"]["is_correct_weak"]
        quality = source["targets_parsed_rows"]["quality_score_perfect"]
        disagreements = source["target_disagreements_parsed_rows"]
        rows.append(
            [
                source_key,
                parsed_n,
                f"{strong['positive_n']}/{parsed_n} ({pct(strong['positive_rate'])})",
                f"{weak['positive_n']}/{parsed_n} ({pct(weak['positive_rate'])})",
                f"{quality['positive_n']}/{parsed_n} ({pct(quality['positive_rate'])})",
                disagreements["strong_vs_weak"],
                disagreements["strong_vs_quality_score_perfect"],
                source["quality_score_parsed_rows"]["unique_count"],
            ]
        )
    lines.append(
        md_table(
            [
                "source",
                "parsed n",
                "strong+",
                "weak+",
                "quality=1+",
                "strong!=weak",
                "strong!=quality=1",
                "quality unique",
            ],
            rows,
        )
    )
    lines.extend(
        [
            "",
            "Initial read: `is_correct_weak` is the informative alternate binary target. `quality_score_perfect` is identical to strong correctness for Qwen and nearly identical for Gemma, so it is mostly a sanity check unless we model the graded score directly.",
            "",
            "## Height Extrapolation",
            "",
        ]
    )
    rows = []
    for source_key, source in report["source_reports"].items():
        height = source["height_extrapolation_parsed_rows"]["targets"]
        for target in ("is_correct_strong", "is_correct_weak", "quality_score_perfect"):
            train = height[target]["train"]
            test = height[target]["test"]
            rows.append(
                [
                    source_key,
                    target,
                    f"{train['positive_n']}/{train['n']} ({pct(train['positive_rate'])})",
                    f"{test['positive_n']}/{test['n']} ({pct(test['positive_rate'])})",
                    "yes" if height[target]["runnable"] else "no",
                ]
            )
    lines.append(md_table(["source", "target", "h1/h2 train +", "h3/h4 test +", "runnable"], rows))
    lines.extend(["", "## Main Activation Alignment", ""])
    rows = []
    for key, alignment in report["main_activation_alignment"].items():
        rows.append(
            [
                key,
                "yes" if alignment.get("available") else "no",
                alignment.get("shape"),
                alignment.get("sidecar_row_count"),
                "yes" if alignment.get("source_file_matches") else "no",
                alignment.get("strong_label_mismatch_count"),
            ]
        )
    lines.append(
        md_table(
            ["site", "available", "shape", "sidecar rows", "source matches", "strong mismatches"],
            rows,
        )
    )
    lines.extend(["", "## Name-Scramble OOD", ""])
    ns = report["name_scramble"]
    rows = []
    for model_key, model_data in ns.items():
        if not model_data.get("available"):
            rows.append([model_key, "no", model_data.get("note", "")])
            continue
        summaries = []
        for task, task_summary in model_data.get("task_summary", {}).items():
            for condition, condition_summary in task_summary.items():
                summaries.append(
                    f"{task}/{condition}: auc {condition_summary.get('auc'):.3f}, drop {condition_summary.get('auc_drop_vs_matched_original'):.3f}"
                )
        rows.append([model_key, "yes", "; ".join(summaries)])
    lines.append(md_table(["model", "available", "summary"], rows))
    lines.extend(["", "## Recommended Next Runs", ""])
    rows = [
        [
            run["priority"],
            run["source"],
            run["split_family"],
            run["target"],
            run["reason"],
        ]
        for run in report["recommended_next_runs"]
    ]
    lines.append(md_table(["priority", "source", "split", "target", "reason"], rows))
    lines.extend(
        [
            "",
            "## Causal-Abstraction Claim",
            "",
            "This artifact is predictive/diagnostic only. It identifies which target variables and OOD splits are runnable before causal or intervention claims are made.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/target_ood_preflight.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/target_ood_preflight.md"))
    args = parser.parse_args()

    report = build_report(args)
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report))
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
