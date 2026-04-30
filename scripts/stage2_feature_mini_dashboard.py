#!/usr/bin/env python3
"""Build local feature dashboards for shortlisted sparse features.

This is a lightweight Neuronpedia substitute for project-specific feature
audits. It analyzes feature activations at the cached Stage 2 decision
position, joins them back to Stage 1 prompts/outputs, and optionally asks an
OpenAI model for qualitative explanations from the top activating examples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_loader import get_openai_gpt_credentials, load_env  # noqa: E402
from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import read_json, read_jsonl, write_json  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def truncate(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    out = str(text).replace("\n", "\\n")
    if len(out) <= max_chars:
        return out
    return out[: max(0, max_chars - 3)] + "..."


def bool_int(value: Any) -> int:
    return int(bool(value))


def activation_vector(top_indices: Any, top_values: Any, feature: int) -> np.ndarray:
    """Return dense per-row activation for one feature from top-k tensors."""
    matches = top_indices == int(feature)
    row_idx, col_idx = matches.nonzero(as_tuple=True)
    values = np.zeros(int(top_indices.shape[0]), dtype=np.float32)
    if int(row_idx.numel()) > 0:
        values[row_idx.cpu().numpy()] = top_values[row_idx, col_idx].float().cpu().numpy()
    return values


def summarize_values(values: np.ndarray, indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {
            "n": 0,
            "density": None,
            "nonzero_n": 0,
            "mean_all": None,
            "mean_nonzero": None,
            "median_all": None,
            "max": None,
        }
    selected = values[np.asarray(indices, dtype=np.int64)]
    nonzero = selected[selected != 0]
    return {
        "n": int(selected.size),
        "density": float(nonzero.size / selected.size) if selected.size else None,
        "nonzero_n": int(nonzero.size),
        "mean_all": float(selected.mean()) if selected.size else None,
        "mean_nonzero": float(nonzero.mean()) if nonzero.size else None,
        "median_all": float(np.median(selected)) if selected.size else None,
        "max": float(selected.max()) if selected.size else None,
    }


def grouped_summary(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    indices: list[int],
    *,
    field: str,
) -> dict[str, Any]:
    out = {}
    groups: dict[str, list[int]] = {}
    for idx in indices:
        key = str(rows[idx].get(field))
        groups.setdefault(key, []).append(idx)
    for key, group_indices in sorted(groups.items()):
        labels = [bool_int(rows[idx].get("is_correct_strong")) for idx in group_indices]
        scores = [float(values[idx]) for idx in group_indices]
        out[key] = {
            **summarize_values(values, group_indices),
            "correct_n": int(sum(labels)),
            "incorrect_n": int(len(labels) - sum(labels)),
            "correct_rate": float(sum(labels) / len(labels)) if labels else None,
            "auc_activation_predicts_correct": _safe_auc(labels, scores),
        }
    return out


def top_examples(
    rows: list[dict[str, Any]],
    source_rows_by_index: dict[int, dict[str, Any]],
    values: np.ndarray,
    indices: list[int],
    *,
    n: int,
    max_prompt_chars: int,
    max_output_chars: int,
) -> list[dict[str, Any]]:
    ranked = sorted(indices, key=lambda idx: float(values[idx]), reverse=True)
    examples = []
    for idx in ranked[:n]:
        sidecar = rows[idx]
        source = source_rows_by_index[int(sidecar["row_index"])]
        structural = source.get("structural") or {}
        ontology_raw = source.get("ontology_raw") or {}
        examples.append(
            {
                "row_index": int(sidecar["row_index"]),
                "example_id": sidecar.get("example_id"),
                "activation": float(values[idx]),
                "height": int(sidecar["height"]),
                "is_correct_strong": bool(sidecar.get("is_correct_strong")),
                "parse_failed": bool(sidecar.get("parse_failed")),
                "error_type": source.get("error_type"),
                "target_concept": structural.get("target_concept"),
                "prompt_text": truncate(source.get("prompt_text"), max_prompt_chars),
                "ground_truth": source.get("ground_truth"),
                "model_output": truncate(source.get("model_output"), max_output_chars),
                "ontology_theories": truncate(ontology_raw.get("theories"), max_prompt_chars),
                "ontology_observations": truncate(ontology_raw.get("observations"), max_prompt_chars),
            }
        )
    return examples


def feature_report_lookup(feature_report: dict[str, Any], *, sae_id: str, task: str, feature: int) -> dict[str, Any] | None:
    try:
        model_report = feature_report["models"][sae_id][task]
    except KeyError:
        return None
    for family in ("top_abs_features", "top_positive_features", "top_negative_features"):
        for row in model_report.get(family, []):
            if int(row["feature"]) == int(feature):
                return {
                    "family": family,
                    "rank": int(row["rank"]),
                    "association": row.get("association"),
                    "sign": row.get("sign"),
                    "weight": float(row["weight"]),
                    "abs_weight": float(row.get("abs_weight", abs(float(row["weight"])))),
                    "activation_all": row.get("activation_all"),
                    "activation_train": row.get("activation_train"),
                }
    return None


def analyze_task(
    *,
    feature_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    activation_site: str,
    sae_id: str,
    top_k: int,
    features: list[int],
    drop_parse_failed: bool,
    top_n: int,
    max_prompt_chars: int,
    max_output_chars: int,
    feature_report: dict[str, Any] | None,
) -> dict[str, Any]:
    feature_stem = activation_stem(model_key=model_key, task=task, layer=layer, activation_site=activation_site)
    prefix = feature_dir / f"{feature_stem}_{sae_id}_top{top_k}"
    meta = read_json(prefix.with_suffix(".meta.json"))
    tensors = load_file(prefix.with_suffix(".safetensors"))
    top_indices = tensors["top_indices"].long()
    top_values = tensors["top_values"].float()
    sidecar = read_jsonl(prefix.with_suffix(".example_ids.jsonl"))

    source_path = Path(meta["source_activation_meta"]["jsonl_path"])
    source_rows = read_jsonl(source_path)
    source_rows_by_index = {int(row["row_index"]) if "row_index" in row else idx: row for idx, row in enumerate(source_rows)}
    if len(sidecar) != int(top_indices.shape[0]):
        raise ValueError(f"{prefix} sidecar rows {len(sidecar)} != feature rows {top_indices.shape[0]}")

    all_indices = list(range(len(sidecar)))
    kept_indices = [
        idx
        for idx, row in enumerate(sidecar)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    labels = [bool_int(sidecar[idx].get("is_correct_strong")) for idx in kept_indices]

    task_report = {
        "feature_path": str(prefix.with_suffix(".safetensors")),
        "meta_path": str(prefix.with_suffix(".meta.json")),
        "source_path": str(source_path),
        "input_rows": len(sidecar),
        "kept_rows": len(kept_indices),
        "drop_parse_failed": drop_parse_failed,
        "features": {},
    }

    for feature in features:
        values = activation_vector(top_indices, top_values, feature)
        scores = [float(values[idx]) for idx in kept_indices]
        label_groups = {
            "correct": [idx for idx in kept_indices if bool(sidecar[idx].get("is_correct_strong"))],
            "incorrect": [idx for idx in kept_indices if not bool(sidecar[idx].get("is_correct_strong"))],
        }
        nonzero_indices = [idx for idx in kept_indices if float(values[idx]) != 0.0]
        active_labels = [bool_int(sidecar[idx].get("is_correct_strong")) for idx in nonzero_indices]
        active_heights = [int(sidecar[idx].get("height")) for idx in nonzero_indices]
        task_report["features"][str(feature)] = {
            "feature": int(feature),
            "coefficient_report": feature_report_lookup(feature_report or {}, sae_id=sae_id, task=task, feature=feature),
            "overall_all_rows": summarize_values(values, all_indices),
            "overall_kept_rows": summarize_values(values, kept_indices),
            "auc_activation_predicts_correct": _safe_auc(labels, scores),
            "point_biserial_corr_activation_correct": (
                float(np.corrcoef(np.asarray(scores), np.asarray(labels, dtype=np.float32))[0, 1])
                if len(set(labels)) >= 2 and len(set(scores)) >= 2
                else None
            ),
            "by_correctness": {
                name: summarize_values(values, group_indices)
                for name, group_indices in label_groups.items()
            },
            "by_height": grouped_summary(sidecar, values, kept_indices, field="height"),
            "by_error_type_active_rows": {
                str(key): int(count)
                for key, count in sorted(
                    {
                        (source_rows_by_index[int(sidecar[idx]["row_index"])].get("error_type") or "none"): sum(
                            1
                            for j in nonzero_indices
                            if (source_rows_by_index[int(sidecar[j]["row_index"])].get("error_type") or "none")
                            == (source_rows_by_index[int(sidecar[idx]["row_index"])].get("error_type") or "none")
                        )
                        for idx in nonzero_indices
                    }.items()
                )
            },
            "active_correct_rate": float(sum(active_labels) / len(active_labels)) if active_labels else None,
            "active_height_counts": {
                str(height): int(active_heights.count(height))
                for height in sorted(set(active_heights))
            },
            "top_examples": top_examples(
                sidecar,
                source_rows_by_index,
                values,
                kept_indices,
                n=top_n,
                max_prompt_chars=max_prompt_chars,
                max_output_chars=max_output_chars,
            ),
        }

    return task_report


def candidate_payload_for_llm(report: dict[str, Any], feature: int, *, examples_per_task: int) -> dict[str, Any]:
    payload = {
        "feature": int(feature),
        "context": (
            "Activations are from the last pre-generation prompt token of Gemma 3 27B "
            "on synthetic ontology reasoning prompts. Correctness is whether the generated "
            "hypothesis exactly matches the ground truth under the project scorer."
        ),
        "tasks": {},
    }
    for task, task_report in report["tasks"].items():
        feature_report = task_report["features"].get(str(feature))
        if feature_report is None:
            continue
        payload["tasks"][task] = {
            "coefficient_report": feature_report.get("coefficient_report"),
            "overall_kept_rows": feature_report.get("overall_kept_rows"),
            "auc_activation_predicts_correct": feature_report.get("auc_activation_predicts_correct"),
            "by_correctness": feature_report.get("by_correctness"),
            "by_height": feature_report.get("by_height"),
            "active_correct_rate": feature_report.get("active_correct_rate"),
            "top_examples": feature_report.get("top_examples", [])[:examples_per_task],
        }
    return payload


def feature_summaries(report: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for feature in report["features"]:
        row: dict[str, Any] = {"feature": int(feature), "tasks": {}}
        for task in ("infer_property", "infer_subtype"):
            task_feature = report["tasks"].get(task, {}).get("features", {}).get(str(feature))
            if task_feature is None:
                continue
            coef = task_feature.get("coefficient_report") or {}
            row["tasks"][task] = {
                "coefficient_rank": coef.get("rank"),
                "coefficient_family": coef.get("family"),
                "coefficient_association": coef.get("association"),
                "coefficient_weight": coef.get("weight"),
                "density": task_feature["overall_kept_rows"]["density"],
                "nonzero_n": task_feature["overall_kept_rows"]["nonzero_n"],
                "auc_activation_predicts_correct": task_feature.get("auc_activation_predicts_correct"),
                "point_biserial_corr_activation_correct": task_feature.get(
                    "point_biserial_corr_activation_correct"
                ),
                "active_correct_rate": task_feature.get("active_correct_rate"),
                "active_height_counts": task_feature.get("active_height_counts"),
            }
        summaries.append(row)
    return summaries


def _is_gpt5_plus(model: str) -> bool:
    low = model.lower()
    return low.startswith("gpt-5") or low.startswith("gpt-6") or low.startswith("o3") or low.startswith("o4")


def explain_feature_with_llm(
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    payload: dict[str, Any],
    max_completion_tokens: int,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    prompt = {
        "instruction": (
            "You are auditing a sparse feature from a Gemma 3 27B feature dictionary. "
            "Use only the provided statistics and top activating examples. Return JSON only. "
            "Do not claim causality. Distinguish ontology-reasoning evidence from lexical, "
            "style, parse-failure, prompt-length, or height confounds."
        ),
        "output_schema": {
            "feature": "integer",
            "short_name": "short descriptive label, or unclear",
            "hypothesis": "what seems to make this feature activate",
            "evidence": ["2-5 concrete observations from the examples/statistics"],
            "confounds": ["possible confounds or reasons this may be a phantom"],
            "reasoning_relevance": "one of: likely_reasoning_related, maybe_reasoning_related, likely_surface_or_style, unclear",
            "confidence": "one of: low, medium, high",
            "steering_priority": "one of: high, medium, low",
            "report_sentence": "one cautious sentence suitable for a report",
        },
        "feature_payload": payload,
    }
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        "response_format": {"type": "json_object"},
    }
    if _is_gpt5_plus(model):
        kwargs["max_completion_tokens"] = max_completion_tokens
    else:
        kwargs["temperature"] = 0
        kwargs["max_tokens"] = max_completion_tokens

    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"parse_error": True, "raw_content": content}
    return {
        "model": model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "response": parsed,
        "usage": completion.usage.model_dump() if completion.usage is not None else None,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# 27B Big-L0 Feature Mini-Dashboard",
        "",
        f"Created: `{report['created_at_utc']}`",
        "",
        "Activations are measured at the cached last pre-generation prompt position.",
        "",
        "## Summary",
        "",
        "Rank comes from the trained sparse probe coefficient report; AUC is the univariate score from this feature alone.",
        "",
        "| Feature | Property rank | Property density | Property AUC | Subtype rank | Subtype density | Subtype AUC | Notes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for feature in report["features"]:
        row = [str(feature)]
        notes = []
        for task in ("infer_property", "infer_subtype"):
            task_feature = report["tasks"].get(task, {}).get("features", {}).get(str(feature))
            if task_feature is None:
                row.extend(["", "", ""])
                continue
            coef = task_feature.get("coefficient_report") or {}
            rank = coef.get("rank")
            density = task_feature["overall_kept_rows"]["density"]
            auc = task_feature.get("auc_activation_predicts_correct")
            row.extend(
                [
                    "" if rank is None else str(rank),
                    "" if density is None else f"{density:.3f}",
                    "" if auc is None else f"{auc:.3f}",
                ]
            )
            if density is not None and density > 0.9:
                notes.append(f"{task} dense")
        row.append("; ".join(notes))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Top Activating Examples", ""])
    for feature in report["features"]:
        lines.extend([f"### Feature {feature}", ""])
        if str(feature) in report.get("llm_explanations", {}):
            llm = report["llm_explanations"][str(feature)]["response"]
            lines.extend(
                [
                    f"- LLM short name: {llm.get('short_name', 'n/a')}",
                    f"- LLM relevance: {llm.get('reasoning_relevance', 'n/a')}",
                    f"- LLM steering priority: {llm.get('steering_priority', 'n/a')}",
                    f"- LLM hypothesis: {llm.get('hypothesis', 'n/a')}",
                    "",
                ]
            )
        for task in ("infer_property", "infer_subtype"):
            task_feature = report["tasks"].get(task, {}).get("features", {}).get(str(feature))
            if task_feature is None:
                continue
            lines.append(f"#### `{task}`")
            lines.append("")
            lines.append("| Act. | H | Correct | Error | Ground truth | Output |")
            lines.append("| ---: | ---: | --- | --- | --- | --- |")
            for ex in task_feature["top_examples"][:5]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"{ex['activation']:.1f}",
                            str(ex["height"]),
                            str(ex["is_correct_strong"]),
                            str(ex["error_type"]),
                            truncate(ex["ground_truth"], 90).replace("|", "\\|"),
                            truncate(ex["model_output"], 160).replace("|", "\\|"),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    feature_report = read_json(args.feature_report) if args.feature_report else None
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model_key,
        "layer": args.layer,
        "activation_site": args.activation_site,
        "sae_id": args.sae_id,
        "top_k": args.top_k,
        "features": args.features,
        "feature_report": str(args.feature_report) if args.feature_report else None,
        "tasks": {},
        "llm_explanations": {},
    }
    for task in args.tasks:
        report["tasks"][task] = analyze_task(
            feature_dir=args.feature_dir,
            model_key=args.model_key,
            task=task,
            layer=args.layer,
            activation_site=args.activation_site,
            sae_id=args.sae_id,
            top_k=args.top_k,
            features=args.features,
            drop_parse_failed=not args.keep_parse_failed,
            top_n=args.top_n,
            max_prompt_chars=args.max_prompt_chars,
            max_output_chars=args.max_output_chars,
            feature_report=feature_report,
        )

    if args.explain_model:
        load_env()
        base_url, api_key = get_openai_gpt_credentials()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY_GPT or OPENAI_API_KEY is required for --explain-model")
        for idx, feature in enumerate(args.features, start=1):
            payload = candidate_payload_for_llm(report, feature, examples_per_task=args.llm_examples_per_task)
            print(f"explaining feature {feature} ({idx}/{len(args.features)}) with {args.explain_model}", flush=True)
            report["llm_explanations"][str(feature)] = explain_feature_with_llm(
                model=args.explain_model,
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                max_completion_tokens=args.max_completion_tokens,
            )
            if idx < len(args.features):
                time.sleep(args.sleep_seconds)

    report["feature_summaries"] = feature_summaries(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("results/stage2/sae_features"))
    parser.add_argument("--activation-site", default="mlp_in_weighted")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--tasks", nargs="+", default=["infer_property", "infer_subtype"])
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--sae-id", default="layer_45_width_262k_l0_big_affine")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--features", type=parse_int_list, required=True)
    parser.add_argument("--feature-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--max-prompt-chars", type=int, default=700)
    parser.add_argument("--max-output-chars", type=int, default=450)
    parser.add_argument("--keep-parse-failed", action="store_true")
    parser.add_argument("--explain-model")
    parser.add_argument("--llm-examples-per-task", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=1600)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    args = parser.parse_args()

    report = run(args)
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(report) + "\n")
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
