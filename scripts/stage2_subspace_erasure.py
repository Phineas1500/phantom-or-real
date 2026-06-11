#!/usr/bin/env python3
"""Multi-layer correctness-subspace erasure necessity test.

Trains a raw correctness probe direction per residual layer, then mean-ablates
the per-layer direction at every position (prompt and decode) during
generation. A necessity claim requires baseline-correct rows to degrade under
raw erasure beyond orthogonal and matched-Gaussian erasure controls; a clean
epiphenomenality claim requires no degradation anywhere. See
docs/causal_handle_directions.md experiment 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import (  # noqa: E402
    generate_one,
    json_default,
    package_version,
    serializable_direction_summary,
    torch_dtype,
)
from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_probes import load_probe_dataset  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    make_gaussian_unit_direction,
    make_orthogonal_unit_direction,
    parse_int_list,
    score_reply,
    select_balanced_stage1_rows,
    train_raw_probe_direction,
)


@dataclass(frozen=True)
class ErasureCondition:
    label: str
    vector_kind: str | None


def parse_condition_kinds(value: str) -> list[str]:
    allowed = {"baseline", "erase_raw", "erase_orthogonal", "erase_gaussian"}
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    return parsed


def make_condition_plan(condition_kinds: list[str]) -> list[ErasureCondition]:
    plan: list[ErasureCondition] = []
    if "baseline" in condition_kinds:
        plan.append(ErasureCondition("baseline", None))
    for kind, vector_kind in (
        ("erase_raw", "raw"),
        ("erase_orthogonal", "orthogonal"),
        ("erase_gaussian", "gaussian"),
    ):
        if kind in condition_kinds:
            plan.append(ErasureCondition(kind, vector_kind))
    if not plan:
        raise ValueError("condition plan is empty")
    if plan[0].label != "baseline":
        raise ValueError("baseline condition is required for paired interpretation")
    return plan


def make_erasure_hook(
    *,
    vector: np.ndarray,
    projection_mean: float,
    projection_std: float,
) -> tuple[Any, dict[str, Any]]:
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    state: dict[str, Any] = {"calls": 0, "positions": 0, "abs_delta_sd_sum": 0.0}

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        key = (str(act.device), act.dtype)
        unit = cache.get(key)
        if unit is None:
            unit = torch.as_tensor(vector, device=act.device, dtype=torch.float32)
            cache[key] = unit
        projection = act.float() @ unit
        delta = projection - float(projection_mean)
        act -= (delta.unsqueeze(-1) * unit).to(act.dtype)
        state["calls"] += 1
        state["positions"] += int(delta.numel())
        state["abs_delta_sd_sum"] += float(delta.abs().sum().item()) / float(projection_std)
        return act

    return hook_fn, state


def summarize_hook_states(states: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        f"L{layer}": {
            "calls": state["calls"],
            "positions": state["positions"],
            "mean_abs_delta_sd": (
                state["abs_delta_sd_sum"] / state["positions"] if state["positions"] else None
            ),
        }
        for layer, state in sorted(states.items())
    }


def projection_stats(x: np.ndarray, unit: np.ndarray) -> dict[str, float]:
    projections = np.asarray(x.astype(np.float64) @ np.asarray(unit, dtype=np.float64))
    return {
        "projection_mean": float(projections.mean()),
        "projection_std": float(projections.std(ddof=0)),
    }


def build_layer_directions(
    *,
    layers: list[int],
    activation_dir: Path,
    model_key: str,
    task: str,
    splits_path: Path,
    source_file: str,
    split_family: str,
    probe_seed: int,
    orthogonal_seed: int,
    gaussian_seed: int,
    c_values: tuple[float, ...],
    max_iter: int,
    solver: str,
) -> dict[int, dict[str, Any]]:
    by_layer: dict[int, dict[str, Any]] = {}
    for layer in layers:
        prefix = activation_dir / f"{model_key}_{task}_L{layer}"
        activation_path = prefix.with_suffix(".safetensors")
        sidecar_path = prefix.with_suffix(".example_ids.jsonl")
        direction = train_raw_probe_direction(
            activation_path=activation_path,
            sidecar_path=sidecar_path,
            splits_path=splits_path,
            source_file=source_file,
            split_family=split_family,
            seed=probe_seed,
            c_values=c_values,
            max_iter=max_iter,
            solver=solver,
        )
        orthogonal = make_orthogonal_unit_direction(
            direction["unit_direction"], seed=orthogonal_seed + layer
        )
        gaussian = make_gaussian_unit_direction(
            direction["unit_direction"], seed=gaussian_seed + layer
        )
        dataset = load_probe_dataset(
            activation_path=activation_path,
            sidecar_path=sidecar_path,
            drop_parse_failed=True,
        )
        by_layer[layer] = {
            "direction": direction,
            "vectors": {
                "raw": direction["unit_direction"],
                "orthogonal": orthogonal,
                "gaussian": gaussian,
            },
            "stats": {
                "raw": {
                    "projection_mean": float(direction["train_projection_mean"]),
                    "projection_std": float(direction["train_projection_std"]),
                },
                "orthogonal": projection_stats(dataset["x"], orthogonal),
                "gaussian": projection_stats(dataset["x"], gaussian),
            },
        }
        print(
            f"L{layer}: best_c={direction['best_c']} val_auc={direction['val_auc']:.4f} "
            f"test_auc={direction['test_auc']:.4f} proj_std={direction['train_projection_std']:.4f}",
            flush=True,
        )
    return by_layer


def save_layer_direction_artifact(path: Path, by_layer: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for layer, entry in by_layer.items():
        for kind, vector in entry["vectors"].items():
            arrays[f"L{layer}_{kind}_unit"] = np.asarray(vector, dtype=np.float32)
        for kind, stats in entry["stats"].items():
            arrays[f"L{layer}_{kind}_projection_mean"] = np.array(
                stats["projection_mean"], dtype=np.float32
            )
            arrays[f"L{layer}_{kind}_projection_std"] = np.array(
                stats["projection_std"], dtype=np.float32
            )
    np.savez_compressed(path, **arrays)


def summarize_erasure_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    def rate(subset: list[dict[str, Any]], key: str) -> float | None:
        return sum(bool(row[key]) for row in subset) / len(subset) if subset else None

    summary: dict[str, Any] = {}
    row_p: dict[str, dict[int, float]] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in condition_rows:
            by_row[int(row["source_row_index"])].append(row)
        row_p[condition] = {
            row_index: rate(samples, "is_correct_strong") for row_index, samples in by_row.items()
        }
        by_original = {}
        for original in (False, True):
            subset = [
                row
                for row in condition_rows
                if bool(row.get("original_is_correct_strong")) is original
            ]
            if subset:
                by_original[str(original).lower()] = {
                    "n": len(subset),
                    "strong_accuracy": rate(subset, "is_correct_strong"),
                    "parse_fail_rate": rate(subset, "parse_failed"),
                }
        summary[condition] = {
            "n_generations": len(condition_rows),
            "n_rows": len(by_row),
            "strong_accuracy": rate(condition_rows, "is_correct_strong"),
            "weak_accuracy": rate(condition_rows, "is_correct_weak"),
            "parse_fail_rate": rate(condition_rows, "parse_failed"),
            "mean_quality": (
                sum(float(row["quality_score"]) for row in condition_rows) / len(condition_rows)
                if condition_rows
                else None
            ),
            "by_original_is_correct_strong": by_original,
        }

    baseline_p = row_p.get("baseline", {})
    paired: dict[str, Any] = {}
    for condition, p_by_row in sorted(row_p.items()):
        if condition == "baseline":
            continue
        deltas = [
            p_by_row[row_index] - baseline_p[row_index]
            for row_index in p_by_row
            if row_index in baseline_p
        ]
        if not deltas:
            continue
        false_to_true = sum(
            baseline_p[row_index] == 0.0 and p_by_row[row_index] > 0.0
            for row_index in p_by_row
            if row_index in baseline_p
        )
        true_to_false = sum(
            baseline_p[row_index] == 1.0 and p_by_row[row_index] < 1.0
            for row_index in p_by_row
            if row_index in baseline_p
        )
        paired[condition] = {
            "paired_n": len(deltas),
            "mean_delta_p_strong": float(np.mean(deltas)),
            "std_delta_p_strong": float(np.std(deltas, ddof=0)),
            "rows_degraded": int(sum(delta < 0 for delta in deltas)),
            "rows_improved": int(sum(delta > 0 for delta in deltas)),
            "false_to_true": int(false_to_true),
            "true_to_false": int(true_to_false),
        }
    return {"by_condition": summary, "paired_vs_baseline": paired}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="15,30,40,45,53")
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=4)
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument(
        "--row-shard",
        default=None,
        help="Optional 'i/n' slice of the selected rows so long sampled runs fit the 4h wall limit.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260427)
    parser.add_argument("--probe-seed", type=int, default=20260472)
    parser.add_argument("--orthogonal-seed", type=int, default=20260545)
    parser.add_argument("--gaussian-seed", type=int, default=20260604)
    parser.add_argument("--c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--conditions", default="baseline,erase_raw,erase_orthogonal,erase_gaussian")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/subspace_erasure_27b_property.jsonl"))
    parser.add_argument("--direction-output", type=Path, default=Path("results/stage2/erasure/subspace_erasure_27b_property_directions.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/subspace_erasure_27b_property.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    if args.samples_per_row < 1:
        raise ValueError("--samples-per-row must be >= 1")
    if args.samples_per_row > 1 and not args.do_sample:
        raise ValueError("--samples-per-row > 1 requires --do-sample")
    layers = parse_int_list(args.layers)
    condition_plan = make_condition_plan(parse_condition_kinds(args.conditions))
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)

    print("Multi-layer subspace erasure", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layers={layers}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"samples_per_row={args.samples_per_row}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)

    selected_rows, selection_summary = select_balanced_stage1_rows(
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=parse_int_list(args.heights),
        per_height_label=args.per_height_label,
        seed=args.selection_seed,
        target_split="test",
    )
    if args.row_shard:
        shard_index, shard_count = (int(part) for part in args.row_shard.split("/"))
        if not (0 <= shard_index < shard_count):
            raise ValueError(f"invalid --row-shard {args.row_shard!r}")
        selected_rows = selected_rows[shard_index::shard_count]
        selection_summary["row_shard"] = args.row_shard
        selection_summary["shard_rows"] = len(selected_rows)
    total_generations = len(selected_rows) * len(condition_plan) * args.samples_per_row
    print(
        f"selected_rows={len(selected_rows)} total_generations={total_generations} "
        f"selection={selection_summary}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selection": selection_summary,
                    "conditions": [condition.__dict__ for condition in condition_plan],
                    "layers": layers,
                    "total_generations": total_generations,
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            ),
            flush=True,
        )
        return 0

    by_layer = build_layer_directions(
        layers=layers,
        activation_dir=args.activation_dir,
        model_key=args.model_key,
        task=args.task,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        probe_seed=args.probe_seed,
        orthogonal_seed=args.orthogonal_seed,
        gaussian_seed=args.gaussian_seed,
        c_values=tuple(float(part) for part in args.c_values.split(",")),
        max_iter=args.max_iter,
        solver=args.solver,
    )
    save_layer_direction_artifact(args.direction_output, by_layer)

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(
        f"scorer_preflight: strong={scorer_preflight['is_correct_strong']} "
        f"parse_failed={scorer_preflight['parse_failed']}",
        flush=True,
    )

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    hook_names = validate_hooks(model, layers)
    hook_name_by_layer = dict(zip(layers, hook_names))
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    print(f"using_hooks={hook_name_by_layer}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            prompt_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if len(token_ids) > args.n_ctx:
                raise ValueError(
                    f"row {stage1_row['row_index']} prompt exceeds n_ctx={args.n_ctx}: {len(token_ids)}"
                )
            print(
                f"row {row_idx}/{len(selected_rows)} source_row={stage1_row['row_index']} "
                f"h={stage1_row.get('height')} original_strong={stage1_row.get('is_correct_strong')} "
                f"prompt_tokens={len(token_ids)}",
                flush=True,
            )
            for condition_index, condition in enumerate(condition_plan):
                for sample_index in range(args.samples_per_row):
                    if args.do_sample:
                        torch.manual_seed(
                            args.selection_seed
                            + int(stage1_row["row_index"]) * 10007
                            + condition_index * 101
                            + sample_index
                        )
                    hook_states: dict[int, dict[str, Any]] = {}
                    fwd_hooks = []
                    if condition.vector_kind is not None:
                        for layer in layers:
                            entry = by_layer[layer]
                            stats = entry["stats"][condition.vector_kind]
                            hook_fn, hook_state = make_erasure_hook(
                                vector=entry["vectors"][condition.vector_kind],
                                projection_mean=stats["projection_mean"],
                                projection_std=stats["projection_std"],
                            )
                            hook_states[layer] = hook_state
                            fwd_hooks.append((hook_name_by_layer[layer], hook_fn))
                    with model.hooks(fwd_hooks=fwd_hooks):
                        new_ids, reply = generate_one(
                            model=model,
                            token_ids=token_ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype,
                        )
                    score = score_reply(stage1_row, reply)
                    output_row = {
                        "schema_version": 1,
                        "source_file": source_file,
                        "source_row_index": int(stage1_row["row_index"]),
                        "example_id": stage1_row.get("example_id"),
                        "task": stage1_row.get("task"),
                        "height": stage1_row.get("height"),
                        "model": args.model,
                        "original_is_correct_strong": bool(stage1_row.get("is_correct_strong")),
                        "original_is_correct_weak": bool(stage1_row.get("is_correct_weak")),
                        "original_parse_failed": bool(stage1_row.get("parse_failed")),
                        "condition": condition.label,
                        "sample_index": sample_index,
                        "method": "multi_layer_subspace_erasure",
                        "target_variable": "free_form_correctness",
                        "representation_type": "raw_direction",
                        "erasure_kind": condition.vector_kind,
                        "erasure_layers": layers,
                        "prompt_token_count": len(token_ids),
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        "hook_summary": summarize_hook_states(hook_states),
                        **score,
                    }
                    rows.append(output_row)
                    fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                    print(
                        f"  {condition.label}[{sample_index}]: strong={output_row['is_correct_strong']} "
                        f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                        f"new_tokens={len(new_ids)}",
                        flush=True,
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    erasure_summary = summarize_erasure_rows(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_subspace_erasure.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "free_form_correctness",
        "split": args.split_family,
        "site_or_layer": ",".join(f"L{layer}" for layer in layers),
        "method": "multi_layer_subspace_erasure",
        "representation_type": "raw_direction",
        "layers": layers,
        "hook_names": hook_name_by_layer,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "direction_output": str(args.direction_output),
        "out_jsonl": str(args.out_jsonl),
        "probe_directions": {
            f"L{layer}": serializable_direction_summary(entry["direction"])
            for layer, entry in by_layer.items()
        },
        "control_projection_stats": {
            f"L{layer}": {kind: stats for kind, stats in entry["stats"].items() if kind != "raw"}
            for layer, entry in by_layer.items()
        },
        "selection": selection_summary,
        "generation": {
            "conditions": [condition.__dict__ for condition in condition_plan],
            "samples_per_row": args.samples_per_row,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "summary": erasure_summary,
        "baseline_metrics": erasure_summary["by_condition"].get("baseline"),
        "intervention_metrics": {
            condition: metrics
            for condition, metrics in erasure_summary["by_condition"].items()
            if condition != "baseline"
        },
        "paired_flips": erasure_summary["paired_vs_baseline"],
        "n": len(rows),
        "controls": [
            "regenerated_baseline",
            "orthogonal_direction",
            "matched_gaussian_noise",
            "positive_control",
        ],
        "causal_abstraction_claim": (
            "Necessity test: mean-ablates the per-layer raw correctness direction at every "
            "position across all listed residual layers during prompt processing and decode. "
            "A necessity claim requires baseline-correct rows to degrade under raw erasure "
            "beyond orthogonal and matched-Gaussian erasure; flat accuracy under raw erasure "
            "supports epiphenomenality of the correctness readout."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
