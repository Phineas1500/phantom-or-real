#!/usr/bin/env python3
"""Steer with a decoder-row bundle derived from a sparse feature probe."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_steer_transcoder_features import (  # noqa: E402
    download_transcoder_params,
    generate_one,
    json_default,
    load_decoder_rows,
    make_steering_hook,
    package_version,
    torch_dtype,
)
from src.activations import load_tl_model, render_chat_text  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
)
from src.stage2_sae import topk_tensors_to_csr  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    build_weighted_decoder_bundle,
    parse_float_list,
    parse_int_list,
    score_reply,
    select_balanced_stage1_rows,
    strength_label,
    summarize_steering_rows,
    train_sparse_probe_bundle_direction,
)


CONTROL_KINDS = ("shuffled", "random", "orthogonal")
WEIGHT_KEYS = ("standardized_coef", "input_weight")


def parse_control_kinds(value: str) -> list[str]:
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(parsed) - set(CONTROL_KINDS))
    if unknown:
        raise ValueError(f"unknown control kind(s): {unknown}")
    return parsed


def parse_c_values(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("expected at least one C value")
    return parsed


def feature_prefix(
    *,
    feature_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    activation_site: str,
    sae_id: str,
    top_k: int,
) -> Path:
    stem = activation_stem(model_key=model_key, task=task, layer=layer, activation_site=activation_site)
    return feature_dir / f"{stem}_{sae_id}_top{top_k}"


def load_sparse_feature_dataset(*, prefix: Path, drop_parse_failed: bool = True) -> dict[str, Any]:
    meta = read_json(prefix.with_suffix(".meta.json"))
    tensors = load_file(prefix.with_suffix(".safetensors"))
    x_all = topk_tensors_to_csr(
        tensors["top_indices"],
        tensors["top_values"],
        d_sae=int(meta["sae_cfg"]["d_sae"]),
    )
    x_all.eliminate_zeros()
    sidecar_all = read_jsonl(prefix.with_suffix(".example_ids.jsonl"))
    if x_all.shape[0] != len(sidecar_all):
        raise ValueError(f"{prefix} rows {x_all.shape[0]} != sidecar rows {len(sidecar_all)}")
    keep_indices = [
        idx
        for idx, row in enumerate(sidecar_all)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    sidecar = [sidecar_all[idx] for idx in keep_indices]
    return {
        "x": x_all[keep_indices],
        "labels": [int(row["is_correct_strong"]) for row in sidecar],
        "sidecar": sidecar,
        "meta": meta,
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
    }


def train_projection_scale(
    *,
    activation_dir: Path,
    activation_site: str,
    model_key: str,
    task: str,
    layer: int,
    direction: np.ndarray,
    splits_path: Path,
    source_file: str,
    split_family: str,
) -> dict[str, Any]:
    stem = activation_stem(model_key=model_key, task=task, layer=layer, activation_site=activation_site)
    prefix = activation_dir / stem
    activations = load_file(prefix.with_suffix(".safetensors"))["activations"].float().cpu().numpy()
    sidecar = read_jsonl(prefix.with_suffix(".example_ids.jsonl"))
    if activations.shape[0] != len(sidecar):
        raise ValueError(f"{prefix} rows {activations.shape[0]} != sidecar rows {len(sidecar)}")
    assignments = read_split_assignments(splits_path)
    train_indices = [
        idx
        for idx, row in enumerate(sidecar)
        if not row.get("parse_failed")
        and assignments.get((source_file, int(row["row_index"])), {}).get(f"{split_family}_split") == "train"
    ]
    if not train_indices:
        raise ValueError(f"no train rows found for projection scale at {prefix}")
    unit = np.asarray(direction, dtype=np.float64)
    projections = activations[train_indices].astype(np.float64) @ unit
    std = float(projections.std(ddof=0))
    if std == 0.0:
        raise ValueError("bundle projection has zero train standard deviation")
    return {
        "activation_path": str(prefix.with_suffix(".safetensors")),
        "sidecar_path": str(prefix.with_suffix(".example_ids.jsonl")),
        "activation_site": activation_site,
        "train_rows": len(train_indices),
        "projection_mean": float(projections.mean()),
        "projection_std": std,
        "projection_min": float(projections.min()),
        "projection_max": float(projections.max()),
    }


def sample_random_features(*, d_sae: int, blocked: set[int], n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    out = []
    used = set(blocked)
    while len(out) < n:
        candidate = rng.randrange(d_sae)
        if candidate in used:
            continue
        out.append(candidate)
        used.add(candidate)
    return out


def make_bundle_variants(
    *,
    selected_features: list[dict[str, Any]],
    decoder_rows: dict[int, np.ndarray],
    d_sae: int,
    controls: list[str],
    weight_key: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {
        "bundle": build_weighted_decoder_bundle(decoder_rows, selected_features, weight_key=weight_key)
    }
    rng = np.random.default_rng(seed)
    if "shuffled" in controls:
        weights = [float(row[weight_key]) for row in selected_features]
        shuffled = list(rng.permutation(weights))
        shuffled_features = [
            {**row, weight_key: float(shuffled[idx])}
            for idx, row in enumerate(selected_features)
        ]
        variants["shuffled"] = build_weighted_decoder_bundle(
            decoder_rows,
            shuffled_features,
            weight_key=weight_key,
        )
        variants["shuffled"]["selected_features"] = shuffled_features
    if "random" in controls:
        selected_ids = [int(row["feature"]) for row in selected_features]
        random_ids = sample_random_features(
            d_sae=d_sae,
            blocked=set(selected_ids),
            n=len(selected_ids),
            seed=seed + 17,
        )
        random_features = [
            {
                **row,
                "feature": int(random_ids[idx]),
                "source_feature": int(row["feature"]),
            }
            for idx, row in enumerate(selected_features)
        ]
        variants["random"] = build_weighted_decoder_bundle(
            decoder_rows,
            random_features,
            weight_key=weight_key,
        )
        variants["random"]["selected_features"] = random_features
    if "orthogonal" in controls:
        from src.stage2_steering import make_orthogonal_unit_direction

        unit = make_orthogonal_unit_direction(variants["bundle"]["unit_direction"], seed=seed + 29)
        variants["orthogonal"] = {
            "unit_direction": unit,
            "raw_direction": unit,
            "raw_norm": 1.0,
            "weight_key": "orthogonal_random",
            "components": [],
        }
    variants["bundle"]["selected_features"] = selected_features
    return variants


def condition_plan(*, strengths: tuple[float, ...], variants: list[str]) -> list[dict[str, Any]]:
    rows = [{"condition": "baseline", "direction_kind": None, "strength_sd": 0.0}]
    for variant in variants:
        for strength in strengths:
            if strength == 0:
                continue
            rows.append(
                {
                    "condition": f"{variant}_{strength_label(strength)}",
                    "direction_kind": variant,
                    "strength_sd": float(strength),
                }
            )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", required=True)
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--hook-name", default="blocks.45.hook_mlp_out")
    parser.add_argument("--activation-site", default="mlp_in_weighted")
    parser.add_argument("--projection-activation-site", default="mlp_out_hook")
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--feature-dir", type=Path, default=Path("results/stage2/sae_features"))
    parser.add_argument("--sae-id", default="layer_45_width_262k_l0_big_affine")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--top-positive", type=int, default=25)
    parser.add_argument("--top-negative", type=int, default=25)
    parser.add_argument("--min-density", type=float, default=0.02)
    parser.add_argument("--max-density", type=float, default=0.50)
    parser.add_argument("--weight-key", choices=WEIGHT_KEYS, default="standardized_coef")
    parser.add_argument("--c-values", type=parse_c_values, default=DEFAULT_C_VALUES)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--controls", type=parse_control_kinds, default=("shuffled", "random", "orthogonal"))
    parser.add_argument("--control-seed", type=int, default=20260553)
    parser.add_argument("--hf-repo-id", default="google/gemma-scope-2-27b-it")
    parser.add_argument("--hf-subfolder", default="transcoder_all")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260427)
    parser.add_argument("--strengths", default="-0.5,0.5")
    parser.add_argument(
        "--intervention-scope",
        choices=("prompt_only", "last_token_each_forward"),
        default="last_token_each_forward",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--direction-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()
    dtype = torch_dtype(args.dtype)
    heights = parse_int_list(args.heights)
    strengths = parse_float_list(args.strengths)
    source_file = str(args.jsonl)
    prefix = feature_prefix(
        feature_dir=args.feature_dir,
        model_key=args.model_key,
        task=args.task,
        layer=args.layer,
        activation_site=args.activation_site,
        sae_id=args.sae_id,
        top_k=args.top_k,
    )

    print("Stage 2 sparse-probe bundle steering", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer} hook={args.hook_name}", flush=True)
    print(f"feature_prefix={prefix}", flush=True)
    print(f"top_positive={args.top_positive} top_negative={args.top_negative}", flush=True)
    print(f"controls={args.controls} strengths={strengths}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    dataset = load_sparse_feature_dataset(prefix=prefix, drop_parse_failed=True)
    source_from_meta = dataset["meta"]["source_activation_meta"]["jsonl_path"]
    if source_from_meta != source_file:
        raise ValueError(f"feature source {source_from_meta} does not match requested {source_file}")
    split_assignments = read_split_assignments(args.splits)
    splits = split_indices_from_assignments(
        dataset["sidecar"],
        assignments=split_assignments,
        source_file=source_file,
        split_field=f"{args.split_family}_split",
    )
    bundle_fit = train_sparse_probe_bundle_direction(
        x=dataset["x"],
        labels=dataset["labels"],
        splits=splits,
        c_values=args.c_values,
        max_iter=args.max_iter,
        solver=args.solver,
        top_positive=args.top_positive,
        top_negative=args.top_negative,
        min_density=args.min_density,
        max_density=args.max_density,
    )
    print(
        f"sparse_probe: best_c={bundle_fit['best_c']} "
        f"val_auc={bundle_fit['val_auc']:.4f} test_auc={bundle_fit['test_auc']:.4f} "
        f"selected={len(bundle_fit['selected_features'])}",
        flush=True,
    )

    config_path, params_path, hf_snapshot_revision = download_transcoder_params(
        hf_repo_id=args.hf_repo_id,
        hf_subfolder=args.hf_subfolder,
        sae_id=args.sae_id,
        hf_revision=args.hf_revision,
        local_files_only=args.local_files_only,
    )
    with safe_open(params_path, framework="pt", device="cpu") as handle:
        d_sae = int(handle.get_slice("w_dec").get_shape()[0])
    selected_feature_ids = set(bundle_fit["selected_feature_ids"])
    random_ids = (
        sample_random_features(
            d_sae=d_sae,
            blocked=selected_feature_ids,
            n=len(selected_feature_ids),
            seed=args.control_seed + 17,
        )
        if "random" in args.controls
        else []
    )
    decoder_ids = sorted(selected_feature_ids | set(random_ids))
    decoder_rows, decoder_summary = load_decoder_rows(params_path, decoder_ids)
    variants = make_bundle_variants(
        selected_features=bundle_fit["selected_features"],
        decoder_rows=decoder_rows,
        d_sae=d_sae,
        controls=args.controls,
        weight_key=args.weight_key,
        seed=args.control_seed,
    )
    projection = train_projection_scale(
        activation_dir=args.activation_dir,
        activation_site=args.projection_activation_site,
        model_key=args.model_key,
        task=args.task,
        layer=args.layer,
        direction=variants["bundle"]["unit_direction"],
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
    )
    projection_std = float(projection["projection_std"])
    print(
        f"projection_scale: site={args.projection_activation_site} "
        f"std={projection_std:.4f} train_rows={projection['train_rows']}",
        flush=True,
    )

    selected_rows, selection_summary = select_balanced_stage1_rows(
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        source_file=source_file,
        split_family=args.split_family,
        heights=heights,
        per_height_label=args.per_height_label,
        seed=args.selection_seed,
        drop_parse_failed=True,
    )
    print(
        f"selected_rows={len(selected_rows)} "
        f"available_counts={selection_summary['available_counts']}",
        flush=True,
    )
    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(
        "scorer_preflight: "
        f"strong={scorer_preflight['is_correct_strong']} parse_failed={scorer_preflight['parse_failed']}",
        flush=True,
    )

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    if args.hook_name not in model.hook_dict:
        sample = sorted(name for name in model.hook_dict if f"blocks.{args.layer}" in name)[:40]
        raise ValueError(f"missing hook {args.hook_name}; layer hook sample: {sample}")
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    print(f"using_hook={args.hook_name}", flush=True)

    plan = condition_plan(strengths=strengths, variants=list(variants))
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
            print(
                f"row {row_idx}/{len(selected_rows)} "
                f"source_row={stage1_row['row_index']} h={stage1_row['height']} "
                f"original_correct={stage1_row['is_correct_strong']} prompt_tokens={len(token_ids)}",
                flush=True,
            )
            for condition in plan:
                hook_state = {"calls": 0, "applications": 0}
                if condition["direction_kind"] is None:
                    new_ids, reply = generate_one(
                        model=model,
                        token_ids=token_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        stop_at_eos=args.stop_at_eos,
                        cache_dtype=dtype,
                    )
                    delta = 0.0
                    vector_norm = None
                else:
                    variant = variants[condition["direction_kind"]]
                    delta = float(condition["strength_sd"]) * projection_std
                    vector = variant["unit_direction"]
                    vector_norm = float(np.linalg.norm(vector))
                    hook_fn, hook_state = make_steering_hook(
                        vector=vector,
                        delta=delta,
                        scope=args.intervention_scope,
                    )
                    with model.hooks(fwd_hooks=[(args.hook_name, hook_fn)]):
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
                    "original_model": stage1_row.get("model"),
                    "original_is_correct_strong": bool(stage1_row.get("is_correct_strong")),
                    "original_is_correct_weak": bool(stage1_row.get("is_correct_weak")),
                    "original_parse_failed": bool(stage1_row.get("parse_failed")),
                    "condition": condition["condition"],
                    "direction_kind": condition["direction_kind"],
                    "strength_sd": condition["strength_sd"],
                    "projection_std": projection_std,
                    "intervention_delta_l2": abs(delta) * vector_norm if vector_norm is not None else 0.0,
                    "intervention_scope": args.intervention_scope,
                    "hook_calls": int(hook_state["calls"]),
                    "hook_applications": int(hook_state["applications"]),
                    "prompt_token_count": len(token_ids),
                    "generated_token_count": len(new_ids),
                    "model_output": reply,
                    **score,
                }
                rows.append(output_row)
                fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"  {condition['condition']}: strong={output_row['is_correct_strong']} "
                    f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                    f"new_tokens={len(new_ids)} hooks={hook_state['applications']}/{hook_state['calls']}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    args.direction_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.direction_output,
        **{f"{name}_unit_direction": value["unit_direction"] for name, value in variants.items()},
        **{f"{name}_raw_direction": value["raw_direction"] for name, value in variants.items()},
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_steer_sparse_probe_bundle.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "layer": args.layer,
        "hook_name": args.hook_name,
        "activation_site": args.activation_site,
        "projection_activation_site": args.projection_activation_site,
        "feature_prefix": str(prefix),
        "sae_id": args.sae_id,
        "top_k": args.top_k,
        "weight_key": args.weight_key,
        "bundle_fit": bundle_fit,
        "bundle_variants": {
            name: {
                "raw_norm": value["raw_norm"],
                "weight_key": value["weight_key"],
                "components": value["components"],
                "selected_features": value.get("selected_features"),
            }
            for name, value in variants.items()
        },
        "projection_scale": projection,
        "hf_repo_id": args.hf_repo_id,
        "hf_subfolder": args.hf_subfolder,
        "hf_revision_requested": args.hf_revision,
        "hf_snapshot_revision": hf_snapshot_revision,
        "sae_config_file": str(config_path),
        "sae_params_file": str(params_path),
        "decoder_summary": decoder_summary,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "out_jsonl": str(args.out_jsonl),
        "direction_output": str(args.direction_output),
        "selection": selection_summary,
        "feature_dataset": {
            "input_rows": dataset["input_rows"],
            "kept_rows": dataset["kept_rows"],
            "source_file": source_from_meta,
        },
        "generation": {
            "conditions": plan,
            "strengths_sd": list(strengths),
            "controls": args.controls,
            "intervention_scope": args.intervention_scope,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "summary": summarize_steering_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"wrote {args.direction_output}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
