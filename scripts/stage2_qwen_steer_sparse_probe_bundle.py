#!/usr/bin/env python3
"""Steer Qwen with a Qwen-Scope decoder-row bundle from a sparse probe.

This is the Hugging Face/Qwen-Scope analogue of
``stage2_steer_sparse_probe_bundle.py``. It trains the same sparse correctness
probe over cached Qwen Scope top-k features, combines selected Qwen Scope
decoder columns into bundle/control directions, and applies them at
``model.model.layers[L]`` during generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_qwen_patch_hf import (  # noqa: E402
    load_hf_model,
    render_tokens,
    torch_dtype,
    validate_hf_layers,
)
from scripts.stage2_qwen_steer_raw_direction import (  # noqa: E402
    generate_one,
    make_hf_steering_hook,
)
from scripts.stage2_steer_sparse_probe_bundle import (  # noqa: E402
    WEIGHT_KEYS,
    condition_plan,
    feature_prefix,
    load_sparse_feature_dataset,
    make_bundle_variants,
    parse_c_values,
    parse_control_kinds,
    sample_random_features,
    train_projection_scale,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.qwen_scope import qwen_scope_filename  # noqa: E402
from src.stage2_probes import DEFAULT_C_VALUES, read_split_assignments, split_indices_from_assignments  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    parse_float_list,
    parse_int_list,
    score_reply,
    select_balanced_stage1_rows,
    summarize_steering_rows,
    train_sparse_probe_bundle_direction,
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def load_qwen_decoder_rows(
    *,
    repo_id: str,
    layer: int,
    revision: str,
    local_files_only: bool,
    feature_ids: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=qwen_scope_filename(layer),
            revision=revision,
            local_files_only=local_files_only,
        )
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "W_dec" not in state:
        raise ValueError(f"{checkpoint_path} does not contain Qwen Scope W_dec")
    w_dec = state["W_dec"]
    if w_dec.ndim != 2:
        raise ValueError(f"{checkpoint_path} W_dec has shape {tuple(w_dec.shape)}")
    d_model, d_sae = int(w_dec.shape[0]), int(w_dec.shape[1])
    rows: dict[int, np.ndarray] = {}
    for feature in sorted(set(int(feature) for feature in feature_ids)):
        if feature < 0 or feature >= d_sae:
            raise ValueError(f"feature {feature} outside decoder width {d_sae}")
        rows[feature] = w_dec[:, feature].float().cpu().numpy()
    return rows, {
        "checkpoint_path": str(checkpoint_path),
        "repo_id": repo_id,
        "revision": revision,
        "local_files_only": local_files_only,
        "d_model": d_model,
        "d_sae": d_sae,
        "loaded_feature_rows": len(rows),
    }


def save_direction_artifact(path: Path, variants: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"{name}_unit_direction": value["unit_direction"] for name, value in variants.items()},
        **{f"{name}_raw_direction": value["raw_direction"] for name, value in variants.items()},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model-key", default="qwen35_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--projection-activation-site", default="resid_post")
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--feature-dir", type=Path, default=Path("results/stage2/sae_features"))
    parser.add_argument("--sae-repo-id", default="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100")
    parser.add_argument("--sae-id", default="qwenscope_qwen35_27b_w80k_l0_100")
    parser.add_argument("--top-k", type=int, default=100)
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
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260523)
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
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--direction-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    load_env()
    args = build_arg_parser().parse_args()
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
    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else None

    print("Qwen Scope sparse-probe bundle steering", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer}", flush=True)
    print(f"feature_prefix={prefix}", flush=True)
    print(f"top_positive={args.top_positive} top_negative={args.top_negative}", flush=True)
    print(f"controls={args.controls} strengths={strengths}", flush=True)
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

    d_sae = int(dataset["meta"]["sae_cfg"]["d_sae"])
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
    decoder_rows, decoder_summary = load_qwen_decoder_rows(
        repo_id=args.sae_repo_id,
        layer=args.layer,
        revision=args.hf_revision,
        local_files_only=args.local_files_only,
        feature_ids=decoder_ids,
    )
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
    save_direction_artifact(args.direction_output, variants)

    plan = condition_plan(strengths=strengths, variants=list(variants))
    if args.dry_run:
        payload = {
            "bundle_fit": bundle_fit,
            "decoder_summary": decoder_summary,
            "projection_scale": projection,
            "selection": selection_summary,
            "conditions": plan,
            "chat_template_kwargs": chat_template_kwargs or {},
            "direction_output": str(args.direction_output),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
        return 0

    model, tokenizer = load_hf_model(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        device=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    validate_hf_layers(model, [args.layer])
    print(f"using_hook=model.model.layers.{args.layer}.output", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            token_ids = render_tokens(
                tokenizer=tokenizer,
                row=stage1_row,
                model_name=args.model,
                chat_template_kwargs=chat_template_kwargs,
            )
            if len(token_ids) > args.n_ctx:
                raise ValueError(f"row {stage1_row['row_index']} exceeds n_ctx={args.n_ctx}: {len(token_ids)}")
            print(
                f"row {row_idx}/{len(selected_rows)} "
                f"source_row={stage1_row['row_index']} h={stage1_row['height']} "
                f"original_correct={stage1_row['is_correct_strong']} prompt_tokens={len(token_ids)}",
                flush=True,
            )
            for condition in plan:
                hook_state = {"calls": 0, "applications": 0}
                handle = None
                delta = 0.0
                vector_norm = None
                try:
                    if condition["direction_kind"] is not None:
                        variant = variants[condition["direction_kind"]]
                        delta = float(condition["strength_sd"]) * projection_std
                        vector = variant["unit_direction"]
                        vector_norm = float(np.linalg.norm(vector))
                        hook_fn, hook_state = make_hf_steering_hook(
                            vector=vector,
                            delta=delta,
                            scope=args.intervention_scope,
                        )
                        handle = model.model.layers[args.layer].register_forward_hook(hook_fn)
                    new_ids, reply = generate_one(
                        model=model,
                        token_ids=token_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        stop_at_eos=args.stop_at_eos,
                    )
                finally:
                    if handle is not None:
                        handle.remove()

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

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_qwen_steer_sparse_probe_bundle.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "layer": args.layer,
        "hook_name": f"model.model.layers.{args.layer}.output",
        "activation_site": args.activation_site,
        "projection_activation_site": args.projection_activation_site,
        "feature_prefix": str(prefix),
        "sae_repo_id": args.sae_repo_id,
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
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "device_map": args.device_map,
            "chat_template_kwargs": chat_template_kwargs or {},
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
