#!/usr/bin/env python3
"""Steer Qwen with explicit Qwen-Scope residual SAE decoder columns.

This is the Qwen/Hugging Face analogue of
``stage2_steer_transcoder_features.py``. It tests shortlisted residual SAE
features one at a time by adding the corresponding Qwen Scope decoder column at
``model.model.layers[L]`` during generation.
"""

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
from safetensors.torch import load_file

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
from scripts.stage2_qwen_steer_sparse_probe_bundle import load_qwen_decoder_rows  # noqa: E402
from scripts.stage2_steer_sparse_probe_bundle import feature_prefix  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.stage2_probes import read_json, read_jsonl, read_split_assignments  # noqa: E402
from src.stage2_steering import (  # noqa: E402
    parse_float_list,
    parse_int_list,
    score_reply,
    select_balanced_stage1_rows,
    strength_label,
    summarize_steering_rows,
)


SCALE_STATS = ("mean_nonzero", "p95_nonzero", "max", "unit")
CONTROL_MODES = ("none", "random_feature")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def activation_vector(top_indices: torch.Tensor, top_values: torch.Tensor, feature: int) -> np.ndarray:
    matches = top_indices == int(feature)
    row_idx, col_idx = matches.nonzero(as_tuple=True)
    values = np.zeros(int(top_indices.shape[0]), dtype=np.float32)
    if int(row_idx.numel()) > 0:
        values[row_idx.cpu().numpy()] = top_values[row_idx, col_idx].float().cpu().numpy()
    return values


def feature_activation_stats(
    *,
    feature_prefix_path: Path,
    features: list[int],
    splits_path: Path,
    split_family: str,
    scale_stat: str,
) -> dict[str, Any]:
    tensors = load_file(feature_prefix_path.with_suffix(".safetensors"))
    top_indices = tensors["top_indices"].long()
    top_values = tensors["top_values"].float()
    sidecar = read_jsonl(feature_prefix_path.with_suffix(".example_ids.jsonl"))
    meta = read_json(feature_prefix_path.with_suffix(".meta.json"))
    if len(sidecar) != int(top_indices.shape[0]):
        raise ValueError(f"{feature_prefix_path} rows {top_indices.shape[0]} != sidecar rows {len(sidecar)}")

    source_file = meta["source_activation_meta"]["jsonl_path"]
    assignments = read_split_assignments(splits_path)
    train_indices: list[int] = []
    kept_indices: list[int] = []
    for idx, row in enumerate(sidecar):
        if row.get("parse_failed"):
            continue
        kept_indices.append(idx)
        assignment = assignments.get((source_file, int(row["row_index"])))
        if assignment is not None and assignment.get(f"{split_family}_split") == "train":
            train_indices.append(idx)
    if not train_indices:
        raise ValueError(f"no kept train rows found for {feature_prefix_path}")

    train_idx = np.asarray(train_indices, dtype=np.int64)
    kept_idx = np.asarray(kept_indices, dtype=np.int64)
    out: dict[str, Any] = {
        "feature_prefix": str(feature_prefix_path),
        "meta_path": str(feature_prefix_path.with_suffix(".meta.json")),
        "feature_file": str(feature_prefix_path.with_suffix(".safetensors")),
        "source_file": source_file,
        "split_family": split_family,
        "scale_stat": scale_stat,
        "kept_rows": len(kept_indices),
        "kept_train_rows": len(train_indices),
        "features": {},
    }
    for feature in features:
        values = activation_vector(top_indices, top_values, feature)
        train_values = values[train_idx]
        kept_values = values[kept_idx]
        train_nonzero = train_values[train_values != 0.0]
        kept_nonzero = kept_values[kept_values != 0.0]
        if scale_stat == "unit":
            scale = 1.0
        elif train_nonzero.size == 0:
            raise ValueError(f"feature {feature} is never active in kept train rows")
        elif scale_stat == "mean_nonzero":
            scale = float(train_nonzero.mean())
        elif scale_stat == "p95_nonzero":
            scale = float(np.quantile(train_nonzero, 0.95))
        elif scale_stat == "max":
            scale = float(train_nonzero.max())
        else:
            raise ValueError(f"unknown scale stat {scale_stat!r}")
        out["features"][str(feature)] = {
            "feature": int(feature),
            "scale": scale,
            "train_density": float(train_nonzero.size / train_values.size),
            "kept_density": float(kept_nonzero.size / kept_values.size),
            "train_nonzero_n": int(train_nonzero.size),
            "kept_nonzero_n": int(kept_nonzero.size),
            "train_mean_nonzero": float(train_nonzero.mean()) if train_nonzero.size else None,
            "train_p95_nonzero": float(np.quantile(train_nonzero, 0.95)) if train_nonzero.size else None,
            "train_max": float(train_nonzero.max()) if train_nonzero.size else None,
            "kept_mean_nonzero": float(kept_nonzero.mean()) if kept_nonzero.size else None,
            "kept_max": float(kept_nonzero.max()) if kept_nonzero.size else None,
        }
    return out


def sample_control_features(*, d_sae: int, features: list[int], seed: int) -> dict[int, int]:
    blocked = set(int(feature) for feature in features)
    rng = random.Random(seed)
    controls: dict[int, int] = {}
    for feature in features:
        for _ in range(10_000):
            candidate = rng.randrange(d_sae)
            if candidate not in blocked and candidate not in controls.values():
                controls[int(feature)] = int(candidate)
                blocked.add(candidate)
                break
        else:
            raise ValueError("failed to sample a random control feature")
    return controls


def condition_rows(
    *,
    features: list[int],
    strengths: tuple[float, ...],
    control_features: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "condition": "baseline",
            "condition_kind": "baseline",
            "feature": None,
            "source_feature": None,
            "strength": 0.0,
        }
    ]
    for feature in features:
        for strength in strengths:
            if strength == 0:
                continue
            rows.append(
                {
                    "condition": f"feature_{feature}_{strength_label(strength)}scale",
                    "condition_kind": "feature",
                    "feature": int(feature),
                    "source_feature": int(feature),
                    "strength": float(strength),
                }
            )
        if feature in control_features:
            control = control_features[feature]
            for strength in strengths:
                if strength == 0:
                    continue
                rows.append(
                    {
                        "condition": f"random_for_{feature}_{control}_{strength_label(strength)}scale",
                        "condition_kind": "random_feature",
                        "feature": int(control),
                        "source_feature": int(feature),
                        "strength": float(strength),
                    }
                )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--model-key", default="qwen35_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=53)
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--feature-dir", type=Path, default=Path("results/stage2/sae_features"))
    parser.add_argument("--sae-repo-id", default="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100")
    parser.add_argument("--sae-id", default="qwenscope_qwen35_27b_w80k_l0_100")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--features", type=parse_int_list, required=True)
    parser.add_argument("--scale-stat", choices=SCALE_STATS, default="mean_nonzero")
    parser.add_argument("--control-mode", choices=CONTROL_MODES, default="random_feature")
    parser.add_argument("--control-seed", type=int, default=20260554)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260523)
    parser.add_argument("--strengths", default="-0.25,0.25")
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

    print("Qwen Scope single-feature steering", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task} layer={args.layer}", flush=True)
    print(f"feature_prefix={prefix}", flush=True)
    print(f"features={args.features} strengths={strengths}", flush=True)
    print(f"scale_stat={args.scale_stat} control_mode={args.control_mode}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    activation_stats = feature_activation_stats(
        feature_prefix_path=prefix,
        features=args.features,
        splits_path=args.splits,
        split_family=args.split_family,
        scale_stat=args.scale_stat,
    )
    source_from_meta = activation_stats["source_file"]
    if source_from_meta != source_file:
        raise ValueError(f"feature source {source_from_meta} does not match requested {source_file}")

    decoder_rows, decoder_summary = load_qwen_decoder_rows(
        repo_id=args.sae_repo_id,
        layer=args.layer,
        revision=args.hf_revision,
        local_files_only=args.local_files_only,
        feature_ids=args.features,
    )
    control_features = (
        sample_control_features(
            d_sae=int(decoder_summary["d_sae"]),
            features=args.features,
            seed=args.control_seed,
        )
        if args.control_mode == "random_feature"
        else {}
    )
    if control_features:
        control_rows, control_summary = load_qwen_decoder_rows(
            repo_id=args.sae_repo_id,
            layer=args.layer,
            revision=args.hf_revision,
            local_files_only=args.local_files_only,
            feature_ids=sorted(control_features.values()),
        )
        decoder_rows.update(control_rows)
        decoder_summary["control_decoder_summary"] = control_summary

    for feature, stats in activation_stats["features"].items():
        print(
            f"feature {feature}: scale={stats['scale']:.4f} "
            f"train_density={stats['train_density']:.4f} train_nonzero={stats['train_nonzero_n']}",
            flush=True,
        )
    if control_features:
        print(f"random_feature_controls={control_features}", flush=True)

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
    print(f"selected_rows={len(selected_rows)} available_counts={selection_summary['available_counts']}", flush=True)
    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)
    scorer_preflight = score_reply(selected_rows[0], selected_rows[0]["ground_truth"])
    print(
        "scorer_preflight: "
        f"strong={scorer_preflight['is_correct_strong']} parse_failed={scorer_preflight['parse_failed']}",
        flush=True,
    )

    plan = condition_rows(features=args.features, strengths=strengths, control_features=control_features)
    if args.dry_run:
        payload = {
            "activation_stats": activation_stats,
            "decoder_summary": decoder_summary,
            "selection": selection_summary,
            "conditions": plan,
            "chat_template_kwargs": chat_template_kwargs or {},
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
                scale = 0.0
                delta = 0.0
                decoder_norm = None
                try:
                    if condition["condition_kind"] != "baseline":
                        feature = int(condition["feature"])
                        source_feature = int(condition["source_feature"])
                        scale = float(activation_stats["features"][str(source_feature)]["scale"])
                        delta = float(condition["strength"]) * scale
                        decoder_vector = decoder_rows[feature]
                        decoder_norm = float(np.linalg.norm(decoder_vector))
                        hook_fn, hook_state = make_hf_steering_hook(
                            vector=decoder_vector,
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
                    "condition_kind": condition["condition_kind"],
                    "feature": condition["feature"],
                    "source_feature": condition["source_feature"],
                    "strength": condition["strength"],
                    "scale": scale,
                    "intervention_delta_l2": abs(delta) * decoder_norm if decoder_norm is not None else 0.0,
                    "decoder_row_norm": decoder_norm,
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
        "script": "scripts/stage2_qwen_steer_single_features.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "layer": args.layer,
        "hook_name": f"model.model.layers.{args.layer}.output",
        "activation_site": args.activation_site,
        "feature_prefix": str(prefix),
        "sae_repo_id": args.sae_repo_id,
        "sae_id": args.sae_id,
        "top_k": args.top_k,
        "features": args.features,
        "control_mode": args.control_mode,
        "control_features": {str(k): v for k, v in control_features.items()},
        "hf_revision_requested": args.hf_revision,
        "decoder_summary": decoder_summary,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "split_family": args.split_family,
        "out_jsonl": str(args.out_jsonl),
        "selection": selection_summary,
        "activation_stats": activation_stats,
        "generation": {
            "conditions": plan,
            "strengths": list(strengths),
            "scale_stat": args.scale_stat,
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
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
