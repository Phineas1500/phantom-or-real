#!/usr/bin/env python3
"""Run a small learned-feature steering pilot for Stage 2.

This steers Gemma Scope 2 affine transcoder features by adding selected decoder
rows to the transcoder target site during generation. For the L45 big-L0
transcoder, the correct target is ``blocks.45.hook_mlp_out``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import input_device_for_model, load_tl_model, render_chat_text  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_paths import activation_stem  # noqa: E402
from src.stage2_probes import read_json, read_jsonl, read_split_assignments  # noqa: E402
from src.stage2_sae import sae_file_name, snapshot_revision_from_path  # noqa: E402
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


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


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
        raise ValueError(f"feature rows {top_indices.shape[0]} != sidecar rows {len(sidecar)}")

    source_file = meta["source_activation_meta"]["jsonl_path"]
    assignments = read_split_assignments(splits_path)
    kept_train_indices = []
    kept_all_indices = []
    for idx, row in enumerate(sidecar):
        if row.get("parse_failed"):
            continue
        kept_all_indices.append(idx)
        assignment = assignments.get((source_file, int(row["row_index"])))
        if assignment is not None and assignment.get(f"{split_family}_split") == "train":
            kept_train_indices.append(idx)

    if not kept_train_indices:
        raise ValueError(f"no kept train rows found for {feature_prefix_path}")

    out: dict[str, Any] = {
        "feature_prefix": str(feature_prefix_path),
        "meta_path": str(feature_prefix_path.with_suffix(".meta.json")),
        "feature_file": str(feature_prefix_path.with_suffix(".safetensors")),
        "source_file": source_file,
        "split_family": split_family,
        "scale_stat": scale_stat,
        "kept_all_rows": len(kept_all_indices),
        "kept_train_rows": len(kept_train_indices),
        "features": {},
    }
    train_idx = np.asarray(kept_train_indices, dtype=np.int64)
    all_idx = np.asarray(kept_all_indices, dtype=np.int64)
    for feature in features:
        values = activation_vector(top_indices, top_values, feature)
        train_values = values[train_idx]
        all_values = values[all_idx]
        train_nonzero = train_values[train_values != 0.0]
        all_nonzero = all_values[all_values != 0.0]
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
            "all_density": float(all_nonzero.size / all_values.size),
            "train_nonzero_n": int(train_nonzero.size),
            "all_nonzero_n": int(all_nonzero.size),
            "train_mean_nonzero": float(train_nonzero.mean()) if train_nonzero.size else None,
            "train_p95_nonzero": float(np.quantile(train_nonzero, 0.95)) if train_nonzero.size else None,
            "train_max": float(train_nonzero.max()) if train_nonzero.size else None,
            "all_mean_nonzero": float(all_nonzero.mean()) if all_nonzero.size else None,
            "all_max": float(all_nonzero.max()) if all_nonzero.size else None,
        }
    return out


def download_transcoder_params(
    *,
    hf_repo_id: str,
    hf_subfolder: str,
    sae_id: str,
    hf_revision: str,
    local_files_only: bool,
) -> tuple[Path, Path, str | None]:
    config_path = Path(
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=sae_file_name(hf_subfolder, sae_id, "config.json"),
            revision=hf_revision,
            local_files_only=local_files_only,
        )
    )
    params_path = Path(
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=sae_file_name(hf_subfolder, sae_id, "params.safetensors"),
            revision=hf_revision,
            local_files_only=local_files_only,
        )
    )
    return config_path, params_path, snapshot_revision_from_path(config_path)


def sample_control_features(
    *,
    d_sae: int,
    features: list[int],
    seed: int,
) -> dict[int, int]:
    blocked = set(features)
    rng = random.Random(seed)
    controls = {}
    for feature in features:
        for _ in range(10_000):
            candidate = rng.randrange(d_sae)
            if candidate not in blocked and candidate not in controls.values():
                controls[feature] = candidate
                blocked.add(candidate)
                break
        else:
            raise ValueError("failed to sample random control feature")
    return controls


def load_decoder_rows(params_path: Path, feature_ids: list[int]) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    vectors: dict[int, np.ndarray] = {}
    with safe_open(params_path, framework="pt", device="cpu") as handle:
        w_dec = handle.get_slice("w_dec")
        d_sae, d_out = w_dec.get_shape()
        for feature_id in feature_ids:
            if feature_id < 0 or feature_id >= d_sae:
                raise ValueError(f"feature {feature_id} outside decoder width {d_sae}")
            row = w_dec[int(feature_id)].float()
            vectors[int(feature_id)] = row.numpy().astype(np.float32)
    norms = {str(feature): float(np.linalg.norm(vector)) for feature, vector in vectors.items()}
    return vectors, {"d_sae": int(d_sae), "d_out": int(d_out), "decoder_row_norms": norms}


def make_steering_hook(
    *,
    vector: np.ndarray,
    delta: float,
    scope: str,
) -> tuple[Any, dict[str, int]]:
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    state = {"calls": 0, "applications": 0}

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        apply = scope == "last_token_each_forward" or state["calls"] == 0
        state["calls"] += 1
        if not apply:
            return act
        key = (str(act.device), act.dtype)
        direction_tensor = cache.get(key)
        if direction_tensor is None:
            direction_tensor = torch.as_tensor(vector, device=act.device, dtype=act.dtype)
            cache[key] = direction_tensor
        act[:, -1, :] = act[:, -1, :] + float(delta) * direction_tensor
        state["applications"] += 1
        return act

    return hook_fn, state


def generate_one(
    *,
    model,
    token_ids: list[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    stop_at_eos: bool,
    cache_dtype: torch.dtype,
) -> tuple[list[int], str]:
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("model has no tokenizer")
    input_device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    previous_default_dtype = torch.get_default_dtype()
    with torch.inference_mode():
        try:
            torch.set_default_dtype(cache_dtype)
            output_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                stop_at_eos=stop_at_eos,
                do_sample=do_sample,
                temperature=temperature,
                prepend_bos=False,
                return_type="tokens",
                verbose=False,
                use_past_kv_cache=True,
            )
        finally:
            torch.set_default_dtype(previous_default_dtype)
    output_ids = output_tokens[0].detach().cpu().tolist()
    new_ids = output_ids[len(token_ids) :]
    reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return new_ids, reply


def condition_rows(
    *,
    features: list[int],
    strengths: tuple[float, ...],
    control_features: dict[int, int],
) -> list[dict[str, Any]]:
    rows = [
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
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", required=True)
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--hook-name", default="blocks.45.hook_mlp_out")
    parser.add_argument("--activation-site", default="mlp_in_weighted")
    parser.add_argument("--feature-dir", type=Path, default=Path("results/stage2/sae_features"))
    parser.add_argument("--sae-id", default="layer_45_width_262k_l0_big_affine")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--features", type=parse_int_list, required=True)
    parser.add_argument("--scale-stat", choices=SCALE_STATS, default="mean_nonzero")
    parser.add_argument("--control-mode", choices=CONTROL_MODES, default="random_feature")
    parser.add_argument("--control-seed", type=int, default=20260551)
    parser.add_argument("--hf-repo-id", default="google/gemma-scope-2-27b-it")
    parser.add_argument("--hf-subfolder", default="transcoder_all")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height-label", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260427)
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
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--out-jsonl", type=Path, required=True)
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
    feature_prefix_path = feature_prefix(
        feature_dir=args.feature_dir,
        model_key=args.model_key,
        task=args.task,
        layer=args.layer,
        activation_site=args.activation_site,
        sae_id=args.sae_id,
        top_k=args.top_k,
    )

    print("Stage 2 learned-feature steering pilot", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layer={args.layer} hook={args.hook_name}", flush=True)
    print(f"features={args.features} strengths={strengths}", flush=True)
    print(f"intervention_scope={args.intervention_scope}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    activation_stats = feature_activation_stats(
        feature_prefix_path=feature_prefix_path,
        features=args.features,
        splits_path=args.splits,
        split_family=args.split_family,
        scale_stat=args.scale_stat,
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
    control_features = (
        sample_control_features(d_sae=d_sae, features=args.features, seed=args.control_seed)
        if args.control_mode == "random_feature"
        else {}
    )
    all_decoder_ids = sorted(set(args.features) | set(control_features.values()))
    decoder_vectors, decoder_summary = load_decoder_rows(params_path, all_decoder_ids)
    for feature, stats in activation_stats["features"].items():
        print(
            f"feature {feature}: scale={stats['scale']:.3f} "
            f"train_density={stats['train_density']:.3f} train_nonzero={stats['train_nonzero_n']}",
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

    plan = condition_rows(features=args.features, strengths=strengths, control_features=control_features)
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
                if condition["condition_kind"] == "baseline":
                    new_ids, reply = generate_one(
                        model=model,
                        token_ids=token_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        stop_at_eos=args.stop_at_eos,
                        cache_dtype=dtype,
                    )
                    scale = 0.0
                    delta = 0.0
                    decoder_norm = None
                else:
                    feature = int(condition["feature"])
                    source_feature = int(condition["source_feature"])
                    scale = float(activation_stats["features"][str(source_feature)]["scale"])
                    delta = float(condition["strength"]) * scale
                    decoder_vector = decoder_vectors[feature]
                    decoder_norm = float(np.linalg.norm(decoder_vector))
                    hook_fn, hook_state = make_steering_hook(
                        vector=decoder_vector,
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
        "script": "scripts/stage2_steer_transcoder_features.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "layer": args.layer,
        "hook_name": args.hook_name,
        "activation_site": args.activation_site,
        "feature_prefix": str(feature_prefix_path),
        "sae_id": args.sae_id,
        "top_k": args.top_k,
        "features": args.features,
        "control_mode": args.control_mode,
        "control_features": {str(k): v for k, v in control_features.items()},
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
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
