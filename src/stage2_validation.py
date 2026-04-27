"""Stage 2 activation-input and artifact validation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .activations import EncodedExample, encode_stage1_rows, read_stage1_rows, sha256_file
from .messages import build_messages


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    return sha256_text(json.dumps(token_ids, separators=(",", ":")))


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def count_jsonl_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def hf_cache_dir(hf_cache: Path | None) -> str | None:
    if hf_cache is None:
        env_home = os.environ.get("HF_HOME")
        if not env_home:
            return None
        root = Path(env_home)
    else:
        root = hf_cache
    return str(root if root.name == "hub" else root / "hub")


def commit_from_obj(obj: Any) -> str | None:
    value = getattr(obj, "_commit_hash", None)
    if value:
        return str(value)
    init_kwargs = getattr(obj, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
        return str(init_kwargs["_commit_hash"])
    return None


def add_check(
    report: dict[str, Any],
    name: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    if status not in {"ok", "warning", "skipped", "failed"}:
        raise ValueError(f"unknown check status: {status}")
    report["checks"].append({"name": name, "status": status, "details": details or {}})
    if status == "failed":
        report["errors"].append(name)
    elif status == "warning":
        report["warnings"].append(name)


def find_stage1_invariant(invariants: dict[str, Any], jsonl_path: Path) -> dict[str, Any] | None:
    stage1 = invariants.get("stage1_jsonls", {})
    candidates = [
        display_path(jsonl_path),
        str(jsonl_path),
        str(jsonl_path.resolve()),
    ]
    for candidate in candidates:
        if candidate in stage1:
            return stage1[candidate]
    return None


def validate_stage1_jsonl(
    report: dict[str, Any],
    *,
    jsonl_path: Path,
    invariants: dict[str, Any],
) -> None:
    observed_sha = sha256_file(jsonl_path)
    observed_rows = count_jsonl_rows(jsonl_path)
    expected = find_stage1_invariant(invariants, jsonl_path)
    details = {
        "jsonl_path": display_path(jsonl_path),
        "observed_sha256": observed_sha,
        "observed_rows": observed_rows,
    }
    if expected is None:
        add_check(report, "stage1_jsonl_invariants", "failed", details)
        return
    details["expected_sha256"] = expected.get("sha256")
    details["expected_rows"] = expected.get("rows")
    ok = observed_sha == expected.get("sha256") and observed_rows == expected.get("rows")
    add_check(report, "stage1_jsonl_invariants", "ok" if ok else "failed", details)


def load_tokenizer_and_config(
    model_name: str,
    *,
    hf_cache: Path | None,
    revision: str | None,
    local_files_only: bool,
):
    from transformers import AutoConfig, AutoTokenizer

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    cache_dir = hf_cache_dir(hf_cache)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    return tokenizer, config


def validate_model_and_tokenizer(
    report: dict[str, Any],
    *,
    model_name: str,
    invariants: dict[str, Any],
    hf_cache: Path | None,
    local_files_only: bool,
):
    model_invariant = invariants.get("models", {}).get(model_name)
    if model_invariant is None:
        add_check(
            report,
            "model_invariants_present",
            "failed",
            {"model_name": model_name, "available_models": sorted(invariants.get("models", {}))},
        )
        return None, None

    revision = model_invariant.get("model_revision_hash") or model_invariant.get("tokenizer_revision_hash")
    tokenizer, config = load_tokenizer_and_config(
        model_name,
        hf_cache=hf_cache,
        revision=revision,
        local_files_only=local_files_only,
    )

    tokenizer_commit = commit_from_obj(tokenizer) or revision
    config_commit = commit_from_obj(config) or revision
    observed_chat_hash = sha256_text(tokenizer.chat_template or "")
    details = {
        "model_name": model_name,
        "expected_model_revision_hash": model_invariant.get("model_revision_hash"),
        "observed_model_revision_hash": config_commit,
        "expected_tokenizer_revision_hash": model_invariant.get("tokenizer_revision_hash"),
        "observed_tokenizer_revision_hash": tokenizer_commit,
        "expected_chat_template_sha256": model_invariant.get("chat_template_sha256"),
        "observed_chat_template_sha256": observed_chat_hash,
        "tokenizer_class": tokenizer.__class__.__name__,
        "config_class": config.__class__.__name__,
        "local_files_only": local_files_only,
    }
    ok = (
        config_commit == model_invariant.get("model_revision_hash")
        and tokenizer_commit == model_invariant.get("tokenizer_revision_hash")
        and observed_chat_hash == model_invariant.get("chat_template_sha256")
    )
    add_check(report, "model_tokenizer_invariants", "ok" if ok else "failed", details)
    return tokenizer, config


def validate_message_builder(report: dict[str, Any], *, model_name: str) -> None:
    messages = build_messages("system", "user", model_name)
    if "gemma" in model_name.lower():
        expected = [{"role": "user", "content": "system\n\nuser"}]
    else:
        expected = [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    add_check(
        report,
        "message_builder_contract",
        "ok" if messages == expected else "failed",
        {"model_name": model_name, "messages": messages},
    )


def summarize_encoded_examples(examples: list[EncodedExample]) -> dict[str, Any]:
    token_counts = [example.token_count for example in examples]
    return {
        "row_count": len(examples),
        "row_index_first": examples[0].row_index if examples else None,
        "row_index_last": examples[-1].row_index if examples else None,
        "token_count_min": min(token_counts) if token_counts else None,
        "token_count_max": max(token_counts) if token_counts else None,
        "token_count_mean": sum(token_counts) / len(token_counts) if token_counts else None,
        "first_example_id": examples[0].example_id if examples else None,
        "first_token_ids_sha256": token_ids_sha256(examples[0].token_ids) if examples else None,
    }


def validate_prompt_encoding(
    report: dict[str, Any],
    *,
    jsonl_path: Path,
    tokenizer,
    model_name: str,
    n_ctx: int,
    height: int | None,
    limit: int | None,
    skip: int,
    drop_parse_failed: bool,
) -> tuple[list[tuple[int, dict[str, Any]]], list[EncodedExample]]:
    rows = read_stage1_rows(
        jsonl_path,
        height=height,
        limit=limit,
        skip=skip,
        drop_parse_failed=drop_parse_failed,
    )
    examples = encode_stage1_rows(rows, tokenizer=tokenizer, model_name=model_name)
    details = summarize_encoded_examples(examples)
    too_long = [example.row_index for example in examples if example.token_count > n_ctx]
    details.update(
        {
            "n_ctx": n_ctx,
            "too_long_count": len(too_long),
            "too_long_row_indices_sample": too_long[:10],
            "height": height,
            "limit": limit,
            "skip": skip,
            "drop_parse_failed": drop_parse_failed,
        }
    )
    add_check(report, "prompt_encoding", "ok" if examples and not too_long else "failed", details)
    return rows, examples


def infer_task(rows: list[tuple[int, dict[str, Any]]], task: str | None) -> str | None:
    if task is not None:
        return task
    observed = sorted({row.get("task") for _idx, row in rows if row.get("task")})
    if len(observed) == 1:
        return observed[0]
    return None


def compare_sidecar_rows(
    *,
    sidecar_rows: list[dict[str, Any]],
    examples: list[EncodedExample],
    layer: int,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    for idx, (sidecar, example) in enumerate(zip(sidecar_rows, examples, strict=False)):
        expected = {
            "row_index": example.row_index,
            "example_id": example.example_id,
            "height": example.height,
            "task": example.task,
            "model": example.model,
            "is_correct_strong": example.is_correct_strong,
            "parse_failed": example.parse_failed,
            "token_count": example.token_count,
            "last_token_position": example.token_count - 1,
            "hook_name": hook_name,
        }
        for key, expected_value in expected.items():
            if sidecar.get(key) != expected_value:
                mismatches.append(
                    {
                        "sidecar_position": idx,
                        "row_index": example.row_index,
                        "field": key,
                        "expected": expected_value,
                        "observed": sidecar.get(key),
                    }
                )
                break
        if len(mismatches) >= 10:
            break
    return {
        "sidecar_rows": len(sidecar_rows),
        "expected_rows": len(examples),
        "mismatch_count_sampled": len(mismatches),
        "mismatches": mismatches,
    }


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_activation_artifact(
    report: dict[str, Any],
    *,
    prefix: Path,
    layer: int,
    jsonl_sha256: str,
    examples: list[EncodedExample],
) -> None:
    from safetensors.torch import load_file

    safetensor_path = prefix.with_suffix(".safetensors")
    sidecar_path = prefix.with_suffix(".example_ids.jsonl")
    meta_path = prefix.with_suffix(".meta.json")
    paths = [safetensor_path, sidecar_path, meta_path]
    missing = [display_path(path) for path in paths if not path.exists()]
    if missing:
        add_check(
            report,
            f"activation_artifact_L{layer}",
            "failed",
            {"prefix": display_path(prefix), "missing": missing},
        )
        return

    meta = read_json(meta_path)
    tensors = load_file(safetensor_path)
    sidecar_rows = read_jsonl_dicts(sidecar_path)
    activation = tensors.get("activations")
    sidecar_comparison = compare_sidecar_rows(
        sidecar_rows=sidecar_rows,
        examples=examples,
        layer=layer,
    )

    shape = list(activation.shape) if activation is not None else None
    dtype = str(activation.dtype) if activation is not None else None
    details = {
        "prefix": display_path(prefix),
        "activation_shape": shape,
        "activation_dtype": dtype,
        "meta_shape": meta.get("shape"),
        "meta_dtype": meta.get("dtype"),
        "meta_row_count": meta.get("row_count"),
        "meta_jsonl_sha256": meta.get("jsonl_sha256"),
        "expected_jsonl_sha256": jsonl_sha256,
        "meta_hook_name": meta.get("hook_name"),
        "expected_hook_name": f"blocks.{layer}.hook_resid_post",
        "sidecar": sidecar_comparison,
    }
    ok = (
        activation is not None
        and shape == meta.get("shape")
        and shape is not None
        and shape[0] == len(examples)
        and dtype == meta.get("dtype")
        and meta.get("row_count") == len(examples)
        and meta.get("jsonl_sha256") == jsonl_sha256
        and meta.get("hook_name") == f"blocks.{layer}.hook_resid_post"
        and sidecar_comparison["sidecar_rows"] == len(examples)
        and sidecar_comparison["mismatch_count_sampled"] == 0
    )
    add_check(report, f"activation_artifact_L{layer}", "ok" if ok else "failed", details)


def validate_activation_artifacts(
    report: dict[str, Any],
    *,
    activation_dir: Path | None,
    model_key: str,
    task: str | None,
    layers: list[int],
    jsonl_path: Path,
    examples: list[EncodedExample],
) -> None:
    if activation_dir is None or not layers:
        add_check(
            report,
            "activation_artifacts",
            "skipped",
            {"reason": "provide --activation-dir and --layers to validate written artifacts"},
        )
        return
    if task is None:
        add_check(
            report,
            "activation_artifacts",
            "failed",
            {"reason": "could not infer one task; provide --task"},
        )
        return

    jsonl_sha = sha256_file(jsonl_path)
    for layer in layers:
        prefix = activation_dir / f"{model_key}_{task}_L{layer}"
        validate_activation_artifact(
            report,
            prefix=prefix,
            layer=layer,
            jsonl_sha256=jsonl_sha,
            examples=examples,
        )


def finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    failed = [check for check in report["checks"] if check["status"] == "failed"]
    report["status"] = "failed" if failed else "ok"
    report["check_counts"] = {
        status: sum(check["status"] == status for check in report["checks"])
        for status in ("ok", "warning", "skipped", "failed")
    }
    return report


def build_validation_report(
    *,
    jsonl_path: Path,
    model_name: str,
    model_key: str,
    invariants_path: Path,
    hf_cache: Path | None,
    local_files_only: bool,
    n_ctx: int,
    height: int | None,
    limit: int | None,
    skip: int,
    drop_parse_failed: bool,
    task: str | None,
    layers: list[int],
    activation_dir: Path | None,
) -> dict[str, Any]:
    invariants = read_json(invariants_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "inputs": {
            "jsonl_path": display_path(jsonl_path),
            "model_name": model_name,
            "model_key": model_key,
            "invariants_path": display_path(invariants_path),
            "hf_cache": str(hf_cache) if hf_cache is not None else None,
            "local_files_only": local_files_only,
            "n_ctx": n_ctx,
            "height": height,
            "limit": limit,
            "skip": skip,
            "drop_parse_failed": drop_parse_failed,
            "task": task,
            "layers": layers,
            "activation_dir": display_path(activation_dir) if activation_dir is not None else None,
        },
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    validate_stage1_jsonl(report, jsonl_path=jsonl_path, invariants=invariants)
    tokenizer, _config = validate_model_and_tokenizer(
        report,
        model_name=model_name,
        invariants=invariants,
        hf_cache=hf_cache,
        local_files_only=local_files_only,
    )
    validate_message_builder(report, model_name=model_name)

    if tokenizer is None:
        add_check(report, "prompt_encoding", "failed", {"reason": "tokenizer unavailable"})
        return finalize_report(report)

    rows, examples = validate_prompt_encoding(
        report,
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        model_name=model_name,
        n_ctx=n_ctx,
        height=height,
        limit=limit,
        skip=skip,
        drop_parse_failed=drop_parse_failed,
    )
    task = infer_task(rows, task)
    validate_activation_artifacts(
        report,
        activation_dir=activation_dir,
        model_key=model_key,
        task=task,
        layers=layers,
        jsonl_path=jsonl_path,
        examples=examples,
    )

    add_check(
        report,
        "stage1_serving_top5_logits",
        "skipped",
        {"reason": "requires the original Stage 1 serving stack or a live comparable endpoint"},
    )
    add_check(
        report,
        "greedy_output_byte_match",
        "skipped",
        {"reason": "requires the original Stage 1 serving stack or a live comparable endpoint"},
    )
    if examples:
        token_counts = [example.token_count for example in examples]
        report["prompt_token_count"] = {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
            "std": math.sqrt(
                sum((count - (sum(token_counts) / len(token_counts))) ** 2 for count in token_counts)
                / len(token_counts)
            ),
        }
    return finalize_report(report)
