#!/usr/bin/env python3
"""Compute saved-activation logit-lens accessibility profiles for Stage 2."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import parse_int_list, sha256_file, tokenizer_pad_token_id  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.stage2_paths import DEFAULT_ACTIVATION_SITE, activation_stem, normalize_activation_site  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    read_json,
    read_jsonl,
    read_split_assignments,
    split_indices_from_assignments,
    write_json,
)


SCORE_NAMES = (
    "gold_mean_logprob",
    "gold_first_token_logprob",
    "output_mean_logprob",
    "gold_minus_output_mean_logprob",
    "gold_vs_output_first_diff_logit_margin",
    "gold_vs_output_first_diff_logprob_margin",
)


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def module_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_hf_lens_model(model_name: str, model_kwargs: dict[str, Any]):
    import transformers
    from transformers import AutoModelForCausalLM

    errors = []
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ValueError as exc:
        errors.append(f"AutoModelForCausalLM: {exc}")

    for auto_name in ("AutoModelForImageTextToText", "AutoModelForConditionalGeneration"):
        auto_cls = getattr(transformers, auto_name, None)
        if auto_cls is None:
            continue
        try:
            return auto_cls.from_pretrained(model_name, **model_kwargs)
        except ValueError as exc:
            errors.append(f"{auto_name}: {exc}")
    raise ValueError("Unable to load model for LAP logit lens. " + " | ".join(errors))


def text_lm_for_lens(model):
    candidates = [model]
    for attr in ("language_model", "text_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            candidates.append(candidate)
    for candidate in candidates:
        if hasattr(candidate, "lm_head") and hasattr(candidate, "model"):
            inner = getattr(candidate, "model")
            if hasattr(inner, "norm"):
                return candidate
    for candidate in candidates:
        if hasattr(candidate, "lm_head"):
            return candidate
    raise AttributeError("could not find a text LM with lm_head on loaded model")


def final_norm_for_lens(text_lm):
    inner = getattr(text_lm, "model", None)
    if inner is not None and hasattr(inner, "norm"):
        return inner.norm
    if hasattr(text_lm, "norm"):
        return text_lm.norm
    return None


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels) if not math.isnan(float(score))]
    if not pairs:
        return None
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (rank + rank + (j - idx) - 1) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in pairs[idx:j])
        rank += j - idx
        idx = j
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def class_counts(labels: list[int]) -> dict[str, int]:
    positive = int(sum(labels))
    return {
        "n": len(labels),
        "positive_n": positive,
        "negative_n": len(labels) - positive,
    }


def first_nonempty_line(text: str | None) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def candidate_text(row: dict[str, Any], mode: str) -> str:
    if mode == "gold":
        return str(row.get("ground_truth") or row.get("ontology_raw", {}).get("hypotheses") or "").strip()
    if mode == "model_output":
        return first_nonempty_line(row.get("model_output"))
    raise ValueError(f"unknown candidate mode {mode!r}")


def encode_candidate(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def first_diff_pair(left: list[int], right: list[int]) -> tuple[int, int] | None:
    if not left or not right:
        return None
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            return left_id, right_id
    return None


def load_rows_by_index(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open() as f:
        for row_index, line in enumerate(f):
            if line.strip():
                rows[row_index] = __import__("json").loads(line)
    return rows


def tensor_from_safetensor(path: Path) -> torch.Tensor:
    tensors = load_file(path)
    if "activations" in tensors:
        return tensors["activations"]
    if len(tensors) == 1:
        return next(iter(tensors.values()))
    raise KeyError(f"{path} has no 'activations' key and multiple tensors: {sorted(tensors)}")


def build_dataset(
    *,
    source_jsonl: Path,
    sidecar_path: Path,
    tokenizer,
    drop_parse_failed: bool,
) -> dict[str, Any]:
    sidecar_all = read_jsonl(sidecar_path)
    source_rows = load_rows_by_index(source_jsonl)
    keep_indices = [
        idx
        for idx, row in enumerate(sidecar_all)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    sidecar = [sidecar_all[idx] for idx in keep_indices]
    rows = []
    missing = []
    candidate_rows = []
    for row in sidecar:
        row_index = int(row["row_index"])
        source_row = source_rows.get(row_index)
        if source_row is None:
            missing.append(row_index)
            continue
        rows.append(source_row)
        gold_text = candidate_text(source_row, "gold")
        output_text = candidate_text(source_row, "model_output")
        gold_ids = encode_candidate(tokenizer, gold_text)
        output_ids = encode_candidate(tokenizer, output_text)
        candidate_rows.append(
            {
                "row_index": row_index,
                "example_id": row.get("example_id"),
                "height": row.get("height"),
                "is_correct_strong": bool(row.get("is_correct_strong")),
                "parse_failed": bool(row.get("parse_failed")),
                "gold_text": gold_text,
                "output_text": output_text,
                "gold_token_ids": gold_ids,
                "output_token_ids": output_ids,
                "first_diff_token_ids": first_diff_pair(gold_ids, output_ids),
            }
        )
    if missing:
        raise KeyError(f"{len(missing)} sidecar row indices missing from {source_jsonl}: sample={missing[:5]}")
    return {
        "sidecar_all": sidecar_all,
        "keep_indices": keep_indices,
        "sidecar": sidecar,
        "rows": rows,
        "candidate_rows": candidate_rows,
        "labels": [int(row["is_correct_strong"]) for row in sidecar],
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
    }


def split_groups(
    *,
    sidecar: list[dict[str, Any]],
    source_jsonl: Path,
    splits_path: Path | None,
    split_family: str,
) -> dict[str, list[int]]:
    groups = {"all": list(range(len(sidecar)))}
    if splits_path is None:
        return groups
    assignments = read_split_assignments(splits_path)
    groups.update(
        split_indices_from_assignments(
            sidecar,
            assignments=assignments,
            source_file=str(source_jsonl),
            split_field=f"{split_family}_split",
        )
    )
    return groups


def apply_lens(
    *,
    text_lm,
    activations: torch.Tensor,
    candidate_rows: list[dict[str, Any]],
    keep_indices: list[int],
    batch_size: int,
    activation_dtype: torch.dtype,
    apply_final_norm: bool,
) -> dict[str, list[float]]:
    norm = final_norm_for_lens(text_lm)
    norm_device = module_device(norm) if norm is not None else module_device(text_lm.lm_head)
    head_device = module_device(text_lm.lm_head)
    scores = {name: [math.nan] * len(candidate_rows) for name in SCORE_NAMES}
    rows_done = 0
    with torch.inference_mode():
        for start in range(0, len(candidate_rows), batch_size):
            stop = min(start + batch_size, len(candidate_rows))
            batch_indices = keep_indices[start:stop]
            hidden = activations[batch_indices].to(device=norm_device, dtype=activation_dtype)
            if apply_final_norm and norm is not None:
                hidden = norm(hidden)
            logits = text_lm.lm_head(hidden.to(head_device)).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            for local_idx, row in enumerate(candidate_rows[start:stop]):
                gold_ids = row["gold_token_ids"]
                output_ids = row["output_token_ids"]
                if gold_ids:
                    gold_tensor = torch.tensor(gold_ids, dtype=torch.long, device=log_probs.device)
                    scores["gold_mean_logprob"][start + local_idx] = float(log_probs[local_idx, gold_tensor].mean().item())
                    scores["gold_first_token_logprob"][start + local_idx] = float(log_probs[local_idx, gold_ids[0]].item())
                if output_ids:
                    output_tensor = torch.tensor(output_ids, dtype=torch.long, device=log_probs.device)
                    output_mean = float(log_probs[local_idx, output_tensor].mean().item())
                    scores["output_mean_logprob"][start + local_idx] = output_mean
                    gold_mean = scores["gold_mean_logprob"][start + local_idx]
                    if not math.isnan(gold_mean):
                        scores["gold_minus_output_mean_logprob"][start + local_idx] = gold_mean - output_mean
                diff = row["first_diff_token_ids"]
                if diff is not None:
                    gold_id, output_id = diff
                    scores["gold_vs_output_first_diff_logit_margin"][start + local_idx] = float(
                        (logits[local_idx, gold_id] - logits[local_idx, output_id]).item()
                    )
                    scores["gold_vs_output_first_diff_logprob_margin"][start + local_idx] = float(
                        (log_probs[local_idx, gold_id] - log_probs[local_idx, output_id]).item()
                    )
            rows_done = stop
            print(f"lens rows {rows_done}/{len(candidate_rows)}", flush=True)
            del hidden, logits, log_probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return scores


def summarize_score(
    *,
    labels: list[int],
    sidecar: list[dict[str, Any]],
    scores: list[float],
    indices: list[int],
) -> dict[str, Any]:
    kept = [idx for idx in indices if not math.isnan(float(scores[idx]))]
    kept_labels = [labels[idx] for idx in kept]
    kept_scores = [float(scores[idx]) for idx in kept]
    by_height = {}
    for height in sorted({sidecar[idx].get("height") for idx in kept}):
        h_indices = [idx for idx in kept if sidecar[idx].get("height") == height]
        h_labels = [labels[idx] for idx in h_indices]
        h_scores = [float(scores[idx]) for idx in h_indices]
        by_height[f"h{height}"] = {
            **class_counts(h_labels),
            "auc": safe_auc(h_labels, h_scores),
        }
    return {
        **class_counts(kept_labels),
        "coverage_n": len(kept),
        "coverage_rate": len(kept) / len(indices) if indices else None,
        "auc": safe_auc(kept_labels, kept_scores),
        "score_mean": float(np.mean(kept_scores)) if kept_scores else None,
        "score_std": float(np.std(kept_scores)) if kept_scores else None,
        "by_height": by_height,
    }


def summarize_layer(
    *,
    labels: list[int],
    sidecar: list[dict[str, Any]],
    scores: dict[str, list[float]],
    groups: dict[str, list[int]],
    eval_splits: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in eval_splits:
        if split not in groups:
            continue
        indices = groups[split]
        result[split] = {
            score_name: summarize_score(
                labels=labels,
                sidecar=sidecar,
                scores=score_values,
                indices=indices,
            )
            for score_name, score_values in scores.items()
        }
        result[split]["split_counts"] = class_counts([labels[idx] for idx in indices])
    return result


def peak_by_score(results: dict[str, Any], *, eval_split: str) -> dict[str, Any]:
    peaks = {}
    for score_name in SCORE_NAMES:
        best_layer = None
        best_auc = None
        for layer_key, layer_result in results.items():
            auc = layer_result.get(eval_split, {}).get(score_name, {}).get("auc")
            if auc is None:
                continue
            if best_auc is None or auc > best_auc:
                best_layer = layer_key
                best_auc = auc
        peaks[score_name] = {"layer": best_layer, "auc": best_auc}
    return peaks


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--activation-site", default=DEFAULT_ACTIVATION_SITE)
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--eval-splits", default="test,all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--activation-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--drop-parse-failed", action="store_true")
    parser.add_argument("--no-final-norm", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    import transformers

    dtype = torch_dtype(args.dtype)
    activation_dtype = torch_dtype(args.activation_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_pad_token_id(tokenizer)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    model = load_hf_lens_model(args.model, model_kwargs)
    if args.device_map == "none":
        model.to(args.device)
    model.eval()
    text_lm = text_lm_for_lens(model)

    activation_site = normalize_activation_site(args.activation_site)
    first_prefix = args.activation_dir / activation_stem(
        model_key=args.model_key,
        task=args.task,
        layer=args.layers[0],
        activation_site=activation_site,
    )
    dataset = build_dataset(
        source_jsonl=args.jsonl,
        sidecar_path=first_prefix.with_suffix(".example_ids.jsonl"),
        tokenizer=tokenizer,
        drop_parse_failed=args.drop_parse_failed,
    )
    groups = split_groups(
        sidecar=dataset["sidecar"],
        source_jsonl=args.jsonl,
        splits_path=args.splits if args.splits else None,
        split_family=args.split_family,
    )
    eval_splits = [part.strip() for part in args.eval_splits.split(",") if part.strip()]
    started = time.monotonic()
    results: dict[str, Any] = {}
    for layer in args.layers:
        prefix = args.activation_dir / activation_stem(
            model_key=args.model_key,
            task=args.task,
            layer=layer,
            activation_site=activation_site,
        )
        print(f"loading activations {prefix.with_suffix('.safetensors')}", flush=True)
        activations = tensor_from_safetensor(prefix.with_suffix(".safetensors"))
        if activations.shape[0] != dataset["input_rows"]:
            raise ValueError(
                f"{prefix.with_suffix('.safetensors')} rows {activations.shape[0]} != sidecar rows {dataset['input_rows']}"
            )
        scores = apply_lens(
            text_lm=text_lm,
            activations=activations,
            candidate_rows=dataset["candidate_rows"],
            keep_indices=dataset["keep_indices"],
            batch_size=args.batch_size,
            activation_dtype=activation_dtype,
            apply_final_norm=not args.no_final_norm,
        )
        results[f"L{layer}"] = {
            "activation_path": str(prefix.with_suffix(".safetensors")),
            "sidecar_path": str(prefix.with_suffix(".example_ids.jsonl")),
            "scores": summarize_layer(
                labels=dataset["labels"],
                sidecar=dataset["sidecar"],
                scores=scores,
                groups=groups,
                eval_splits=eval_splits,
            ),
        }
        del activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    peak_split = "test" if "test" in eval_splits else eval_splits[0]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_kind": "lap_logit_lens_accessibility",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "gold_vs_foil_margin",
        "representation_type": "raw_direction",
        "method": "saved_activation_logit_lens",
        "jsonl": str(args.jsonl),
        "jsonl_sha256": sha256_file(args.jsonl),
        "activation_dir": str(args.activation_dir),
        "activation_site": activation_site,
        "layers": args.layers,
        "split_family": args.split_family,
        "splits_path": str(args.splits) if args.splits else None,
        "eval_splits": eval_splits,
        "drop_parse_failed": args.drop_parse_failed,
        "input_rows": dataset["input_rows"],
        "kept_rows": dataset["kept_rows"],
        "candidate_modes": {
            "gold": "ground_truth / ontology_raw.hypotheses",
            "foil": "first non-empty line of model_output",
        },
        "score_names": list(SCORE_NAMES),
        "lens": {
            "apply_final_norm": not args.no_final_norm,
            "dtype": str(dtype),
            "activation_dtype": str(activation_dtype),
            "device_map": args.device_map,
            "transformers_version": transformers.__version__,
            "loaded_model_class": type(model).__name__,
            "text_lm_class": type(text_lm).__name__,
            "norm_device": str(module_device(final_norm_for_lens(text_lm) or text_lm.lm_head)),
            "lm_head_device": str(module_device(text_lm.lm_head)),
        },
        "results": results,
        "peak_by_score": peak_by_score({layer: data["scores"] for layer, data in results.items()}, eval_split=peak_split),
        "elapsed_seconds": time.monotonic() - started,
        "notes": [
            "Training-free accessibility proxy; it is not a causal intervention.",
            "Multi-token candidate scores reuse the same pre-generation lens distribution for each candidate token.",
            "The foil is the first generated line, so correct rows can contain paraphrases rather than strict wrong foils.",
        ],
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
