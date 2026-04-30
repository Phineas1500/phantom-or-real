#!/usr/bin/env python3
"""Stage 2 Phase B.3.a: prompt-length and within-height diagnostics for probes."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_phase0 import add_prompt_length_fallback, add_prompt_token_counts, load_stage1_records  # noqa: E402
from src.stage2_probes import (  # noqa: E402
    _safe_auc,
    bootstrap_auc_ci,
    load_probe_dataset,
    read_json,
    read_split_assignments,
    split_indices_from_assignments,
    write_json,
)


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _has_two_classes(labels: list[int], indices: list[int]) -> bool:
    return len({labels[idx] for idx in indices}) == 2


def _fit_logistic_with_c_sweep(
    *,
    x: np.ndarray,
    labels: list[int],
    train_indices: list[int],
    val_indices: list[int],
    c_values: tuple[float, ...],
    max_iter: int,
) -> tuple[Any, float, float | None]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not c_values:
        raise ValueError("c_values must be non-empty")

    best_model = None
    best_c = None
    best_val_auc = -math.inf
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=max_iter, solver="lbfgs"),
        )
        model.fit(x[train_indices], [labels[idx] for idx in train_indices])
        val_scores = [float(score) for score in model.predict_proba(x[val_indices])[:, 1]]
        val_auc = _safe_auc([labels[idx] for idx in val_indices], val_scores)
        rank_auc = val_auc if val_auc is not None else -math.inf
        if rank_auc > best_val_auc:
            best_val_auc = rank_auc
            best_model = model
            best_c = float(c_value)
    assert best_model is not None
    return best_model, best_c, (None if best_val_auc == -math.inf else float(best_val_auc))


def _safe_partial_corr(x: np.ndarray, y: np.ndarray, covars: np.ndarray) -> float | None:
    from sklearn.linear_model import LinearRegression

    if len(x) == 0 or len(y) == 0 or x.shape[0] != y.shape[0] or covars.shape[0] != x.shape[0]:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    if covars.ndim != 2 or covars.shape[1] == 0:
        return None

    lr = LinearRegression()
    lr.fit(covars, x)
    x_res = x - lr.predict(covars)
    lr.fit(covars, y)
    y_res = y - lr.predict(covars)
    x_std = float(np.std(x_res))
    y_std = float(np.std(y_res))
    if x_std < 1e-12 or y_std < 1e-12:
        return None
    return float(np.corrcoef(x_res, y_res)[0, 1])


def _model_scores(model: Any, x: np.ndarray, indices: list[int]) -> tuple[list[float], list[float]]:
    if not indices:
        return [], []
    x_sub = x[indices]
    probs = [float(score) for score in model.predict_proba(x_sub)[:, 1]]
    if hasattr(model, "decision_function"):
        logits = [float(score) for score in model.decision_function(x_sub)]
    else:
        eps = 1e-9
        logits = [float(math.log(max(p, eps) / max(1.0 - p, eps))) for p in probs]
    return probs, logits


def _headline_layers(probe_auc: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for task, by_layer in probe_auc["results"].items():
        best_layer = None
        best_val = -math.inf
        for layer, by_probe_type in by_layer.items():
            logistic = by_probe_type.get("logistic", {})
            s1 = logistic.get("s1")
            if not s1 or s1.get("status") != "ok":
                continue
            val_auc = s1.get("val_auc")
            if val_auc is None:
                continue
            if float(val_auc) > best_val:
                best_val = float(val_auc)
                best_layer = layer
        if best_layer is None:
            raise ValueError(f"could not determine headline layer for {task}")
        out[task] = best_layer
    return out


def _records_by_row_index(
    *,
    jsonl_path: Path,
    model_name: str,
    task: str,
    length_mode: str,
    hf_cache: Path | None,
    allow_tokenizer_fallback: bool,
) -> dict[int, dict[str, Any]]:
    records = load_stage1_records([jsonl_path], models=[model_name], tasks=[task])
    if length_mode == "tokenizer":
        try:
            add_prompt_token_counts(records, hf_cache=hf_cache)
        except Exception as exc:  # noqa: BLE001
            if not allow_tokenizer_fallback:
                raise
            print(
                f"[WARN] tokenizer length mode failed for {jsonl_path.name}; "
                f"falling back to whitespace lengths. error={type(exc).__name__}: {exc}"
            )
            add_prompt_length_fallback(records)
    else:
        add_prompt_length_fallback(records)
    return {int(record["row_index"]): record for record in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-auc", type=Path, default=Path("results/stage2/probe_auc.json"))
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--activation-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--model-key", default="gemma3_4b")
    parser.add_argument("--model-name", default="gemma3-4b")
    parser.add_argument("--split-families", nargs="+", choices=("s1", "s2", "s3"), default=("s1", "s3"))
    parser.add_argument("--c-values", type=parse_float_list, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--length-mode", choices=("tokenizer", "whitespace"), default="whitespace")
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--no-tokenizer-fallback", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/stage2/probe_diagnostics_length_4b.json"))
    args = parser.parse_args()

    probe_auc = read_json(args.probe_auc)
    assignments = read_split_assignments(args.splits)
    headline_by_task = _headline_layers(probe_auc)

    output: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model_key,
        "model_name": args.model_name,
        "probe_auc_source": str(args.probe_auc),
        "splits_source": str(args.splits),
        "activation_dir": str(args.activation_dir),
        "split_families": list(args.split_families),
        "length_mode": args.length_mode,
        "headline_layers": headline_by_task,
        "length_regression": {},
        "within_height_auc": {},
    }

    for task, layer_key in headline_by_task.items():
        layer_num = int(layer_key.lstrip("L"))
        prefix = args.activation_dir / f"{args.model_key}_{task}_{layer_key}"
        meta = read_json(prefix.with_suffix(".meta.json"))
        source_file = meta["jsonl_path"]

        dataset = load_probe_dataset(
            activation_path=prefix.with_suffix(".safetensors"),
            sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
            drop_parse_failed=True,
        )
        x = dataset["x"]
        labels = dataset["labels"]
        sidecar = dataset["sidecar"]
        row_records = _records_by_row_index(
            jsonl_path=Path(source_file),
            model_name=args.model_name,
            task=task,
            length_mode=args.length_mode,
            hf_cache=args.hf_cache,
            allow_tokenizer_fallback=not args.no_tokenizer_fallback,
        )
        lengths = np.array(
            [
                float(row_records[int(row["row_index"])]["prompt_token_count"])
                for row in sidecar
            ],
            dtype=np.float64,
        )
        heights = np.array([float(row["height"]) for row in sidecar], dtype=np.float64)

        output["length_regression"].setdefault(task, {})
        output["within_height_auc"].setdefault(task, {})

        for split_family in args.split_families:
            split_field = f"{split_family}_split"
            split_indices = split_indices_from_assignments(
                sidecar,
                assignments=assignments,
                source_file=source_file,
                split_field=split_field,
            )
            train_indices = split_indices["train"]
            val_indices = split_indices["val"]
            test_indices = split_indices["test"]

            if not _has_two_classes(labels, train_indices) or not _has_two_classes(labels, val_indices) or not _has_two_classes(labels, test_indices):
                output["length_regression"][task][split_family] = {
                    "status": "skipped_no_evaluable_holdout",
                    "layer": layer_key,
                    "n_train": len(train_indices),
                    "n_val": len(val_indices),
                    "n_test": len(test_indices),
                    "partial_corr_probe_logit_vs_correct_given_length": None,
                    "partial_corr_probe_logit_vs_correct_given_length_height": None,
                }
                output["within_height_auc"][task][split_family] = {
                    "status": "skipped_no_evaluable_holdout",
                    "layer": layer_key,
                    "per_height": {},
                }
                continue

            model, best_c, val_auc = _fit_logistic_with_c_sweep(
                x=x,
                labels=labels,
                train_indices=train_indices,
                val_indices=val_indices,
                c_values=args.c_values,
                max_iter=args.max_iter,
            )
            test_probs, test_logits = _model_scores(model, x, test_indices)
            y_test = np.array([labels[idx] for idx in test_indices], dtype=np.float64)
            length_test = lengths[test_indices]
            height_test = heights[test_indices]
            logits_test = np.array(test_logits, dtype=np.float64)

            pc_len = _safe_partial_corr(
                logits_test,
                y_test,
                np.column_stack([length_test]),
            )
            pc_len_height = _safe_partial_corr(
                logits_test,
                y_test,
                np.column_stack([length_test, height_test]),
            )
            output["length_regression"][task][split_family] = {
                "status": "ok",
                "layer": layer_key,
                "best_c": best_c,
                "val_auc": val_auc,
                "test_auc": _safe_auc(list(y_test.astype(int)), test_probs),
                "partial_corr_probe_logit_vs_correct_given_length": pc_len,
                "partial_corr_probe_logit_vs_correct_given_length_height": pc_len_height,
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "n_test": len(test_indices),
            }

            per_height: dict[str, Any] = {}
            for height in (2, 3, 4):
                height_train = [idx for idx in train_indices if int(sidecar[idx]["height"]) == height]
                height_val = [idx for idx in val_indices if int(sidecar[idx]["height"]) == height]
                height_test = [idx for idx in test_indices if int(sidecar[idx]["height"]) == height]
                key = f"h{height}"

                if (
                    len(height_train) == 0
                    or len(height_val) == 0
                    or len(height_test) == 0
                    or not _has_two_classes(labels, height_train)
                    or not _has_two_classes(labels, height_val)
                    or not _has_two_classes(labels, height_test)
                ):
                    per_height[key] = {
                        "status": "skipped_no_evaluable_holdout",
                        "n_train": len(height_train),
                        "n_val": len(height_val),
                        "n_test": len(height_test),
                        "test_auc": None,
                        "test_auc_ci": None,
                        "best_c": None,
                        "val_auc": None,
                    }
                    continue

                model_h, best_c_h, val_auc_h = _fit_logistic_with_c_sweep(
                    x=x,
                    labels=labels,
                    train_indices=height_train,
                    val_indices=height_val,
                    c_values=args.c_values,
                    max_iter=args.max_iter,
                )
                probs_h, _ = _model_scores(model_h, x, height_test)
                y_h = [labels[idx] for idx in height_test]
                per_height[key] = {
                    "status": "ok",
                    "n_train": len(height_train),
                    "n_val": len(height_val),
                    "n_test": len(height_test),
                    "best_c": best_c_h,
                    "val_auc": val_auc_h,
                    "test_auc": _safe_auc(y_h, probs_h),
                    "test_auc_ci": bootstrap_auc_ci(
                        y_h,
                        probs_h,
                        seed=args.seed + layer_num * 100 + height,
                        samples=args.bootstrap_samples,
                    ),
                }

            output["within_height_auc"][task][split_family] = {
                "status": "ok",
                "layer": layer_key,
                "per_height": per_height,
            }

    write_json(args.output, output)
    print(args.output)


if __name__ == "__main__":
    main()
