"""Stage 2 raw-activation probe helpers."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = tuple(SPLIT_FRACTIONS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def stratified_split_indices(labels: list[int], *, seed: int) -> dict[str, list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)

    splits = {split: [] for split in SPLITS}
    for label, indices in sorted(by_label.items()):
        rng = random.Random(seed + label * 1009)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n >= 3:
            n_val = max(1, round(n * SPLIT_FRACTIONS["val"]))
            n_test = max(1, round(n * SPLIT_FRACTIONS["test"]))
            n_train = max(1, n - n_val - n_test)
        else:
            n_train, n_val, n_test = n, 0, 0
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    for split in SPLITS:
        splits[split].sort()
    return splits


def _class_counts(labels: list[int], indices: list[int]) -> dict[str, int]:
    positive = sum(labels[idx] for idx in indices)
    return {
        "n": len(indices),
        "positive_n": int(positive),
        "negative_n": int(len(indices) - positive),
    }


def _has_two_classes(labels: list[int], indices: list[int]) -> bool:
    return len({labels[idx] for idx in indices}) == 2


def _predict_scores(model, x: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    return [float(score) for score in model.predict_proba(x[indices])[:, 1]]


def train_logistic_probe(
    x: np.ndarray,
    labels: list[int],
    sidecar_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    splits = stratified_split_indices(labels, seed=seed)
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    split_counts = {split: _class_counts(labels, indices) for split, indices in splits.items()}

    if not _has_two_classes(labels, train_indices):
        return {"status": "skipped_one_class_train", "split_counts": split_counts}
    if not _has_two_classes(labels, val_indices) or not _has_two_classes(labels, test_indices):
        return {"status": "skipped_no_evaluable_holdout", "split_counts": split_counts}

    best = None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000),
        )
        model.fit(x[train_indices], [labels[idx] for idx in train_indices])
        val_scores = _predict_scores(model, x, val_indices)
        val_auc = _safe_auc([labels[idx] for idx in val_indices], val_scores)
        rank_auc = val_auc if val_auc is not None else -math.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "c": c_value,
                "val_auc": val_auc,
                "rank_auc": rank_auc,
            }

    assert best is not None
    model = best["model"]
    test_scores = _predict_scores(model, x, test_indices)
    per_height = {}
    for height in sorted({sidecar_rows[idx].get("height") for idx in test_indices}):
        height_indices = [idx for idx in test_indices if sidecar_rows[idx].get("height") == height]
        height_scores = _predict_scores(model, x, height_indices)
        per_height[f"h{height}"] = {
            **_class_counts(labels, height_indices),
            "auc": _safe_auc([labels[idx] for idx in height_indices], height_scores),
        }

    return {
        "status": "ok",
        "best_c": best["c"],
        "val_auc": best["val_auc"],
        "test_auc": _safe_auc([labels[idx] for idx in test_indices], test_scores),
        "split_counts": split_counts,
        "per_height": per_height,
    }


def load_activation_matrix(path: Path) -> np.ndarray:
    from safetensors.torch import load_file

    tensor = load_file(path)["activations"]
    return tensor.to(dtype=tensor.dtype).float().cpu().numpy()


def run_raw_activation_probe(
    *,
    activation_path: Path,
    sidecar_path: Path,
    seed: int,
    drop_parse_failed: bool = True,
) -> dict[str, Any]:
    x_all = load_activation_matrix(activation_path)
    sidecar_all = read_jsonl(sidecar_path)
    if x_all.shape[0] != len(sidecar_all):
        raise ValueError(f"{activation_path} rows {x_all.shape[0]} != sidecar rows {len(sidecar_all)}")

    keep_indices = [
        idx
        for idx, row in enumerate(sidecar_all)
        if not (drop_parse_failed and row.get("parse_failed"))
    ]
    x = x_all[keep_indices]
    sidecar = [sidecar_all[idx] for idx in keep_indices]
    labels = [int(row["is_correct_strong"]) for row in sidecar]
    result = train_logistic_probe(x, labels, sidecar, seed=seed)
    result.update(
        {
            "activation_path": str(activation_path),
            "sidecar_path": str(sidecar_path),
            "drop_parse_failed": drop_parse_failed,
            "input_rows": len(sidecar_all),
            "kept_rows": len(sidecar),
            "d_model": int(x.shape[1]) if x.ndim == 2 else None,
        }
    )
    return result


def run_probe_grid(
    *,
    activation_dir: Path,
    model_key: str,
    tasks: list[str],
    layers: list[int],
    seed: int,
    drop_parse_failed: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_dir": str(activation_dir),
        "model_key": model_key,
        "tasks": tasks,
        "layers": layers,
        "seed": seed,
        "drop_parse_failed": drop_parse_failed,
        "results": {},
        "best_by_task": {},
    }
    for task in tasks:
        report["results"][task] = {}
        for layer in layers:
            prefix = activation_dir / f"{model_key}_{task}_L{layer}"
            report["results"][task][f"L{layer}"] = run_raw_activation_probe(
                activation_path=prefix.with_suffix(".safetensors"),
                sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
                seed=seed + layer,
                drop_parse_failed=drop_parse_failed,
            )
        ok_layers = {
            layer_key: data
            for layer_key, data in report["results"][task].items()
            if data.get("status") == "ok" and data.get("val_auc") is not None
        }
        if ok_layers:
            best_key, best_data = max(ok_layers.items(), key=lambda item: item[1]["val_auc"])
            report["best_by_task"][task] = {
                "layer": best_key,
                "val_auc": best_data["val_auc"],
                "test_auc": best_data["test_auc"],
            }
        else:
            report["best_by_task"][task] = None
    return report
