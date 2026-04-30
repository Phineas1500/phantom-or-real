"""Stage 2 raw-activation probe helpers."""

from __future__ import annotations

import json
import math
import pickle
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .stage2_paths import DEFAULT_ACTIVATION_SITE, activation_stem


SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = tuple(SPLIT_FRACTIONS)
DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _safe_balanced_accuracy(labels: list[int], predictions: list[int]) -> float | None:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(labels, predictions))


def bootstrap_auc_ci(
    labels: list[int],
    scores: list[float],
    *,
    seed: int,
    samples: int,
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    if samples <= 0 or len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    label_array = np.array(labels)
    score_array = np.array(scores)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(samples):
        indices = rng.integers(0, len(label_array), len(label_array))
        sampled_labels = label_array[indices]
        if len(set(int(label) for label in sampled_labels)) < 2:
            continue
        aucs.append(float(roc_auc_score(sampled_labels, score_array[indices])))
    if not aucs:
        return None
    low, high = np.quantile(aucs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "alpha": alpha,
        "samples_requested": samples,
        "samples_used": len(aucs),
        "low": float(low),
        "mean": float(np.mean(aucs)),
        "high": float(high),
    }


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


def read_split_assignments(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["source_file"], int(row["row_index"])): row
        for row in read_jsonl(path)
    }


def split_indices_from_assignments(
    sidecar_rows: list[dict[str, Any]],
    *,
    assignments: dict[tuple[str, int], dict[str, Any]],
    source_file: str,
    split_field: str,
) -> dict[str, list[int]]:
    split_indices = {split: [] for split in SPLITS}
    missing = []
    for idx, row in enumerate(sidecar_rows):
        key = (source_file, int(row["row_index"]))
        assignment = assignments.get(key)
        if assignment is None:
            missing.append(key)
            continue
        split = assignment[split_field]
        if split not in split_indices:
            raise ValueError(f"unknown split {split!r} for {key}")
        split_indices[split].append(idx)
    if missing:
        raise KeyError(f"{len(missing)} sidecar rows missing split assignments; sample={missing[:5]}")
    return split_indices


def shuffled_labels(labels: list[int], *, seed: int) -> list[int]:
    rng = random.Random(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return shuffled


def _predict_scores(model, x: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    return [float(score) for score in model.predict_proba(x[indices])[:, 1]]


def _cosine_scores(x: np.ndarray, direction: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    eps = 1e-12
    x_subset = x[indices]
    x_norms = np.linalg.norm(x_subset, axis=1)
    direction_norm = np.linalg.norm(direction)
    denom = np.maximum(x_norms * max(direction_norm, eps), eps)
    return [float(score) for score in (x_subset @ direction) / denom]


def _best_threshold_balanced_accuracy(labels: list[int], scores: list[float]) -> tuple[float, float]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores length mismatch")
    if len(labels) == 0:
        raise ValueError("cannot tune threshold on empty set")
    if len(set(labels)) < 2:
        raise ValueError("threshold tuning requires both classes")

    sorted_unique = sorted(set(float(score) for score in scores))
    if len(sorted_unique) == 1:
        candidates = [sorted_unique[0]]
    else:
        candidates = [sorted_unique[0] - 1e-9]
        candidates.extend(
            (sorted_unique[idx] + sorted_unique[idx + 1]) / 2.0
            for idx in range(len(sorted_unique) - 1)
        )
        candidates.append(sorted_unique[-1] + 1e-9)

    best_threshold = candidates[0]
    best_ba = -math.inf
    for threshold in candidates:
        predictions = [1 if score >= threshold else 0 for score in scores]
        ba = _safe_balanced_accuracy(labels, predictions)
        if ba is None:
            continue
        if ba > best_ba:
            best_ba = ba
            best_threshold = threshold

    if best_ba == -math.inf:
        raise ValueError("unable to tune threshold")
    return float(best_threshold), float(best_ba)


def _is_sparse_matrix(x: Any) -> bool:
    try:
        from scipy import sparse
    except Exception:  # noqa: BLE001
        return False
    return bool(sparse.issparse(x))


def _make_logistic_pipeline(x: Any, *, c_value: float, max_iter: int, solver: str = "lbfgs"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(with_mean=not _is_sparse_matrix(x)),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=max_iter, solver=solver),
    )


def train_logistic_probe_with_splits(
    x: np.ndarray,
    labels: list[int],
    sidecar_rows: list[dict[str, Any]],
    *,
    splits: dict[str, list[int]],
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    split_counts = {split: _class_counts(labels, indices) for split, indices in splits.items()}

    if not _has_two_classes(labels, train_indices):
        return {"status": "skipped_one_class_train", "split_counts": split_counts}
    if not _has_two_classes(labels, val_indices) or not _has_two_classes(labels, test_indices):
        return {"status": "skipped_no_evaluable_holdout", "split_counts": split_counts}

    if not c_values:
        raise ValueError("at least one C value is required")
    best = None
    for c_value in c_values:
        model = _make_logistic_pipeline(x, c_value=c_value, max_iter=max_iter, solver=solver)
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
    test_labels = [labels[idx] for idx in test_indices]
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
        "probe_type": "logistic",
        "split_indices": splits,
        "best_c": best["c"],
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "val_auc": best["val_auc"],
        "test_auc": _safe_auc(test_labels, test_scores),
        "test_auc_ci": bootstrap_auc_ci(
            test_labels,
            test_scores,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "split_counts": split_counts,
        "per_height": per_height,
        "_artifact_model": model,
    }


def train_diffmeans_probe_with_splits(
    x: np.ndarray,
    labels: list[int],
    sidecar_rows: list[dict[str, Any]],
    *,
    splits: dict[str, list[int]],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    split_counts = {split: _class_counts(labels, indices) for split, indices in splits.items()}

    if not _has_two_classes(labels, train_indices):
        return {"status": "skipped_one_class_train", "split_counts": split_counts}
    if not _has_two_classes(labels, val_indices) or not _has_two_classes(labels, test_indices):
        return {"status": "skipped_no_evaluable_holdout", "split_counts": split_counts}

    train_correct = [idx for idx in train_indices if labels[idx] == 1]
    train_incorrect = [idx for idx in train_indices if labels[idx] == 0]
    if not train_correct or not train_incorrect:
        return {"status": "skipped_one_class_train", "split_counts": split_counts}

    mean_correct = np.mean(x[train_correct], axis=0)
    mean_incorrect = np.mean(x[train_incorrect], axis=0)
    direction_raw = mean_correct - mean_incorrect
    direction_raw_norm = float(np.linalg.norm(direction_raw))
    if direction_raw_norm <= 0.0:
        return {"status": "skipped_zero_direction", "split_counts": split_counts}
    direction = direction_raw / direction_raw_norm

    val_scores = _cosine_scores(x, direction, val_indices)
    val_labels = [labels[idx] for idx in val_indices]
    threshold, val_balanced_accuracy = _best_threshold_balanced_accuracy(val_labels, val_scores)

    test_scores = _cosine_scores(x, direction, test_indices)
    test_labels = [labels[idx] for idx in test_indices]
    test_predictions = [1 if score >= threshold else 0 for score in test_scores]
    per_height = {}
    for height in sorted({sidecar_rows[idx].get("height") for idx in test_indices}):
        height_indices = [idx for idx in test_indices if sidecar_rows[idx].get("height") == height]
        height_scores = _cosine_scores(x, direction, height_indices)
        per_height[f"h{height}"] = {
            **_class_counts(labels, height_indices),
            "auc": _safe_auc([labels[idx] for idx in height_indices], height_scores),
        }

    return {
        "status": "ok",
        "probe_type": "diffmeans",
        "split_indices": splits,
        "score_mode": "cosine",
        "direction_raw_norm": direction_raw_norm,
        "threshold": threshold,
        "val_balanced_accuracy": val_balanced_accuracy,
        "val_auc": _safe_auc(val_labels, val_scores),
        "test_auc": _safe_auc(test_labels, test_scores),
        "test_auc_ci": bootstrap_auc_ci(
            test_labels,
            test_scores,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "test_balanced_accuracy": _safe_balanced_accuracy(test_labels, test_predictions),
        "split_counts": split_counts,
        "per_height": per_height,
        "_artifact_direction": direction.astype(np.float32),
    }


def train_logistic_probe(
    x: np.ndarray,
    labels: list[int],
    sidecar_rows: list[dict[str, Any]],
    *,
    seed: int,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    bootstrap_samples: int = 0,
) -> dict[str, Any]:
    splits = stratified_split_indices(labels, seed=seed)
    return train_logistic_probe_with_splits(
        x,
        labels,
        sidecar_rows,
        splits=splits,
        c_values=c_values,
        max_iter=max_iter,
        solver=solver,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
    )


def load_activation_matrix(path: Path) -> np.ndarray:
    from safetensors.torch import load_file

    tensor = load_file(path)["activations"]
    return tensor.to(dtype=tensor.dtype).float().cpu().numpy()


def load_probe_dataset(
    *,
    activation_path: Path,
    sidecar_path: Path,
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
    return {
        "x": x,
        "sidecar": sidecar,
        "labels": labels,
        "input_rows": len(sidecar_all),
        "kept_rows": len(sidecar),
        "d_model": int(x.shape[1]) if x.ndim == 2 else None,
    }


def run_raw_activation_probe(
    *,
    activation_path: Path,
    sidecar_path: Path,
    seed: int,
    drop_parse_failed: bool = True,
    split_assignments: dict[tuple[str, int], dict[str, Any]] | None = None,
    source_file: str | None = None,
    split_family: str = "s1",
    shuffle_labels: bool = False,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    bootstrap_samples: int = 0,
    probe_type: str = "logistic",
) -> dict[str, Any]:
    dataset = load_probe_dataset(
        activation_path=activation_path,
        sidecar_path=sidecar_path,
        drop_parse_failed=drop_parse_failed,
    )
    x = dataset["x"]
    sidecar = dataset["sidecar"]
    labels = dataset["labels"]
    if shuffle_labels:
        labels = shuffled_labels(labels, seed=seed)
    train_fn_kwargs = {
        "x": x,
        "labels": labels,
        "sidecar_rows": sidecar,
        "bootstrap_samples": bootstrap_samples,
    }
    if split_assignments is None:
        if probe_type == "logistic":
            result = train_logistic_probe(
                x,
                labels,
                sidecar,
                seed=seed,
                c_values=c_values,
                max_iter=max_iter,
                solver=solver,
                bootstrap_samples=bootstrap_samples,
            )
        elif probe_type == "diffmeans":
            result = train_diffmeans_probe_with_splits(
                **train_fn_kwargs,
                splits=stratified_split_indices(labels, seed=seed),
                bootstrap_seed=seed,
            )
        else:
            raise ValueError(f"unknown probe_type: {probe_type}")
        split_mode = "stratified_random"
    else:
        if source_file is None:
            raise ValueError("source_file is required when split_assignments are provided")
        splits = split_indices_from_assignments(
            sidecar,
            assignments=split_assignments,
            source_file=source_file,
            split_field=f"{split_family}_split",
        )
        if probe_type == "logistic":
            result = train_logistic_probe_with_splits(
                x,
                labels,
                sidecar,
                splits=splits,
                c_values=c_values,
                max_iter=max_iter,
                solver=solver,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=seed,
            )
        elif probe_type == "diffmeans":
            result = train_diffmeans_probe_with_splits(
                **train_fn_kwargs,
                splits=splits,
                bootstrap_seed=seed,
            )
        else:
            raise ValueError(f"unknown probe_type: {probe_type}")
        split_mode = split_family
    result.update(
        {
            "probe_type": probe_type,
            "activation_path": str(activation_path),
            "sidecar_path": str(sidecar_path),
            "drop_parse_failed": drop_parse_failed,
            "shuffle_labels": shuffle_labels,
            "split_mode": split_mode,
            "source_file": source_file,
            "input_rows": dataset["input_rows"],
            "kept_rows": dataset["kept_rows"],
            "d_model": dataset["d_model"],
        }
    )
    return result


def _git_commit_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return proc.stdout.strip() or None


def save_probe_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def run_probe_grid(
    *,
    activation_dir: Path,
    model_key: str,
    tasks: list[str],
    layers: list[int],
    seed: int,
    drop_parse_failed: bool = True,
    splits_path: Path | None = None,
    split_family: str = "s1",
    shuffle_labels: bool = False,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    bootstrap_samples: int = 0,
    activation_site: str = DEFAULT_ACTIVATION_SITE,
    probe_type: str = "logistic",
    save_probes_dir: Path | None = None,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path) if splits_path is not None else None
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_dir": str(activation_dir),
        "activation_site": activation_site,
        "model_key": model_key,
        "tasks": tasks,
        "layers": layers,
        "seed": seed,
        "drop_parse_failed": drop_parse_failed,
        "splits_path": str(splits_path) if splits_path is not None else None,
        "split_family": split_family,
        "shuffle_labels": shuffle_labels,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "probe_type": probe_type,
        "save_probes_dir": str(save_probes_dir) if save_probes_dir is not None else None,
        "results": {},
        "best_by_task": {},
    }
    commit_sha = _git_commit_sha()
    for task in tasks:
        report["results"][task] = {}
        for layer in layers:
            prefix = activation_dir / activation_stem(
                model_key=model_key,
                task=task,
                layer=layer,
                activation_site=activation_site,
            )
            meta = read_json(prefix.with_suffix(".meta.json"))
            report["results"][task][f"L{layer}"] = run_raw_activation_probe(
                activation_path=prefix.with_suffix(".safetensors"),
                sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
                seed=seed + layer,
                drop_parse_failed=drop_parse_failed,
                split_assignments=split_assignments,
                source_file=meta.get("jsonl_path"),
                split_family=split_family,
                shuffle_labels=shuffle_labels,
                c_values=c_values,
                max_iter=max_iter,
                solver=solver,
                bootstrap_samples=bootstrap_samples,
                probe_type=probe_type,
            )
            layer_result = report["results"][task][f"L{layer}"]
            artifact_model = layer_result.pop("_artifact_model", None)
            artifact_direction = layer_result.pop("_artifact_direction", None)
            if save_probes_dir is not None and layer_result.get("status") == "ok":
                split_indices = layer_result.get("split_indices", {})
                filtered_sidecar = [
                    row
                    for row in read_jsonl(prefix.with_suffix(".example_ids.jsonl"))
                    if (not drop_parse_failed or not row.get("parse_failed"))
                ]
                sidecar_row_indices = {
                    split: [int(filtered_sidecar[idx]["row_index"]) for idx in indices]
                    for split, indices in split_indices.items()
                }
                sidecar_example_ids = {
                    split: [filtered_sidecar[idx].get("example_id") for idx in indices]
                    for split, indices in split_indices.items()
                }
                probe_payload: dict[str, Any] = {
                    "schema_version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "commit_sha": commit_sha,
                    "model_key": model_key,
                    "task": task,
                    "layer": layer,
                    "probe_type": probe_type,
                    "split_family": split_family,
                    "seed": seed + layer,
                    "split_mode": layer_result.get("split_mode"),
                    "activation_path": layer_result.get("activation_path"),
                    "sidecar_path": layer_result.get("sidecar_path"),
                    "source_file": layer_result.get("source_file"),
                    "drop_parse_failed": layer_result.get("drop_parse_failed"),
                    "shuffle_labels": layer_result.get("shuffle_labels"),
                    "split_counts": layer_result.get("split_counts"),
                    "split_indices": split_indices,
                    "threshold": layer_result.get("threshold"),
                    "sidecar_row_indices": sidecar_row_indices,
                    "sidecar_example_ids": sidecar_example_ids,
                }
                if probe_type == "logistic":
                    probe_payload["best_c"] = layer_result.get("best_c")
                    probe_payload["model"] = artifact_model
                if probe_type == "diffmeans":
                    probe_payload["direction"] = artifact_direction
                    probe_payload["score_mode"] = layer_result.get("score_mode")
                    probe_payload["direction_raw_norm"] = layer_result.get("direction_raw_norm")

                artifact_path = save_probes_dir / (
                    f"{model_key}_{task}_L{layer}_{split_family}.{probe_type}.pkl"
                )
                save_probe_artifact(artifact_path, probe_payload)
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


def _split_counts_for_transfer(
    labels: list[int],
    splits: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    return {split: _class_counts(labels, indices) for split, indices in splits.items()}


def _per_height_auc(
    *,
    model: Any,
    x: np.ndarray,
    labels: list[int],
    sidecar_rows: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    per_height = {}
    for height in sorted({sidecar_rows[idx].get("height") for idx in indices}):
        height_indices = [idx for idx in indices if sidecar_rows[idx].get("height") == height]
        scores = _predict_scores(model, x, height_indices)
        per_height[f"h{height}"] = {
            **_class_counts(labels, height_indices),
            "auc": _safe_auc([labels[idx] for idx in height_indices], scores),
        }
    return per_height


def train_cross_task_transfer(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    source_splits: dict[str, list[int]],
    target_splits: dict[str, list[int]],
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    source_labels = source["labels"]
    target_labels = target["labels"]
    train_indices = source_splits["train"]
    val_indices = source_splits["val"]
    source_test_indices = source_splits["test"]
    target_test_indices = target_splits["test"]

    source_split_counts = _split_counts_for_transfer(source_labels, source_splits)
    target_split_counts = _split_counts_for_transfer(target_labels, target_splits)
    if not _has_two_classes(source_labels, train_indices):
        return {
            "status": "skipped_one_class_source_train",
            "source_split_counts": source_split_counts,
            "target_split_counts": target_split_counts,
        }
    if not _has_two_classes(source_labels, val_indices) or not _has_two_classes(source_labels, source_test_indices):
        return {
            "status": "skipped_no_evaluable_source_holdout",
            "source_split_counts": source_split_counts,
            "target_split_counts": target_split_counts,
        }
    if not _has_two_classes(target_labels, target_test_indices):
        return {
            "status": "skipped_no_evaluable_target_test",
            "source_split_counts": source_split_counts,
            "target_split_counts": target_split_counts,
        }
    if not c_values:
        raise ValueError("at least one C value is required")

    best = None
    for c_value in c_values:
        model = _make_logistic_pipeline(source["x"], c_value=c_value, max_iter=max_iter, solver=solver)
        model.fit(source["x"][train_indices], [source_labels[idx] for idx in train_indices])
        val_scores = _predict_scores(model, source["x"], val_indices)
        val_auc = _safe_auc([source_labels[idx] for idx in val_indices], val_scores)
        rank_auc = val_auc if val_auc is not None else -math.inf
        if best is None or rank_auc > best["rank_auc"]:
            best = {
                "model": model,
                "c": c_value,
                "source_val_auc": val_auc,
                "rank_auc": rank_auc,
            }

    assert best is not None
    model = best["model"]
    source_test_scores = _predict_scores(model, source["x"], source_test_indices)
    source_test_labels = [source_labels[idx] for idx in source_test_indices]
    target_test_scores = _predict_scores(model, target["x"], target_test_indices)
    target_test_labels = [target_labels[idx] for idx in target_test_indices]
    return {
        "status": "ok",
        "best_c": best["c"],
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "source_val_auc": best["source_val_auc"],
        "source_test_auc": _safe_auc(source_test_labels, source_test_scores),
        "source_test_auc_ci": bootstrap_auc_ci(
            source_test_labels,
            source_test_scores,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "target_test_auc": _safe_auc(target_test_labels, target_test_scores),
        "target_test_auc_ci": bootstrap_auc_ci(
            target_test_labels,
            target_test_scores,
            seed=bootstrap_seed + 100_003,
            samples=bootstrap_samples,
        ),
        "source_split_counts": source_split_counts,
        "target_split_counts": target_split_counts,
        "target_per_height": _per_height_auc(
            model=model,
            x=target["x"],
            labels=target_labels,
            sidecar_rows=target["sidecar"],
            indices=target_test_indices,
        ),
    }


def _load_dataset_for_grid(
    *,
    activation_dir: Path,
    model_key: str,
    task: str,
    layer: int,
    drop_parse_failed: bool,
    activation_site: str = DEFAULT_ACTIVATION_SITE,
) -> dict[str, Any]:
    prefix = activation_dir / activation_stem(
        model_key=model_key,
        task=task,
        layer=layer,
        activation_site=activation_site,
    )
    meta = read_json(prefix.with_suffix(".meta.json"))
    dataset = load_probe_dataset(
        activation_path=prefix.with_suffix(".safetensors"),
        sidecar_path=prefix.with_suffix(".example_ids.jsonl"),
        drop_parse_failed=drop_parse_failed,
    )
    dataset.update(
        {
            "task": task,
            "layer": layer,
            "activation_path": str(prefix.with_suffix(".safetensors")),
            "sidecar_path": str(prefix.with_suffix(".example_ids.jsonl")),
            "source_file": meta.get("jsonl_path"),
        }
    )
    return dataset


def run_cross_task_transfer_grid(
    *,
    activation_dir: Path,
    model_key: str,
    tasks: list[str],
    layers: list[int],
    splits_path: Path,
    seed: int,
    split_family: str = "s1",
    drop_parse_failed: bool = True,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    bootstrap_samples: int = 0,
    activation_site: str = DEFAULT_ACTIVATION_SITE,
) -> dict[str, Any]:
    split_assignments = read_split_assignments(splits_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_dir": str(activation_dir),
        "activation_site": activation_site,
        "model_key": model_key,
        "tasks": tasks,
        "layers": layers,
        "splits_path": str(splits_path),
        "split_family": split_family,
        "seed": seed,
        "drop_parse_failed": drop_parse_failed,
        "c_values": list(c_values),
        "max_iter": max_iter,
        "solver": solver,
        "bootstrap_samples": bootstrap_samples,
        "results": {},
        "best_by_transfer": {},
    }

    for layer in layers:
        layer_datasets = {
            task: _load_dataset_for_grid(
                activation_dir=activation_dir,
                model_key=model_key,
                task=task,
                layer=layer,
                drop_parse_failed=drop_parse_failed,
                activation_site=activation_site,
            )
            for task in tasks
        }
        layer_splits = {
            task: split_indices_from_assignments(
                dataset["sidecar"],
                assignments=split_assignments,
                source_file=dataset["source_file"],
                split_field=f"{split_family}_split",
            )
            for task, dataset in layer_datasets.items()
        }
        for source_task in tasks:
            for target_task in tasks:
                if source_task == target_task:
                    continue
                transfer_key = f"{source_task}_to_{target_task}"
                report["results"].setdefault(transfer_key, {})
                report["results"][transfer_key][f"L{layer}"] = {
                    **train_cross_task_transfer(
                        source=layer_datasets[source_task],
                        target=layer_datasets[target_task],
                        source_splits=layer_splits[source_task],
                        target_splits=layer_splits[target_task],
                        c_values=c_values,
                        max_iter=max_iter,
                        solver=solver,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=seed + layer * 10_009,
                    ),
                    "source_task": source_task,
                    "target_task": target_task,
                    "layer": layer,
                    "source_file": layer_datasets[source_task]["source_file"],
                    "target_file": layer_datasets[target_task]["source_file"],
                }

    for transfer_key, by_layer in report["results"].items():
        ok_layers = {
            layer_key: data
            for layer_key, data in by_layer.items()
            if data.get("status") == "ok" and data.get("source_val_auc") is not None
        }
        if ok_layers:
            best_key, best_data = max(ok_layers.items(), key=lambda item: item[1]["source_val_auc"])
            report["best_by_transfer"][transfer_key] = {
                "layer": best_key,
                "source_val_auc": best_data["source_val_auc"],
                "source_test_auc": best_data["source_test_auc"],
                "target_test_auc": best_data["target_test_auc"],
            }
        else:
            report["best_by_transfer"][transfer_key] = None
    return report
