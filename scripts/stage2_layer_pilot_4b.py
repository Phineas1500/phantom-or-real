#!/usr/bin/env python3
"""4B layer-selection pilot: sample 500 rows, extract 34 layers, rank by probe AUC."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import read_stage1_rows, run_extraction  # noqa: E402
from src.env_loader import load_env  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_400_100_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        train_size=400,
        test_size=100,
        random_state=seed,
        stratify=y,
    )
    return train_idx, test_idx


def fit_auc(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    clf = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"),
    )
    clf.fit(x[train_idx], y[train_idx])
    scores = clf.predict_proba(x[test_idx])[:, 1]
    return float(roc_auc_score(y[test_idx], scores))


def top_layer_in_pool(auc_by_layer: dict[int, float], pool: list[int], used: set[int]) -> int:
    candidates = [layer for layer in pool if layer in auc_by_layer and layer not in used]
    if not candidates:
        raise ValueError(f"No available candidates in pool after exclusions: pool={pool}, used={sorted(used)}")
    return max(candidates, key=lambda layer: auc_by_layer[layer])


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model-key", default="gemma3_4b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage2/pilots/layer_selection_4b"))
    parser.add_argument("--output", type=Path, default=Path("docs/layer_selection.json"))
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--per-height", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--load-mode", choices=["no-processing", "default"], default="no-processing")
    args = parser.parse_args()

    # 1) Read and sample 125 per height from parse_failed=False rows
    rows = read_stage1_rows(args.jsonl)
    by_height: dict[int, list[dict]] = defaultdict(list)
    for _row_idx, row in rows:
        if row.get("task") != args.task:
            continue
        if row.get("parse_failed"):
            continue
        h = row.get("height")
        if h in {1, 2, 3, 4}:
            by_height[int(h)].append(row)

    sampled_rows: list[dict] = []
    sample_counts: dict[str, int] = {}
    for h in [1, 2, 3, 4]:
        pool = by_height[h]
        if len(pool) < args.per_height:
            raise ValueError(f"height={h} has only {len(pool)} usable rows, need {args.per_height}")
        rng = random.Random(args.seed + h * 1009)
        picks = rng.sample(pool, args.per_height)
        sampled_rows.extend(picks)
        sample_counts[f"h{h}"] = len(picks)

    # deterministic ordering
    sampled_rows.sort(key=lambda r: r["example_id"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pilot_jsonl = args.out_dir / f"{args.model_key}_{args.task}_pilot500.jsonl"
    with pilot_jsonl.open("w") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # 2) Extract all 34 layers in one pass
    layers = list(range(34))
    run_extraction(
        jsonl_path=pilot_jsonl,
        model_name=args.model,
        model_key=args.model_key,
        task=args.task,
        layers=layers,
        batch_size=args.batch_size,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        out_dir=args.out_dir,
        load_mode=args.load_mode,
    )

    # 3) Fixed 400/100 split for per-layer AUC
    sidecar_path = args.out_dir / f"{args.model_key}_{args.task}_L0.example_ids.jsonl"
    sidecar_rows = read_jsonl(sidecar_path)
    y = np.array([int(row["is_correct_strong"]) for row in sidecar_rows], dtype=np.int64)
    if len(y) != 500:
        raise ValueError(f"Expected 500 sampled rows, got {len(y)}")
    train_idx, test_idx = stratified_400_100_split(y, seed=args.seed)

    auc_by_layer: dict[int, float] = {}
    for layer in layers:
        act_path = args.out_dir / f"{args.model_key}_{args.task}_L{layer}.safetensors"
        x = load_file(act_path)["activations"].float().cpu().numpy()
        auc_by_layer[layer] = fit_auc(x, y, train_idx, test_idx)

    # 4) Layer-pick logic from plan
    best_overall = max(layers, key=lambda l: auc_by_layer[l])
    first_third = list(range(0, 12))
    last_third = list(range(22, 34))

    selected: list[int] = [best_overall]
    used = {best_overall}

    best_first = top_layer_in_pool(auc_by_layer, first_third, used=set())
    if best_first not in used:
        selected.append(best_first)
        used.add(best_first)
    else:
        selected.append(top_layer_in_pool(auc_by_layer, first_third, used=used))
        used.add(selected[-1])

    if best_overall in last_third:
        # replacement rule: use second-highest in 0..22 instead of "best last-third"
        replacement_pool = list(range(0, 23))
        third = top_layer_in_pool(auc_by_layer, replacement_pool, used=used)
    else:
        third = top_layer_in_pool(auc_by_layer, last_third, used=used)
    selected.append(third)

    # ensure uniqueness and stable int list
    selected = list(dict.fromkeys(selected))
    if len(selected) != 3:
        raise ValueError(f"Expected 3 unique selected layers, got {selected}")

    # 5) Emit report
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = None

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "pilot_jsonl": str(pilot_jsonl),
        "sample_counts": sample_counts,
        "layers_evaluated": layers,
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "auc_by_layer": {f"L{layer}": auc_by_layer[layer] for layer in layers},
        "selected_layers": selected,
        "selection_logic": {
            "best_overall": best_overall,
            "best_first_third_0_11": best_first,
            "best_overall_in_last_third": best_overall in last_third,
            "third_pick_rule": "best_last_third_22_33 unless best_overall in last_third, then best_in_0_22_excluding_selected",
        },
        "git_commit_sha": git_sha,
    }

    write_json(args.output, report)
    print(args.output)
    print("selected_layers:", selected)


if __name__ == "__main__":
    main()