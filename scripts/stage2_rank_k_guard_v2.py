#!/usr/bin/env python3
"""Rank-k guard v2: fresh-row expansion of the L30 compact-core claim.

The 457012 guard passed leave-one-row-out on the 13 rows that defined the
compact core. This job re-tests sufficiency on rows that contributed to
neither the PCA bases nor the original row selection: it draws fresh
baseline-wrong property rows, captures hinted/unhinted L30 states in-job,
and runs the rank-4/rank-8 LOO adds against an in-job concept-replacement
denominator. Pre-registered in docs/causal_handle_directions.md
("Pre-Registered Manuscript-Hardening Jobs", item A).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_subtype_discriminator import (  # noqa: E402
    Arm,
    fit_pca_basis,
    json_default,
    make_position_add_hook,
    make_replace_hook,
    parse_int_list,
    rank_k_reconstruction,
    row_bootstrap_ci,
    summarize_generation_rows,
)

COMPOSITE_MANIFEST_ROWS = [3073, 3290, 3415, 4322, 4675, 6188, 6327, 8035, 8298, 8874, 9549, 10079, 10714]


def select_fresh_rows(
    jsonl: Path,
    *,
    exclude: set[int],
    heights: list[int],
    per_height: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded balanced draw of parse-ok, strong-incorrect rows outside `exclude`."""
    pool: dict[int, list[dict[str, Any]]] = {height: [] for height in heights}
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            if row_index in exclude or not line.strip():
                continue
            row = json.loads(line)
            height = row.get("height")
            if height not in pool or row.get("parse_failed") or row.get("is_correct_strong"):
                continue
            row["row_index"] = row_index
            pool[height].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for height in heights:
        candidates = pool[height]
        if len(candidates) < per_height:
            raise ValueError(f"height {height}: only {len(candidates)} eligible rows, need {per_height}")
        selected.extend(rng.sample(candidates, per_height))
    return sorted(selected, key=lambda row: row["row_index"])


def shard_rows(rows: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    """Interleaved slice so each shard keeps the height balance."""
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} out of range for shard_count {shard_count}")
    by_height: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_height.setdefault(row.get("height"), []).append(row)
    picked: list[dict[str, Any]] = []
    for height in sorted(by_height):
        picked.extend(by_height[height][shard_index::shard_count])
    return sorted(picked, key=lambda row: row["row_index"])


def build_arms(ranks: list[int], layer: int) -> list[Arm]:
    arms = [
        Arm("unhinted_baseline", "none", "none"),
        Arm("hinted_baseline", "hinted_prompt", "unhinted_baseline"),
        Arm(f"L{layer}_concept_replace", "concept_replace", "unhinted_baseline", (layer,)),
        Arm(f"L{layer}_random_replace", "random_replace", "unhinted_baseline", (layer,)),
    ]
    for rank in ranks:
        arms.append(Arm(f"rank{rank}_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), rank, "leave_one_row_out"))
    return arms


def build_specificity_arms(layer: int, rank: int, draws: int) -> list[Arm]:
    """Random-basis specificity ladder for claim 8 (pre-registered item C)."""
    arms = [
        Arm("unhinted_baseline", "none", "none"),
        Arm("hinted_baseline", "hinted_prompt", "unhinted_baseline"),
        Arm(f"rank{rank}_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), rank, "leave_one_row_out"),
        Arm(f"mean_only_add_L{layer}", "mean_add", "unhinted_baseline", (layer,), rank, "leave_one_row_out"),
    ]
    for draw in range(1, draws + 1):
        arms.append(Arm(f"rand_subspace_add_L{layer}_d{draw}", "rand_subspace_add", "unhinted_baseline", (layer,), rank, "random_orthonormal"))
    for draw in range(1, draws + 1):
        arms.append(Arm(f"rand_norm_add_L{layer}_d{draw}", "rand_norm_add", "unhinted_baseline", (layer,), rank, "norm_matched_gaussian"))
    return arms


def build_predcoeff_arms(layer: int, rank: int) -> list[Arm]:
    """Predicted-coefficient repair arms for the hint-free ladder (pre-registered item E)."""
    return [
        Arm("unhinted_baseline", "none", "none"),
        Arm("hinted_baseline", "hinted_prompt", "unhinted_baseline"),
        Arm(f"rank{rank}_dev_add_L{layer}", "rank_dev_add", "unhinted_baseline", (layer,), rank, "dev_full"),
        Arm(f"mean_only_dev_add_L{layer}", "mean_dev_add", "unhinted_baseline", (layer,), rank, "dev_full"),
        Arm(f"rank{rank}_pred_add_L{layer}", "pred_add", "unhinted_baseline", (layer,), rank, "ridge_predicted"),
        Arm(f"rank{rank}_shufpred_add_L{layer}", "shufpred_add", "unhinted_baseline", (layer,), rank, "ridge_shuffled"),
    ]


def build_classmean_arms(layer: int, rank: int) -> list[Arm]:
    """Natural class-mean repair arms (pre-registered item F(ii))."""
    return [
        Arm("unhinted_baseline", "none", "none"),
        Arm("hinted_baseline", "hinted_prompt", "unhinted_baseline"),
        Arm(f"rank{rank}_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), rank, "leave_one_row_out"),
        Arm(f"class_mean_raw_add_L{layer}", "class_mean_raw_add", "unhinted_baseline", (layer,), rank, "natural_class_mean"),
        Arm(f"class_mean_proj_add_L{layer}", "class_mean_proj_add", "unhinted_baseline", (layer,), rank, "natural_class_mean_rank_projected"),
        Arm(f"rand_norm_add_L{layer}_d1", "rand_norm_add", "unhinted_baseline", (layer,), rank, "norm_matched_gaussian"),
    ]


def load_class_mean_vector(npz_path: Path, manifest_path: Path, layer: int) -> np.ndarray:
    """Natural correct-minus-incorrect class mean of per-row position-mean states."""
    labels = {}
    with manifest_path.open() as f:
        for line in f:
            row = json.loads(line)
            labels[int(row["source_row_index"])] = bool(row["is_correct_strong"])
    data = np.load(npz_path)
    per_class: dict[bool, list[np.ndarray]] = {True: [], False: []}
    for row_index, label in labels.items():
        key = f"L{layer}_row{row_index}_unhinted_concept_states"
        if key in data:
            per_class[label].append(data[key].astype(np.float64).mean(axis=0))
    if not per_class[True] or not per_class[False]:
        raise ValueError(f"capture npz {npz_path} lacks one of the classes for layer {layer}")
    return np.stack(per_class[True]).mean(axis=0) - np.stack(per_class[False]).mean(axis=0)


def class_mean_add_matrix(
    arm: Arm,
    class_vector: np.ndarray,
    basis: dict[str, Any],
    recon_pca: np.ndarray,
) -> np.ndarray:
    """Class vector (raw or basis-projected) tiled per position, norm-matched to the LOO recon."""
    if arm.kind == "class_mean_proj_add":
        components = basis["components"]
        direction = (class_vector @ components.T) @ components
    else:
        direction = class_vector
    tiled = np.tile(direction.astype(np.float64), (recon_pca.shape[0], 1))
    target = np.linalg.norm(recon_pca.astype(np.float64), axis=1, keepdims=True)
    current = np.maximum(np.linalg.norm(tiled, axis=1, keepdims=True), 1e-8)
    return (tiled * (target / current)).astype(np.float32)


def build_classmean_b_arms(layer: int, rank: int, draws: int) -> list[Arm]:
    """Shuffled-label projected control + riders (pre-registered item F(ii)-b)."""
    arms = [Arm("unhinted_baseline", "none", "none")]
    for draw in range(1, draws + 1):
        arms.append(Arm(f"shuflabel_proj_add_L{layer}_d{draw}", "shuflabel_proj_add", "unhinted_baseline", (layer,), rank, "shuffled_label_class_mean_projected"))
    arms.append(Arm(f"signflip_proj_add_L{layer}", "signflip_proj_add", "unhinted_baseline", (layer,), rank, "negated_class_mean_projected"))
    arms.append(Arm(f"fixednorm_proj_add_L{layer}", "fixednorm_proj_add", "unhinted_baseline", (layer,), rank, "class_mean_projected_fixed_pooled_norm"))
    return arms


def build_classmean_c_arms(layer: int, rank: int) -> list[Arm]:
    """Deployment riders: position-leak + collateral arms (pre-registered item F(ii)-c)."""
    return [
        Arm("unhinted_baseline", "none", "none"),
        Arm(f"fixednorm_proj_add_L{layer}", "fixednorm_proj_add", "unhinted_baseline", (layer,), rank, "class_mean_projected_fixed_pooled_norm"),
        Arm(f"fixednorm_allpos_add_L{layer}", "fixednorm_allpos_add", "unhinted_baseline", (layer,), rank, "class_mean_projected_all_concept_positions"),
        Arm("correct_unhinted_baseline", "none_correct", "none"),
        Arm(f"correct_fixednorm_add_L{layer}", "correct_fixednorm_add", "correct_unhinted_baseline", (layer,), rank, "class_mean_projected_full_basis_correct_rows"),
    ]


def select_correct_rows(
    jsonl: Path,
    *,
    exclude: set[int],
    heights: list[int],
    per_height: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded balanced draw of parse-ok, strong-CORRECT rows outside `exclude`."""
    pool: dict[int, list[dict[str, Any]]] = {height: [] for height in heights}
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            if row_index in exclude or not line.strip():
                continue
            row = json.loads(line)
            height = row.get("height")
            if height not in pool or row.get("parse_failed") or not row.get("is_correct_strong"):
                continue
            row["row_index"] = row_index
            pool[height].append(row)
    rng = random.Random(seed)
    selected = []
    for height in heights:
        candidates = pool[height]
        if len(candidates) < per_height:
            raise ValueError(f"height {height}: only {len(candidates)} correct rows, need {per_height}")
        selected.extend(rng.sample(candidates, per_height))
    return sorted(selected, key=lambda row: row["row_index"])


def all_concept_names(stage1_row: dict[str, Any]) -> list[str]:
    fol = stage1_row["ontology_fol_structured"]
    names: set[str] = set()
    for section in ("inheritance", "membership"):
        for fact in fol.get(section, []) or []:
            for key in ("subject", "object"):
                value = fact.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
    names.add(fol["hypothesis"]["subject"])
    return sorted(names)


def load_capture_row_means(npz_path: Path, manifest_path: Path, layer: int) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
    labels: dict[int, bool] = {}
    with manifest_path.open() as f:
        for line in f:
            row = json.loads(line)
            labels[int(row["source_row_index"])] = bool(row["is_correct_strong"])
    data = np.load(npz_path)
    means = {
        row: data[f"L{layer}_row{row}_unhinted_concept_states"].astype(np.float64).mean(axis=0)
        for row in labels
        if f"L{layer}_row{row}_unhinted_concept_states" in data.files
    }
    return means, {row: labels[row] for row in means}


def class_vector_from_labels(row_means: dict[int, np.ndarray], labels: dict[int, bool]) -> np.ndarray:
    pos = np.stack([row_means[r] for r in sorted(row_means) if labels[r]])
    neg = np.stack([row_means[r] for r in sorted(row_means) if not labels[r]])
    return pos.mean(axis=0) - neg.mean(axis=0)


def shuffled_label_vectors(
    row_means: dict[int, np.ndarray], labels: dict[int, bool], *, draws: int, seed: int
) -> list[np.ndarray]:
    rows = sorted(row_means)
    values = np.array([labels[r] for r in rows])
    out = []
    for draw in range(1, draws + 1):
        rng = np.random.default_rng(seed + draw)
        permuted = dict(zip(rows, values[rng.permutation(len(rows))]))
        out.append(class_vector_from_labels(row_means, permuted))
    return out


def load_dev_states(npz_path: Path, layer: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per-row unhinted concept states and concept deltas from a composite states npz."""
    import re

    data = np.load(npz_path)
    unhinted: dict[int, np.ndarray] = {}
    delta: dict[int, np.ndarray] = {}
    for key in data.files:
        match = re.match(rf"L{layer}_row(\d+)_(unhinted_concept_states|concept_delta)$", key)
        if not match:
            continue
        row = int(match.group(1))
        arr = data[key].astype(np.float64)
        if match.group(2) == "concept_delta":
            delta[row] = arr
        else:
            unhinted[row] = arr
    if not unhinted or set(unhinted) != set(delta):
        raise ValueError(f"dev npz {npz_path} lacks paired unhinted/delta arrays for layer {layer}")
    return unhinted, delta


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, ...]:
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
    xs = (x - mu) / sd
    n = xs.shape[0]
    gram = xs @ xs.T
    weights = xs.T @ np.linalg.solve(gram + alpha * np.eye(n), y - y.mean(axis=0))
    return mu, sd, weights, y.mean(axis=0)


def _ridge_predict(model: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    mu, sd, weights, bias = model
    return ((x - mu) / sd) @ weights + bias


def fit_coeff_predictor(
    dev_unhinted: dict[int, np.ndarray],
    dev_coeffs: dict[int, np.ndarray],
    *,
    alphas: tuple[float, ...],
    shuffle_seed: int | None = None,
) -> dict[str, Any]:
    """Ridge from unhinted concept states to dev-basis coefficients.

    Alpha is picked by leave-one-dev-row-out mean cosine. With shuffle_seed,
    training targets are reassigned by a seeded row derangement (positions
    resampled from the partner row), breaking the X->Y pairing while keeping
    output scale.
    """
    rows = sorted(dev_unhinted)
    targets = dict(dev_coeffs)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        partner = list(rows)
        while True:
            rng.shuffle(partner)
            if all(a != b for a, b in zip(rows, partner)):
                break
        targets = {}
        for row, mate in zip(rows, partner):
            src = dev_coeffs[mate]
            idx = rng.integers(0, src.shape[0], size=dev_unhinted[row].shape[0])
            targets[row] = src[idx]

    def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        return num / den

    best: tuple[float, float] | None = None
    for alpha in alphas:
        scores = []
        for held in rows:
            train = [r for r in rows if r != held]
            model = _ridge_fit(
                np.concatenate([dev_unhinted[r] for r in train], axis=0),
                np.concatenate([targets[r] for r in train], axis=0),
                alpha,
            )
            scores.append(float(cosines(_ridge_predict(model, dev_unhinted[held]), targets[held]).mean()))
        mean_score = float(np.mean(scores))
        if best is None or mean_score > best[0]:
            best = (mean_score, alpha)
    loo_cosine, alpha = best
    model = _ridge_fit(
        np.concatenate([dev_unhinted[r] for r in rows], axis=0),
        np.concatenate([targets[r] for r in rows], axis=0),
        alpha,
    )
    return {"model": model, "alpha": alpha, "loo_cosine": loo_cosine, "shuffle_seed": shuffle_seed}


def draw_index(label: str) -> int:
    return int(label.rsplit("_d", 1)[1])


def control_add_matrix(
    arm: Arm,
    concept_delta: np.ndarray,
    basis: dict[str, Any],
    recon_pca: np.ndarray,
    *,
    control_seed: int,
    shard_index: int,
    source_row_index: int,
    q_cache: dict[int, np.ndarray],
) -> np.ndarray:
    """Per-position add matrix for the control arms, norm-matched to the PCA reconstruction."""
    mean = np.asarray(basis["mean"], dtype=np.float32)
    if arm.kind == "mean_add":
        return np.tile(mean, (concept_delta.shape[0], 1))
    if arm.kind == "rand_subspace_add":
        draw = draw_index(arm.label)
        q = q_cache.get(draw)
        if q is None:
            assert arm.rank_k is not None
            rng = np.random.default_rng(control_seed + 7919 * draw + 104729 * shard_index)
            q, _ = np.linalg.qr(rng.standard_normal((concept_delta.shape[1], arm.rank_k)))
            q = q.astype(np.float32)
            q_cache[draw] = q
        centered = concept_delta - mean
        rand_comp = (centered @ q) @ q.T
        pca_comp = recon_pca - mean
        target = np.linalg.norm(pca_comp, axis=1, keepdims=True)
        current = np.maximum(np.linalg.norm(rand_comp, axis=1, keepdims=True), 1e-8)
        return mean + rand_comp * (target / current)
    if arm.kind == "rand_norm_add":
        draw = draw_index(arm.label)
        rng = np.random.default_rng(control_seed + 7919 * draw + 104729 * shard_index + source_row_index)
        noise = rng.standard_normal(concept_delta.shape).astype(np.float32)
        target = np.linalg.norm(recon_pca, axis=1, keepdims=True)
        current = np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-8)
        return noise * (target / current)
    raise ValueError(f"not a control arm kind: {arm.kind}")


def hint_validated_summary(
    rows_out: list[dict[str, Any]],
    arms: list[Arm],
    seed: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Secondary pre-registered slice: rows where the hinted prompt repairs."""
    hinted: dict[int, list[float]] = {}
    for row in rows_out:
        if row["condition"] == "hinted_baseline":
            hinted.setdefault(row["source_row_index"], []).append(float(row["is_correct_strong"]))
    validated = {row_id for row_id, vals in hinted.items() if float(np.mean(vals)) >= threshold}
    subset = [row for row in rows_out if row["source_row_index"] in validated]
    return {
        "threshold": threshold,
        "validated_rows": sorted(validated),
        "n_validated_rows": len(validated),
        "summary": summarize_generation_rows(subset, arms, seed) if validated else {},
    }


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    job = report.get("slurm_job_id") or "local"
    shard = report["shard"]
    titles = {
        "rank8_specificity_controls_fresh_rows": "Rank-8 Specificity Controls (fresh rows)",
        "rank8_predicted_coefficients_fresh_rows": "Rank-8 Predicted-Coefficient Repair (fresh rows)",
    }
    title = titles.get(report["method"], "Rank-k Guard v2 (fresh rows)")
    lines = [
        f"# {title} - Job {job} - shard {shard['index']} of {shard['count']}",
        "",
        f"Output JSON: `{report['output']}`",
        f"Rows: {report['prepared_rows']} prepared from {report['selected_rows']} fresh-selection rows.",
        "",
        "## Causal arms (row-paired bootstrap vs in-job unhinted baseline)",
        "",
        "| arm | P(strong) | dP vs reference (CI95) | reference |",
        "| --- | ---: | ---: | --- |",
    ]
    for cond, entry in sorted(report["summary"].items()):
        if "paired_delta_vs_reference" in entry:
            ci = entry["paired_ci95"]
            delta = f"{entry['paired_delta_vs_reference']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        else:
            delta = "-"
        lines.append(f"| {cond} | {entry['strong_accuracy']:.3f} | {delta} | {entry['reference']} |")
    hv = report["hint_validated"]
    lines.extend(
        [
            "",
            f"Hint-validated rows (hinted P(strong) >= {hv['threshold']}): {hv['n_validated_rows']}.",
            "",
            f"Reading rule: {report['pre_registered_decision_rule']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--rank-list", default="4,8")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height", type=int, default=16)
    parser.add_argument("--selection-seed", type=int, default=20260702)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--exclude-rows", default=",".join(str(r) for r in COMPOSITE_MANIFEST_ROWS))
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260702)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--specificity-controls", action="store_true")
    parser.add_argument("--control-draws", type=int, default=4)
    parser.add_argument("--control-seed", type=int, default=20260704)
    parser.add_argument("--predicted-coefficients", action="store_true")
    parser.add_argument(
        "--dev-states-npz",
        type=Path,
        default=Path("results/stage2/erasure/focus_state_composite_27b_property_states.npz"),
    )
    parser.add_argument("--ridge-alphas", default="1e2,1e3,1e4,1e5,1e6")
    parser.add_argument("--pred-shuffle-seed", type=int, default=20260704)
    parser.add_argument("--class-mean", action="store_true")
    parser.add_argument("--class-mean-b", action="store_true")
    parser.add_argument("--shuffle-draws", type=int, default=4)
    parser.add_argument("--shuffle-label-seed", type=int, default=20260705)
    parser.add_argument("--class-mean-c", action="store_true")
    parser.add_argument("--correct-per-height", type=int, default=8)
    parser.add_argument("--correct-seed", type=int, default=20260706)
    parser.add_argument(
        "--capture-npz",
        type=Path,
        default=Path("results/stage2/erasure/natural_state_capture_27b_property_L30.npz"),
    )
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        default=Path("results/stage2/erasure/natural_state_capture_27b_property_L30.manifest.jsonl"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--states-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()

    if sum([args.specificity_controls, args.predicted_coefficients, args.class_mean, args.class_mean_b, args.class_mean_c]) > 1:
        raise ValueError("mode flags are mutually exclusive")
    if args.specificity_controls:
        stem = f"rank8_specificity_27b_property_shard{args.shard_index}of{args.shard_count}"
        method = "rank8_specificity_controls_fresh_rows"
    elif args.predicted_coefficients:
        stem = f"rank8_predcoeff_27b_property_shard{args.shard_index}of{args.shard_count}"
        method = "rank8_predicted_coefficients_fresh_rows"
    elif args.class_mean:
        stem = f"classmean_repair_27b_property_shard{args.shard_index}of{args.shard_count}"
        method = "natural_class_mean_repair_fresh_rows"
    elif args.class_mean_b:
        stem = f"classmean_b_controls_27b_property_shard{args.shard_index}of{args.shard_count}"
        method = "shuffled_label_projected_controls_fresh_rows"
    elif args.class_mean_c:
        stem = f"classmean_c_deployment_27b_property_shard{args.shard_index}of{args.shard_count}"
        method = "classmean_deployment_riders"
    else:
        stem = f"rank_k_guard_v2_27b_property_shard{args.shard_index}of{args.shard_count}"
        method = "rank_k_guard_v2_fresh_rows"
    out_jsonl = args.out_jsonl or Path(f"results/stage2/erasure/{stem}.jsonl")
    states_output = args.states_output or Path(f"results/stage2/erasure/{stem}_states.npz")
    output = args.output or Path(f"docs/{stem}.json")
    summary_md = args.summary_md or Path(f"docs/{stem}_summary.md")

    ranks = parse_int_list(args.rank_list)
    heights = parse_int_list(args.heights)
    exclude = {int(part) for part in args.exclude_rows.split(",") if part.strip()}
    all_rows = select_fresh_rows(
        args.jsonl, exclude=exclude, heights=heights, per_height=args.per_height, seed=args.selection_seed
    )
    selected_rows = shard_rows(all_rows, args.shard_index, args.shard_count)
    if args.class_mean_c:
        capture_row_ids = set()
        with args.capture_manifest.open() as f:
            for line in f:
                capture_row_ids.add(int(json.loads(line)["source_row_index"]))
        correct_exclude = exclude | {int(r["row_index"]) for r in all_rows} | capture_row_ids
        correct_rows = select_correct_rows(
            args.jsonl, exclude=correct_exclude, heights=heights,
            per_height=args.correct_per_height, seed=args.correct_seed,
        )
        for row in selected_rows:
            row["row_class"] = "failing"
        for row in correct_rows:
            row["row_class"] = "correct"
        print(f"correct-side rows: {[r['row_index'] for r in correct_rows]}", flush=True)
        selected_rows = selected_rows + correct_rows
    if args.specificity_controls:
        if len(ranks) != 1:
            raise ValueError("--specificity-controls expects a single rank in --rank-list")
        arms = build_specificity_arms(args.layer, ranks[0], args.control_draws)
    elif args.predicted_coefficients:
        if len(ranks) != 1:
            raise ValueError("--predicted-coefficients expects a single rank in --rank-list")
        arms = build_predcoeff_arms(args.layer, ranks[0])
    elif args.class_mean:
        if len(ranks) != 1:
            raise ValueError("--class-mean expects a single rank in --rank-list")
        arms = build_classmean_arms(args.layer, ranks[0])
    else:
        arms = build_arms(ranks, args.layer)

    class_vector = None
    if args.class_mean:
        class_vector = load_class_mean_vector(args.capture_npz, args.capture_manifest, args.layer)
        print(
            f"class_mean vector: |v|={np.linalg.norm(class_vector):.1f} from {args.capture_npz}",
            flush=True,
        )

    shuf_vectors: list[np.ndarray] = []
    if args.class_mean_b:
        arms = build_classmean_b_arms(args.layer, ranks[0], args.shuffle_draws)
        row_means, capture_labels = load_capture_row_means(args.capture_npz, args.capture_manifest, args.layer)
        class_vector = class_vector_from_labels(row_means, capture_labels)
        shuf_vectors = shuffled_label_vectors(
            row_means, capture_labels, draws=args.shuffle_draws, seed=args.shuffle_label_seed
        )
        print(
            f"class_mean_b: |real|={np.linalg.norm(class_vector):.1f} "
            f"shuffled norms={[f'{np.linalg.norm(v):.1f}' for v in shuf_vectors]}",
            flush=True,
        )
    if args.class_mean_c:
        arms = build_classmean_c_arms(args.layer, ranks[0])
        row_means, capture_labels = load_capture_row_means(args.capture_npz, args.capture_manifest, args.layer)
        class_vector = class_vector_from_labels(row_means, capture_labels)
        print(f"class_mean_c: |real|={np.linalg.norm(class_vector):.1f}", flush=True)

    dev_basis = None
    predictors: dict[str, dict[str, Any]] = {}
    if args.predicted_coefficients:
        alphas = tuple(float(part) for part in args.ridge_alphas.split(","))
        dev_unhinted, dev_delta = load_dev_states(args.dev_states_npz, args.layer)
        dev_basis = fit_pca_basis(dev_delta, ranks[0])
        dev_coeffs = {
            row: (arr - dev_basis["mean"][None, :]) @ dev_basis["components"].T
            for row, arr in dev_delta.items()
        }
        predictors["pred"] = fit_coeff_predictor(dev_unhinted, dev_coeffs, alphas=alphas)
        predictors["shufpred"] = fit_coeff_predictor(
            dev_unhinted, dev_coeffs, alphas=alphas, shuffle_seed=args.pred_shuffle_seed
        )
        print(
            f"dev_basis rows={len(dev_delta)} evr={dev_basis['explained_variance_ratio']:.3f} "
            f"pred alpha={predictors['pred']['alpha']} loo_cos={predictors['pred']['loo_cosine']:+.3f} "
            f"shufpred alpha={predictors['shufpred']['alpha']} loo_cos={predictors['shufpred']['loo_cosine']:+.3f}",
            flush=True,
        )
    total = len(selected_rows) * len(arms) * args.samples_per_row
    print(
        f"selected_rows={len(selected_rows)} of {len(all_rows)} "
        f"(shard {args.shard_index}/{args.shard_count}) arms={[arm.label for arm in arms]} "
        f"total_generations={total}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": [int(row["row_index"]) for row in selected_rows],
                    "heights": {f"h{h}": sum(1 for row in selected_rows if row.get("height") == h) for h in heights},
                    "arms": [asdict(arm) for arm in arms],
                    "total_generations": total,
                },
                indent=2,
                default=json_default,
            ),
            flush=True,
        )
        return 0

    import torch
    from dotenv import load_dotenv

    from scripts.stage2_decode_time_correction import torch_dtype  # noqa: E402
    from scripts.stage2_hint_delta import concept_positions, prompt_cache  # noqa: E402
    from scripts.stage2_hint_state_interchange import generate_sample_batch  # noqa: E402
    from scripts.stage2_interchange_concept_analysis import canon, subjects_of  # noqa: E402
    from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
    from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
    from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
    from src.bd_path import ensure_on_path  # noqa: E402
    from src.stage2_steering import score_reply  # noqa: E402

    load_dotenv()
    torch.set_grad_enabled(False)
    ensure_on_path()
    model = load_tl_model(args.model, n_devices=args.n_devices, n_ctx=args.n_ctx, dtype=torch_dtype(args.dtype), load_mode=args.load_mode)
    dtype = torch_dtype(args.dtype)
    hook_name = validate_hooks(model, [args.layer])[0]
    tokenizer = model.tokenizer
    print(f"using_hook={hook_name}", flush=True)

    prepared = []
    delta_by_row: dict[int, np.ndarray] = {}
    unhinted_by_row: dict[int, np.ndarray] = {}
    for stage1_row in selected_rows:
        source_row_index = int(stage1_row["row_index"])
        gold_concept = stage1_row["ontology_fol_structured"]["hypothesis"]["subject"]
        receiver_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=stage1_row["prompt_text"], model_name=args.model, add_generation_prompt=True)
        receiver_ids = tokenizer(receiver_text, add_special_tokens=False)["input_ids"]
        hinted_user = make_user_prompt(stage1_row, "hint_concept_first")
        hinted_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=hinted_user, model_name=args.model, add_generation_prompt=True)
        hinted_ids = tokenizer(hinted_text, add_special_tokens=False)["input_ids"]
        h_start, r_start, block_len = longest_common_token_block(hinted_ids, receiver_ids)
        if block_len < args.min_block_tokens:
            print(f"skip row {source_row_index}: block={block_len}", flush=True)
            continue
        positions_r = concept_positions(tokenizer, receiver_text, gold_concept, r_start, block_len)
        if not positions_r:
            print(f"skip row {source_row_index}: no concept positions", flush=True)
            continue
        rel = [pos - r_start for pos in positions_r]
        rng = random.Random(args.sample_seed + source_row_index)
        random_rel = sorted(rng.sample(range(block_len), len(rel)))
        hinted_cache = prompt_cache(model, hinted_ids, [hook_name])
        unhinted_cache = prompt_cache(model, receiver_ids, [hook_name])
        h_block = hinted_cache[hook_name][h_start : h_start + block_len].detach().cpu()
        u_block = unhinted_cache[hook_name][r_start : r_start + block_len].detach().cpu()
        concept_delta = h_block[rel] - u_block[rel]
        unhinted_concept = u_block[rel].numpy().astype(np.float32)
        row_class = stage1_row.get("row_class", "failing")
        if row_class == "failing":
            delta_by_row[source_row_index] = concept_delta.numpy().astype(np.float32)
            unhinted_by_row[source_row_index] = unhinted_concept
        all_rel: list[int] = []
        if args.class_mean_c:
            seen = set()
            for name in all_concept_names(stage1_row):
                for pos in concept_positions(tokenizer, receiver_text, name, r_start, block_len):
                    seen.add(pos - r_start)
            all_rel = sorted(seen)
        prepared.append(
            {
                "row_class": row_class,
                "all_rel": all_rel,
                "row": stage1_row,
                "source_row_index": source_row_index,
                "gold_concept": gold_concept,
                "receiver_ids": receiver_ids,
                "hinted_ids": hinted_ids,
                "r_start": r_start,
                "block_len": block_len,
                "rel": rel,
                "random_rel": random_rel,
                "h_block": h_block,
                "concept_delta": concept_delta,
            }
        )
        print(f"prepared row {source_row_index}: block={block_len} concept_tokens={len(rel)}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_after_skip = len(prepared) * len(arms) * args.samples_per_row
    print(f"prepared_rows={len(prepared)} total_generations={total_after_skip}", flush=True)

    basis_cache: dict[tuple[int, int], dict[str, Any]] = {}
    q_cache: dict[int, np.ndarray] = {}

    def basis_for(row_id: int, rank_k: int) -> dict[str, Any]:
        key = (row_id, rank_k)
        if key not in basis_cache:
            basis_cache[key] = fit_pca_basis(delta_by_row, rank_k, exclude_rows={row_id})
        return basis_cache[key]

    recon_norm_by_row: dict[int, float] = {}
    if args.class_mean_b or args.class_mean_c:
        for prep in prepared:
            if prep.get("row_class", "failing") != "failing":
                continue
            row_id = prep["source_row_index"]
            basis = basis_for(row_id, ranks[0])
            recon = rank_k_reconstruction(prep["concept_delta"], basis).numpy().astype(np.float64)
            recon_norm_by_row[row_id] = float(np.linalg.norm(recon, axis=1).mean())
        print(f"recon norms per row: {sorted(recon_norm_by_row.values())[:3]}..", flush=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    basis_records: list[dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for arm_index, arm in enumerate(arms):
            arm_started = time.time()
            for prep_index, prep in enumerate(prepared):
                if (prep.get("row_class", "failing") == "correct") != arm.label.startswith("correct_"):
                    continue
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                start = prep["r_start"]
                token_ids = prep["receiver_ids"]
                fwd_hooks = []
                basis = None
                if arm.kind == "hinted_prompt":
                    token_ids = prep["hinted_ids"]
                elif arm.kind in {"concept_replace", "random_replace"}:
                    rel_positions = prep["rel"] if arm.kind == "concept_replace" else prep["random_rel"]
                    positions = [start + rel_pos for rel_pos in rel_positions]
                    states = prep["h_block"][rel_positions]
                    fwd_hooks.append((hook_name, make_replace_hook(states, positions)))
                elif arm.kind == "rank_k_add":
                    assert arm.rank_k is not None
                    basis = basis_for(prep["source_row_index"], arm.rank_k)
                    recon = rank_k_reconstruction(prep["concept_delta"], basis)
                    positions = [start + rel_pos for rel_pos in prep["rel"]]
                    fwd_hooks.append((hook_name, make_position_add_hook(recon, positions, 1.0)))
                    basis_records.append(
                        {
                            "condition": arm.label,
                            "source_row_index": prep["source_row_index"],
                            "layer": args.layer,
                            "rank_k": arm.rank_k,
                            "basis_mode": arm.basis_mode,
                            "n_source_rows": len(basis["source_rows"]),
                            "excluded": basis["exclude_rows"],
                            "explained_variance_ratio": basis["explained_variance_ratio"],
                        }
                    )
                elif arm.kind in {"fixednorm_allpos_add", "correct_fixednorm_add"}:
                    assert arm.rank_k is not None and class_vector is not None
                    basis = basis_for(prep["source_row_index"], arm.rank_k)
                    components = basis["components"]
                    direction = (class_vector @ components.T) @ components
                    rel_positions = prep["all_rel"] if arm.kind == "fixednorm_allpos_add" else prep["rel"]
                    tiled = np.tile(direction, (len(rel_positions), 1))
                    others = [v for r, v in recon_norm_by_row.items() if r != prep["source_row_index"]]
                    target = np.full((len(rel_positions), 1), float(np.mean(others)))
                    current = np.maximum(np.linalg.norm(tiled, axis=1, keepdims=True), 1e-8)
                    vec = (tiled * (target / current)).astype(np.float32)
                    positions = [start + rel_pos for rel_pos in rel_positions]
                    fwd_hooks.append((hook_name, make_position_add_hook(torch.from_numpy(vec), positions, 1.0)))
                    basis_records.append(
                        {
                            "condition": arm.label,
                            "source_row_index": prep["source_row_index"],
                            "row_class": prep.get("row_class", "failing"),
                            "layer": args.layer,
                            "rank_k": arm.rank_k,
                            "basis_mode": arm.basis_mode,
                            "n_positions_written": len(rel_positions),
                            "n_gold_positions": len(prep["rel"]),
                            "fixed_norm_target": float(target[0, 0]),
                            "mean_add_vector_norm": float(np.linalg.norm(vec, axis=1).mean()),
                        }
                    )
                elif arm.kind in {"shuflabel_proj_add", "signflip_proj_add", "fixednorm_proj_add"}:
                    assert arm.rank_k is not None and class_vector is not None
                    basis = basis_for(prep["source_row_index"], arm.rank_k)
                    components = basis["components"]
                    recon_pca = rank_k_reconstruction(prep["concept_delta"], basis).numpy().astype(np.float64)
                    if arm.kind == "shuflabel_proj_add":
                        source_vec = shuf_vectors[draw_index(arm.label) - 1]
                    elif arm.kind == "signflip_proj_add":
                        source_vec = -class_vector
                    else:
                        source_vec = class_vector
                    direction = (source_vec @ components.T) @ components
                    tiled = np.tile(direction, (len(prep["rel"]), 1))
                    if arm.kind == "fixednorm_proj_add":
                        others = [v for r, v in recon_norm_by_row.items() if r != prep["source_row_index"]]
                        target = np.full((len(prep["rel"]), 1), float(np.mean(others)))
                    else:
                        target = np.linalg.norm(recon_pca, axis=1, keepdims=True)
                    current = np.maximum(np.linalg.norm(tiled, axis=1, keepdims=True), 1e-8)
                    vec = (tiled * (target / current)).astype(np.float32)
                    positions = [start + rel_pos for rel_pos in prep["rel"]]
                    fwd_hooks.append((hook_name, make_position_add_hook(torch.from_numpy(vec), positions, 1.0)))
                    basis_records.append(
                        {
                            "condition": arm.label,
                            "source_row_index": prep["source_row_index"],
                            "layer": args.layer,
                            "rank_k": arm.rank_k,
                            "basis_mode": arm.basis_mode,
                            "mean_add_vector_norm": float(np.linalg.norm(vec, axis=1).mean()),
                            "mean_pca_recon_norm": float(np.linalg.norm(recon_pca, axis=1).mean()),
                            "cos_to_real_proj": float(
                                np.dot(direction, (class_vector @ components.T) @ components)
                                / max(np.linalg.norm(direction) * np.linalg.norm((class_vector @ components.T) @ components), 1e-12)
                            ),
                        }
                    )
                elif arm.kind in {"class_mean_raw_add", "class_mean_proj_add"}:
                    assert arm.rank_k is not None and class_vector is not None
                    basis = basis_for(prep["source_row_index"], arm.rank_k)
                    recon_pca = rank_k_reconstruction(prep["concept_delta"], basis).numpy().astype(np.float32)
                    vec = class_mean_add_matrix(arm, class_vector, basis, recon_pca)
                    positions = [start + rel_pos for rel_pos in prep["rel"]]
                    fwd_hooks.append((hook_name, make_position_add_hook(torch.from_numpy(vec), positions, 1.0)))
                    basis_records.append(
                        {
                            "condition": arm.label,
                            "source_row_index": prep["source_row_index"],
                            "layer": args.layer,
                            "rank_k": arm.rank_k,
                            "basis_mode": arm.basis_mode,
                            "class_vector_norm": float(np.linalg.norm(class_vector)),
                            "mean_add_vector_norm": float(np.linalg.norm(vec, axis=1).mean()),
                            "mean_pca_recon_norm": float(np.linalg.norm(recon_pca, axis=1).mean()),
                            "proj_fraction_of_class_vector": float(
                                np.linalg.norm((class_vector @ basis["components"].T) @ basis["components"])
                                / max(np.linalg.norm(class_vector), 1e-8)
                            ),
                        }
                    )
                elif arm.kind in {"rank_dev_add", "mean_dev_add", "pred_add", "shufpred_add"}:
                    assert dev_basis is not None
                    basis = dev_basis
                    mean = dev_basis["mean"]
                    components = dev_basis["components"]
                    row_unhinted = unhinted_by_row[prep["source_row_index"]].astype(np.float64)
                    true_coeffs = (
                        prep["concept_delta"].numpy().astype(np.float64) - mean[None, :]
                    ) @ components.T
                    record: dict[str, Any] = {
                        "condition": arm.label,
                        "source_row_index": prep["source_row_index"],
                        "layer": args.layer,
                        "rank_k": arm.rank_k,
                        "basis_mode": arm.basis_mode,
                    }
                    if arm.kind == "rank_dev_add":
                        vec = mean[None, :] + true_coeffs @ components
                    elif arm.kind == "mean_dev_add":
                        vec = np.tile(mean, (len(prep["rel"]), 1))
                    else:
                        predictor = predictors["pred" if arm.kind == "pred_add" else "shufpred"]
                        coeffs = _ridge_predict(predictor["model"], row_unhinted)
                        vec = mean[None, :] + coeffs @ components
                        num = (coeffs * true_coeffs).sum(axis=1)
                        den = (
                            np.linalg.norm(coeffs, axis=1) * np.linalg.norm(true_coeffs, axis=1) + 1e-12
                        )
                        record.update(
                            {
                                "ridge_alpha": predictor["alpha"],
                                "ridge_loo_cosine_dev": predictor["loo_cosine"],
                                "pred_true_cosine": float((num / den).mean()),
                                "mean_pred_coeff_norm": float(np.linalg.norm(coeffs, axis=1).mean()),
                                "mean_true_coeff_norm": float(np.linalg.norm(true_coeffs, axis=1).mean()),
                            }
                        )
                    record["mean_add_vector_norm"] = float(np.linalg.norm(vec, axis=1).mean())
                    positions = [start + rel_pos for rel_pos in prep["rel"]]
                    fwd_hooks.append(
                        (hook_name, make_position_add_hook(torch.from_numpy(vec.astype(np.float32)), positions, 1.0))
                    )
                    basis_records.append(record)
                elif arm.kind in {"mean_add", "rand_subspace_add", "rand_norm_add"}:
                    assert arm.rank_k is not None
                    basis = basis_for(prep["source_row_index"], arm.rank_k)
                    recon_pca = rank_k_reconstruction(prep["concept_delta"], basis).numpy().astype(np.float32)
                    vec = control_add_matrix(
                        arm,
                        prep["concept_delta"].numpy().astype(np.float32),
                        basis,
                        recon_pca,
                        control_seed=args.control_seed,
                        shard_index=args.shard_index,
                        source_row_index=prep["source_row_index"],
                        q_cache=q_cache,
                    )
                    positions = [start + rel_pos for rel_pos in prep["rel"]]
                    fwd_hooks.append((hook_name, make_position_add_hook(torch.from_numpy(vec), positions, 1.0)))
                    basis_records.append(
                        {
                            "condition": arm.label,
                            "source_row_index": prep["source_row_index"],
                            "layer": args.layer,
                            "rank_k": arm.rank_k,
                            "basis_mode": arm.basis_mode,
                            "control_seed": args.control_seed,
                            "mean_add_vector_norm": float(np.linalg.norm(vec, axis=1).mean()),
                            "mean_pca_recon_norm": float(np.linalg.norm(recon_pca, axis=1).mean()),
                        }
                    )
                with model.hooks(fwd_hooks=fwd_hooks):
                    batch = generate_sample_batch(
                        model=model,
                        token_ids=token_ids,
                        n_samples=args.samples_per_row,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        stop_at_eos=True,
                        cache_dtype=dtype,
                    )
                for sample_index, (new_ids, reply) in enumerate(batch):
                    score = score_reply(prep["row"], reply)
                    out = {
                        "schema_version": 1,
                        "source_row_index": prep["source_row_index"],
                        "example_id": prep["row"].get("example_id"),
                        "height": prep["row"].get("height"),
                        "model": args.model,
                        "task": args.task,
                        "condition": arm.label,
                        "arm_kind": arm.kind,
                        "reference": arm.reference,
                        "patch_layers": list(arm.layers) if arm.layers else None,
                        "rank_k": arm.rank_k,
                        "basis_mode": arm.basis_mode,
                        "basis_exclude_rows": basis["exclude_rows"] if basis else None,
                        "basis_source_rows": basis["source_rows"] if basis else None,
                        "basis_explained_variance_ratio": basis["explained_variance_ratio"] if basis else None,
                        "sample_index": sample_index,
                        "method": method,
                        "target_variable": "target_concept",
                        "representation_type": "patched_residual_state",
                        "gold_concept": prep["gold_concept"],
                        "n_concept_positions": len(prep["rel"]),
                        "targets_gold_concept": canon(prep["gold_concept"]) in subjects_of(reply),
                        "row_class": prep.get("row_class", "failing"),
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        **score,
                    }
                    rows_out.append(out)
                    fout.write(json.dumps(out, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                strong_rate = float(np.mean([row["is_correct_strong"] for row in rows_out[-args.samples_per_row :]]))
                print(f"arm {arm_index + 1}/{len(arms)} {arm.label} row {prep_index + 1}/{len(prepared)}: P(strong)={strong_rate:.2f}", flush=True)
            print(f"ARM DONE {arm.label}: {time.time() - arm_started:.0f}s elapsed_total={time.time() - started:.0f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    state_arrays = {f"L{args.layer}_row{row_id}_concept_delta": arr for row_id, arr in delta_by_row.items()}
    if args.predicted_coefficients:
        state_arrays.update(
            {f"L{args.layer}_row{row_id}_unhinted_concept_states": arr for row_id, arr in unhinted_by_row.items()}
        )
    states_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(states_output, **state_arrays)

    summary = summarize_generation_rows(rows_out, arms, args.sample_seed)
    hint_validated = hint_validated_summary(rows_out, arms, args.sample_seed)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_rank_k_guard_v2.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": method,
        "representation_type": "patched_residual_state",
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "selection": {
            "jsonl": str(args.jsonl),
            "rule": "parse_ok, strong_incorrect, heights balanced, excluding composite-manifest rows",
            "heights": heights,
            "per_height": args.per_height,
            "selection_seed": args.selection_seed,
            "excluded_rows": sorted(exclude),
            "all_selected_rows": [int(row["row_index"]) for row in all_rows],
            "shard_rows": [int(row["row_index"]) for row in selected_rows],
        },
        "selected_rows": len(selected_rows),
        "prepared_rows": len(prepared),
        "layer": args.layer,
        "rank_list": ranks,
        "arms": [asdict(arm) for arm in arms],
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "hint_validated": hint_validated,
        "basis_records": basis_records,
        "states_output": str(states_output),
        "out_jsonl": str(out_jsonl),
        "output": str(output),
        "summary_md": str(summary_md),
        "n": len(rows_out),
        "resolved_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "controls": (
            [
                "in-job unhinted baseline",
                "hinted-prompt per-row validation arm",
                "in-job rank-8 LOO positive reference",
                "LOO mean-only decomposition arm",
                "random orthonormal rank-k subspace adds, per-position norm-matched to the PCA non-mean component",
                "matched-norm Gaussian per-position adds",
            ]
            if args.specificity_controls
            else [
                "in-job unhinted baseline",
                "hinted-prompt per-row validation arm",
                "dev-basis true-coefficient ceiling arm",
                "dev-basis mean-only floor arm",
                "row-derangement shuffled-ridge control (identical pipeline, broken X->Y pairing)",
            ]
            if args.predicted_coefficients
            else [
                "in-job unhinted baseline",
                "hinted-prompt per-row validation arm",
                "in-job rank-8 LOO positive reference",
                "lever-subspace projection arm (channel dissociation)",
                "matched-norm Gaussian noise floor (item C seed)",
            ]
            if args.class_mean
            else [
                "in-job unhinted baseline (determinism integrity check vs F(ii))",
                "shuffled-label projected class-mean family (4 draws)",
                "sign-flipped real projected vector",
                "fixed pooled-norm donor-free variant",
            ]
            if args.class_mean_b
            else [
                "in-job unhinted baselines on both row classes",
                "all-concept-positions arm (no gold knowledge at inference)",
                "fresh naturally-correct collateral slice (capture/guard/composite-disjoint)",
            ]
            if args.class_mean_c
            else [
                "in-job unhinted baseline",
                "hinted-prompt per-row validation arm",
                "in-job concept-position replacement denominator",
                "matched random-position replacement",
                "leave-one-row-out PCA bases fit within shard",
            ]
        ),
        "predictor_config": (
            {
                "dev_states_npz": str(args.dev_states_npz),
                "dev_rows": sorted(dev_basis["source_rows"]),
                "dev_basis_explained_variance_ratio": dev_basis["explained_variance_ratio"],
                "ridge_alphas": args.ridge_alphas,
                "pred_alpha": predictors["pred"]["alpha"],
                "pred_loo_cosine_dev": predictors["pred"]["loo_cosine"],
                "shufpred_alpha": predictors["shufpred"]["alpha"],
                "shufpred_loo_cosine_dev": predictors["shufpred"]["loo_cosine"],
                "pred_shuffle_seed": args.pred_shuffle_seed,
            }
            if args.predicted_coefficients
            else None
        ),
        "pre_registered_decision_rule": (
            (
                "Gate: pooled rank8_loo CI must exclude zero. Specificity passes if pooled rand_norm CI "
                "includes zero and the paired (rank8 - rand_norm) difference CI excludes zero; fails if "
                "rand_norm CI excludes zero at >=50% of the rank8 effect. mean_only and rand_subspace "
                "decompose the carrier per the wording grid in docs/causal_handle_directions.md item C."
            )
            if args.specificity_controls
            else (
                "Gate: pooled rank8_dev CI must exclude zero (dev-basis transfer). SUCCESS if pooled "
                "rank8_pred CI excludes zero AND paired (pred - mean_only_dev) CI excludes zero AND "
                "paired (pred - shufpred) CI excludes zero. PARTIAL if pred excludes zero but "
                "(pred - mean_only_dev) straddles zero. FAIL otherwise. Exploratory: no current-paper "
                "claim moves; rules in docs/causal_handle_directions.md item E."
            )
            if args.predicted_coefficients
            else (
                "Gate: rank8_loo CI excludes zero. Natural-delta CAUSAL if class_mean_raw CI excludes "
                "zero AND paired (class_mean_raw - rand_norm_d1) CI excludes zero. Channel dissociation "
                "read per docs/causal_handle_directions.md item F(ii). Exploratory: no current-paper "
                "claim moves."
            )
            if args.class_mean
            else (
                "Integrity gate: in-job unhinted_baseline must reproduce F(ii)'s per-row outcomes. "
                "LABEL-SPECIFIC if paired (F(ii) real_proj - shuflabel family) CI excludes zero AND "
                "family < 50% of real_proj. GENERIC if the paired CI includes zero. Sign-flip and "
                "fixednorm are riders. Rules in docs/causal_handle_directions.md item F(ii)-b."
            )
            if args.class_mean_b
            else (
                "POSITION-FREE if fixednorm_allpos CI excludes zero AND >=50% of fixednorm_proj "
                "(paired). COLLATERAL-SAFE if correct-side dP point >= -0.10 AND CI low >= -0.20; "
                "HARMFUL if CI entirely below -0.10. Rules in docs/causal_handle_directions.md "
                "item F(ii)-c. Exploratory."
            )
            if args.class_mean_c
            else (
                "Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of "
                "the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes "
                "the compact-core claim to recognition-gap-style rows rather than failing the guard."
            )
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    write_markdown_summary(summary_md, report)
    print(f"wrote {output}", flush=True)
    print(f"wrote {summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
