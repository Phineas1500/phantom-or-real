"""Split utilities for Stage 2 probe experiments.

Exposes:
  make_s1_split  — height-stratified random 70/15/15.
  canonical_skeleton_key — BFS-rename topology fingerprint.
  make_s2_split  — StratifiedGroupKFold skeleton-held-out split.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)

_PARTITION_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# S1 — height-stratified random split
# ---------------------------------------------------------------------------

def make_s1_split(
    rows: Sequence[dict],
    *,
    seed: int,
) -> dict[str, str]:
    """Height-stratified random 70/15/15 split over parse_failed=False rows.

    Returns {example_id: "train" | "val" | "test"}.
    Rows with parse_failed=True are assigned "excluded" and not counted
    toward any partition.
    """
    rng = np.random.default_rng(seed)

    by_height: dict[int, list[str]] = defaultdict(list)
    result: dict[str, str] = {}

    for row in rows:
        eid = row["example_id"]
        if row.get("parse_failed", False):
            result[eid] = "excluded"
            continue
        by_height[row["height"]].append(eid)

    for height, eids in by_height.items():
        arr = np.array(eids)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        for eid in arr[:n_train]:
            result[eid] = "train"
        for eid in arr[n_train : n_train + n_val]:
            result[eid] = "val"
        for eid in arr[n_train + n_val :]:
            result[eid] = "test"

    return result


# ---------------------------------------------------------------------------
# Skeleton key (BFS canonical topology fingerprint)
# ---------------------------------------------------------------------------

def canonical_skeleton_key(row: dict) -> str:
    """Fingerprint the ontology topology independent of concept names.

    Builds parent→children adjacency from inheritance edges, BFS-renames
    every node to an integer starting from the root, then encodes the
    sorted edge list together with (height, task).
    """
    inheritance: dict[str, list[str]] = row.get("ontology_fol_structured", {}).get("inheritance", {})
    height: int = row.get("height", 0)
    task: str = row.get("task", "")

    if not inheritance:
        # Flat h=1 topology: no edges, key is just (height, task).
        return f"h{height}_{task}_flat"

    # Build parent→children from child→[parent] representation.
    children_of: dict[str, list[str]] = defaultdict(list)
    all_nodes: set[str] = set()
    child_nodes: set[str] = set()
    for child, parents in inheritance.items():
        all_nodes.add(child)
        child_nodes.add(child)
        for parent in parents:
            all_nodes.add(parent)
            children_of[parent].append(child)

    # Root = node(s) that are never a child.
    roots = sorted(all_nodes - child_nodes)
    if not roots:
        # Cycle or degenerate; fall back to hash of raw edges.
        raw = json_stable(inheritance)
        return f"h{height}_{task}_degenerate_{hashlib.md5(raw.encode()).hexdigest()[:8]}"

    # BFS rename from roots (sorted for determinism).
    id_map: dict[str, int] = {}
    queue: deque[str] = deque(sorted(roots))
    counter = 0
    while queue:
        node = queue.popleft()
        if node in id_map:
            continue
        id_map[node] = counter
        counter += 1
        for child in sorted(children_of.get(node, [])):
            if child not in id_map:
                queue.append(child)

    # Encode edge set as sorted tuples of integer IDs.
    int_edges = sorted(
        (id_map[child], id_map[parents[0]])
        for child, parents in inheritance.items()
        if child in id_map and parents and parents[0] in id_map
    )
    edge_str = ";".join(f"{c}-{p}" for c, p in int_edges)
    return f"h{height}_{task}_{edge_str}"


def json_stable(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True)


# ---------------------------------------------------------------------------
# S2 — ontology-skeleton-held-out split (StratifiedGroupKFold)
# ---------------------------------------------------------------------------

def make_s2_split(
    rows: Sequence[dict],
    *,
    seed: int,
) -> tuple[dict[str, str], dict]:
    """Skeleton-held-out split using StratifiedGroupKFold.

    Groups = canonical_skeleton_key. Stratification target = (height, is_correct_strong).
    Holds out ~15% of skeletons for val, ~15% for test.

    Returns:
        assignment: {example_id: "train" | "val" | "test" | "excluded"}
        meta: residual imbalance table and per-partition counts.
    """
    filtered = [r for r in rows if not r.get("parse_failed", False)]
    excluded = {r["example_id"]: "excluded"
                for r in rows if r.get("parse_failed", False)}

    if not filtered:
        return excluded, {"error": "no rows after parse_failed filter"}

    example_ids = [r["example_id"] for r in filtered]
    groups = [canonical_skeleton_key(r) for r in filtered]
    strat_labels = [f"h{r['height']}_{int(r.get('is_correct_strong') or 0)}"
                    for r in filtered]

    X_dummy = np.zeros((len(filtered), 1))

    # StratifiedGroupKFold: use n_splits ≈ 1/val_frac ≈ 7 to get ~14% held-out per fold.
    n_splits = round(1.0 / VAL_FRAC)  # 7 folds → each fold is ~14%
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits_iter = sgkf.split(X_dummy, strat_labels, groups)

    # First fold → test; second fold → val; rest → train.
    fold_indices = list(splits_iter)
    # fold_indices[i] = (train_idx, test_idx) where test_idx is the held-out fold.
    # We want two held-out folds: one for val and one for test.
    _, test_idx = fold_indices[0]
    _, val_idx = fold_indices[1]

    test_set = set(test_idx.tolist())
    val_set = set(val_idx.tolist())
    # Any overlap between val and test goes to test.
    val_set -= test_set

    assignment: dict[str, str] = dict(excluded)
    for i, eid in enumerate(example_ids):
        if i in test_set:
            assignment[eid] = "test"
        elif i in val_set:
            assignment[eid] = "val"
        else:
            assignment[eid] = "train"

    # Residual imbalance per partition.
    meta = _compute_residual_imbalance(filtered, assignment)
    return assignment, meta


def _compute_residual_imbalance(rows: list[dict], assignment: dict[str, str]) -> dict:
    counts: dict[str, dict] = {p: defaultdict(int) for p in _PARTITION_NAMES}
    for row in rows:
        eid = row["example_id"]
        part = assignment.get(eid, "excluded")
        if part == "excluded":
            continue
        key = f"h{row['height']}_{int(row.get('is_correct_strong') or 0)}"
        counts[part][key] += 1

    total = sum(sum(v.values()) for v in counts.values())
    return {
        "partition_counts": {p: dict(c) for p, c in counts.items()},
        "total_non_excluded": total,
    }