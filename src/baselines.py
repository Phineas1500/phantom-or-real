"""B0 metadata-only baseline family for Stage 2 probe comparison.

Pre-output baselines (fair comparison for activation probes):
  b0_height    — single feature: height.
  b0_prompt    — height + prompt_token_count + structural features.
  b0_namefreq  — b0_prompt + train-set concept-name frequency per class.

Diagnostic (post-output, not a fair activation-probe baseline):
  d0_parsefail — b0_prompt + parse_failed (always 0 inside splits; kept as
                 an interpretation tool when run on the full un-split corpus).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.utils import resample

C_GRID = [0.01, 0.1, 1.0, 10.0]
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0

VARIANT_FEATURE_NAMES: dict[str, list[str]] = {
    "b0_height": ["height"],
    "b0_prompt": [
        "height",
        "prompt_token_count",
        "num_theories_axioms",
        "num_observations",
        "parent_salience",
        "num_direct_paths",
    ],
    "b0_namefreq": [
        "height",
        "prompt_token_count",
        "num_theories_axioms",
        "num_observations",
        "parent_salience",
        "num_direct_paths",
        "namefreq_pos",
        "namefreq_neg",
    ],
    "d0_parsefail": [
        "height",
        "prompt_token_count",
        "num_theories_axioms",
        "num_observations",
        "parent_salience",
        "num_direct_paths",
        "parse_failed",
    ],
}


def compute_token_counts(
    rows: Sequence[dict],
    tokenizer,
    *,
    model_name: str = "google/gemma-3-4b-it",
) -> dict[str, int]:
    """Tokenize the rendered chat prompt for each row and return a count dict.

    Must be called from the script (not this module) since it requires an
    already-loaded HF tokenizer to avoid importing transformers at module level.
    """
    from src.messages import render_chat_text

    result: dict[str, int] = {}
    for row in rows:
        text = render_chat_text(
            tokenizer,
            system=row["system_prompt"],
            user=row["prompt_text"],
            model_name=model_name,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        result[row["example_id"]] = len(ids)
    return result


def _namefreq_lookup(
    rows: Sequence[dict],
    split_assignment: dict[str, str],
) -> dict[str, tuple[int, int]]:
    """Count concept-name occurrences in TRAIN rows only, by class.

    Returns {concept_name_lower: (pos_count, neg_count)}.
    Counts are computed exclusively from rows assigned "train" so that
    test/val rows see only train-derived frequencies — no leakage.
    """
    pos: dict[str, int] = defaultdict(int)
    neg: dict[str, int] = defaultdict(int)
    for row in rows:
        eid = row["example_id"]
        if split_assignment.get(eid) != "train":
            continue
        concept = row["structural"]["target_concept"].lower()
        if row.get("is_correct_strong"):
            pos[concept] += 1
        else:
            neg[concept] += 1
    all_concepts = set(pos) | set(neg)
    return {c: (pos[c], neg[c]) for c in all_concepts}


def _build_xy(
    rows: Sequence[dict],
    variant: str,
    *,
    token_counts: dict[str, int] | None,
    namefreq: dict[str, tuple[int, int]] | None,
    include_parse_failed: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, example_ids) arrays for the given rows.

    Rows with parse_failed=True are excluded unless include_parse_failed=True.
    """
    if variant not in VARIANT_FEATURE_NAMES:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANT_FEATURE_NAMES)}")
    if variant in ("b0_prompt", "b0_namefreq", "d0_parsefail") and token_counts is None:
        raise ValueError(f"variant {variant!r} requires token_counts dict")
    if variant == "b0_namefreq" and namefreq is None:
        raise ValueError("b0_namefreq requires namefreq dict (call _namefreq_lookup first)")

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    ids: list[str] = []

    for row in rows:
        eid = row["example_id"]
        pf = bool(row.get("parse_failed", False))
        if pf and not include_parse_failed:
            continue
        label = int(bool(row.get("is_correct_strong", False)))
        st = row.get("structural", {})

        vec: list[float] = []
        if variant == "b0_height":
            vec = [float(row["height"])]

        elif variant == "b0_prompt":
            vec = [
                float(row["height"]),
                float(token_counts[eid]),  # type: ignore[index]
                float(st.get("num_theories_axioms", 0)),
                float(st.get("num_observations", 0)),
                float(st.get("parent_salience", 0)),
                float(st.get("num_direct_paths", 0)),
            ]

        elif variant == "b0_namefreq":
            concept = st.get("target_concept", "").lower()
            nf_pos, nf_neg = namefreq.get(concept, (0, 0))  # type: ignore[union-attr]
            vec = [
                float(row["height"]),
                float(token_counts[eid]),  # type: ignore[index]
                float(st.get("num_theories_axioms", 0)),
                float(st.get("num_observations", 0)),
                float(st.get("parent_salience", 0)),
                float(st.get("num_direct_paths", 0)),
                float(nf_pos),
                float(nf_neg),
            ]

        elif variant == "d0_parsefail":
            vec = [
                float(row["height"]),
                float(token_counts[eid]),  # type: ignore[index]
                float(st.get("num_theories_axioms", 0)),
                float(st.get("num_observations", 0)),
                float(st.get("parent_salience", 0)),
                float(st.get("num_direct_paths", 0)),
                float(pf),
            ]

        X_rows.append(vec)
        y_rows.append(label)
        ids.append(eid)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), ids


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, yp)))
    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, 100 * alpha)), float(np.percentile(aucs, 100 * (1 - alpha)))


def _optimal_threshold_balanced_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Return (best_balanced_accuracy, best_threshold) tuned on the given set."""
    best_ba, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        ba = float(balanced_accuracy_score(y_true, y_pred))
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_ba, best_t


def _per_height_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    heights: np.ndarray,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for h in sorted(np.unique(heights)):
        mask = heights == h
        yt, yp = y_true[mask], y_prob[mask]
        if len(np.unique(yt)) < 2:
            result[str(int(h))] = None
        else:
            result[str(int(h))] = float(roc_auc_score(yt, yp))
    return result


def train_and_evaluate(
    rows: Sequence[dict],
    variant: str,
    split_assignment: dict[str, str],
    *,
    token_counts: dict[str, int] | None = None,
) -> dict:
    """Full train/val C-sweep/test pipeline for one (variant, split) combination.

    split_assignment maps example_id → "train" | "val" | "test" | "excluded".
    Returns a result dict ready to serialize as JSON.
    """
    include_pf = variant == "d0_parsefail"

    # Build namefreq lookup from train rows only (critical: no test leakage).
    namefreq = _namefreq_lookup(rows, split_assignment) if variant == "b0_namefreq" else None

    X_all, y_all, ids_all = _build_xy(
        rows, variant,
        token_counts=token_counts,
        namefreq=namefreq,
        include_parse_failed=include_pf,
    )
    id_to_idx = {eid: i for i, eid in enumerate(ids_all)}
    heights_all = np.array([rows[i]["height"] for i, row in enumerate(rows)
                             if row["example_id"] in id_to_idx], dtype=np.int32)
    # Re-index correctly
    row_by_id = {row["example_id"]: row for row in rows}
    heights_all = np.array([row_by_id[eid]["height"] for eid in ids_all], dtype=np.int32)

    def _partition_indices(part: str) -> np.ndarray:
        return np.array([id_to_idx[eid] for eid in ids_all
                         if split_assignment.get(eid) == part], dtype=np.intp)

    train_idx = _partition_indices("train")
    val_idx = _partition_indices("val")
    test_idx = _partition_indices("test")

    if len(train_idx) == 0:
        raise ValueError("no train rows in split_assignment")
    if len(test_idx) == 0:
        raise ValueError("no test rows in split_assignment")

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # C sweep on val.
    best_C, best_val_auc = C_GRID[0], -1.0
    for C in C_GRID:
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=0)
        clf.fit(X_train, y_train)
        if len(np.unique(y_val)) < 2:
            best_C = C_GRID[-1]
            break
        val_prob = clf.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_prob))
        if val_auc > best_val_auc:
            best_val_auc, best_C = val_auc, C

    # Threshold selection on val.
    clf_final = LogisticRegression(C=best_C, solver="liblinear", max_iter=1000, random_state=0)
    clf_final.fit(X_train, y_train)
    if len(val_idx) > 0 and len(np.unique(y_val)) >= 2:
        val_prob_final = clf_final.predict_proba(X_val)[:, 1]
        _, best_threshold = _optimal_threshold_balanced_accuracy(y_val, val_prob_final)
    else:
        best_threshold = 0.5

    # Final evaluation on test.
    test_prob = clf_final.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) >= 2 else None
    auc_lo, auc_hi = (_bootstrap_auc_ci(y_test, test_prob)
                      if (auc is not None and len(test_idx) >= 30) else (None, None))

    test_pred = (test_prob >= best_threshold).astype(int)
    bal_acc = float(balanced_accuracy_score(y_test, test_pred))

    heights_test = heights_all[test_idx]
    per_h_auc = _per_height_auc(y_test, test_prob, heights_test)

    return {
        "variant": variant,
        "feature_names": VARIANT_FEATURE_NAMES[variant],
        "best_C": best_C,
        "best_threshold": best_threshold,
        "auc": auc,
        "auc_ci_95": [auc_lo, auc_hi],
        "balanced_accuracy": bal_acc,
        "per_height_auc": per_h_auc,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }