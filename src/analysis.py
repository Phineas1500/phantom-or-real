"""Phase 4 analysis: accuracy curves, structural slicing, output strategy."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .bd_path import ensure_on_path
from .export import read_jsonl
from .gemma3_parse import parse_hypotheses as parse_hypotheses_from_response

ensure_on_path()

from evaluate import (  # noqa: E402
    KNOWN_ENTITIES,
    parse_hypothesis_structure,
    wilson_confidence_interval,
)


def wilson_ci(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    return wilson_confidence_interval(p, n)


def first_hypothesis_is_entity_level(model_output: str) -> bool | None:
    """Return True if model's first parsed hypothesis has an entity (proper-noun)
    subject, False if concept-level, None if unparseable.
    """
    hyps = parse_hypotheses_from_response(model_output)
    if not hyps:
        return None
    struct = parse_hypothesis_structure(hyps[0])
    if not struct:
        return None
    subject, _ = struct
    subject_first = subject.strip().split()[0].lower() if subject else ""
    return subject_first in KNOWN_ENTITIES


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    strong = sum(1 for r in rows if r["is_correct_strong"])
    weak = sum(1 for r in rows if r["is_correct_weak"])
    parse_fail = sum(1 for r in rows if r["parse_failed"])

    fm_counter: Counter[str] = Counter()
    for r in rows:
        fm = r.get("failure_mode")
        if fm:
            fm_counter[fm] += 1

    return {
        "n": n,
        "positive_n": strong,
        "negative_n": n - strong,
        "strong_accuracy": strong / n if n else None,
        "weak_accuracy": weak / n if n else None,
        "strong_ci95": wilson_ci(strong / n, n) if n else None,
        "weak_ci95": wilson_ci(weak / n, n) if n else None,
        "parse_fail_rate": parse_fail / n if n else None,
        "failure_modes": dict(fm_counter),
    }


def summarize_by_height(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_h[r["height"]].append(r)
    return {f"h{h}": summarize_group(by_h[h]) for h in sorted(by_h)}


def summarize_by_structure(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Slice by has_direct_member and parent_salience buckets."""
    by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_h[r["height"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for h, hrows in sorted(by_h.items()):
        hkey = f"h{h}"
        out[hkey] = {}

        # Direct member slicing
        for name, pred in (
            ("has_direct_member", lambda r: r["structural"]["has_direct_member"]),
            ("no_direct_member", lambda r: not r["structural"]["has_direct_member"]),
        ):
            subset = [r for r in hrows if pred(r)]
            out[hkey][name] = summarize_group(subset)

        # Parent salience buckets
        for name, lo, hi in (("salience_1_2", 1, 2), ("salience_3_5", 3, 5), ("salience_6p", 6, 10**9)):
            subset = [r for r in hrows if lo <= r["structural"]["parent_salience"] <= hi]
            out[hkey][name] = summarize_group(subset)

    return out


def summarize_output_strategy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_h[r["height"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for h, hrows in sorted(by_h.items()):
        entity = 0
        concept = 0
        unparsed = 0
        for r in hrows:
            decision = first_hypothesis_is_entity_level(r["model_output"])
            if decision is None:
                unparsed += 1
            elif decision:
                entity += 1
            else:
                concept += 1
        out[f"h{h}"] = {
            "entity_level": entity,
            "concept_level": concept,
            "unparseable": unparsed,
            "entity_frac": entity / len(hrows) if hrows else None,
            "concept_frac": concept / len(hrows) if hrows else None,
        }
    return out


def error_type_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Distribution of error_type by height across incorrect examples."""
    by_h: dict[int, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r["is_correct_strong"]:
            continue
        et = r.get("error_type") or "unclassified"
        by_h[r["height"]][et] += 1
    return {f"h{h}": dict(by_h[h]) for h in sorted(by_h)}


def error_type_by_structure(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """Plan §5.5: does error_type distribution correlate with has_direct_member?

    Returns, per height, the absolute error-type counts split into
    {has_direct_member, no_direct_member}.
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["is_correct_strong"]:
            continue
        by_h[r["height"]].append(r)

    for h, hrows in sorted(by_h.items()):
        direct: Counter[str] = Counter()
        nodirect: Counter[str] = Counter()
        for r in hrows:
            et = r.get("error_type") or "unclassified"
            (direct if r["structural"]["has_direct_member"] else nodirect)[et] += 1
        out[f"h{h}"] = {
            "has_direct_member": dict(direct),
            "no_direct_member": dict(nodirect),
        }
    return out


def build_summary(jsonl_path: Path) -> dict[str, Any]:
    rows = read_jsonl(jsonl_path)
    if not rows:
        return {"file": str(jsonl_path), "n": 0}
    return {
        "file": str(jsonl_path),
        "task": rows[0]["task"],
        "model": rows[0]["model"],
        "by_height": summarize_by_height(rows),
        "by_structure": summarize_by_structure(rows),
        "output_strategy": summarize_output_strategy(rows),
        "error_types": error_type_distribution(rows),
        "error_types_by_structure": error_type_by_structure(rows),
    }


def build_summaries(jsonl_paths: Iterable[Path]) -> dict[str, Any]:
    return {p.name: build_summary(p) for p in jsonl_paths}


def write_summaries(paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summaries(paths)

    acc = {}
    struct = {}
    for fname, s in summary.items():
        if s.get("n") == 0:
            continue
        key = f"{s['model']}__{s['task']}"
        acc[key] = {
            "by_height": s["by_height"],
            "output_strategy": s["output_strategy"],
            "error_types": s["error_types"],
            "error_types_by_structure": s["error_types_by_structure"],
        }
        struct[key] = s["by_structure"]

    with (out_dir / "summary_accuracy.json").open("w") as f:
        json.dump(acc, f, indent=2, default=_json_default)
    with (out_dir / "summary_by_structure.json").open("w") as f:
        json.dump(struct, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not JSON-serializable: {type(obj)}")
