"""Stage 2 Phase 0 inventory, split, and metadata-baseline helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = tuple(SPLIT_FRACTIONS)
SPLIT_FAMILY_FIELDS = {"s1": "s1_split", "s2": "s2_split", "s3": "s3_split"}
FEATURE_SETS = ("b0_height", "b0_prompt", "b0_namefreq")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any, *, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stage1_jsonl_paths(jsonl_dir: Path) -> list[Path]:
    paths = sorted(jsonl_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files found in {jsonl_dir}")
    return paths


def read_jsonl_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if line.strip():
                rows.append((idx, json.loads(line)))
    return rows


def load_stage1_records(
    paths: list[Path],
    *,
    models: list[str] | None = None,
    tasks: list[str] | None = None,
) -> list[dict[str, Any]]:
    model_filter = set(models or [])
    task_filter = set(tasks or [])
    records = []
    for path in paths:
        source_file = display_path(path)
        for row_index, row in read_jsonl_rows(path):
            if model_filter and row["model"] not in model_filter:
                continue
            if task_filter and row["task"] not in task_filter:
                continue
            records.append(
                {
                    "source_file": source_file,
                    "row_index": row_index,
                    "row_id": f"{source_file}:{row_index}",
                    "model": row["model"],
                    "task": row["task"],
                    "height": int(row["height"]),
                    "example_id": row.get("example_id"),
                    "is_correct_strong": bool(row["is_correct_strong"]),
                    "parse_failed": bool(row["parse_failed"]),
                    "topology_hash": canonical_topology_hash(row),
                    "row": row,
                }
            )
    return records


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    positive = sum(record["is_correct_strong"] for record in records)
    parse_failed = sum(record["parse_failed"] for record in records)
    non_parse = [record for record in records if not record["parse_failed"]]
    non_parse_positive = sum(record["is_correct_strong"] for record in non_parse)
    failure_modes = Counter(record["row"].get("failure_mode") or "none" for record in records)
    error_types = Counter(record["row"].get("error_type") or "none" for record in records)
    return {
        "n": n,
        "positive_n": positive,
        "negative_n": n - positive,
        "positive_rate": positive / n if n else None,
        "parse_failed_n": parse_failed,
        "parse_failed_rate": parse_failed / n if n else None,
        "non_parse_failed_n": len(non_parse),
        "non_parse_failed_positive_n": non_parse_positive,
        "non_parse_failed_negative_n": len(non_parse) - non_parse_positive,
        "non_parse_failed_positive_rate": non_parse_positive / len(non_parse) if non_parse else None,
        "failure_modes": _counter_to_sorted_dict(failure_modes),
        "error_types": _counter_to_sorted_dict(error_types),
    }


def build_inventory(
    paths: list[Path],
    *,
    low_class_threshold: int = 100,
    models: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict[str, Any]:
    records = load_stage1_records(paths, models=models, tasks=tasks)
    by_model_task_height: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    per_cell: Counter[tuple[str, str, int, bool, bool]] = Counter()

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["task"], record["height"])].append(record)
        per_cell[
            (
                record["model"],
                record["task"],
                record["height"],
                record["parse_failed"],
                record["is_correct_strong"],
            )
        ] += 1

    warnings = []
    for (model, task, height), cell_records in sorted(grouped.items()):
        summary = _summarize_records(cell_records)
        key = f"{model}__{task}"
        by_model_task_height[key][f"h{height}"] = summary
        if (summary["parse_failed_rate"] or 0.0) > 0.05:
            warnings.append(
                {
                    "type": "high_parse_failure_rate",
                    "model": model,
                    "task": task,
                    "height": height,
                    "parse_failed_rate": summary["parse_failed_rate"],
                }
            )
        for label_name, count_name in (
            ("positive", "non_parse_failed_positive_n"),
            ("negative", "non_parse_failed_negative_n"),
        ):
            if summary[count_name] < low_class_threshold:
                warnings.append(
                    {
                        "type": "low_non_parse_class_count",
                        "model": model,
                        "task": task,
                        "height": height,
                        "label": label_name,
                        "count": summary[count_name],
                        "threshold": low_class_threshold,
                    }
                )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "low_class_threshold": low_class_threshold,
        "filters": {
            "models": models or "all",
            "tasks": tasks or "all",
        },
        "files": {
            display_path(path): {
                "rows": len(read_jsonl_rows(path)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
        "total_rows": len(records),
        "by_model_task_height": dict(by_model_task_height),
        "per_cell_counts": [
            {
                "model": model,
                "task": task,
                "height": height,
                "parse_failed": parse_failed,
                "is_correct_strong": is_correct,
                "count": count,
            }
            for (model, task, height, parse_failed, is_correct), count in sorted(per_cell.items())
        ],
        "warnings": warnings,
    }


def _inheritance_edges(row: dict[str, Any]) -> list[tuple[str, str]]:
    inheritance = row["ontology_fol_structured"].get("inheritance", {})
    edges = []
    for child, parents in inheritance.items():
        for parent in parents:
            edges.append((child.lower(), parent.lower()))
    return sorted(edges)


def canonical_topology(row: dict[str, Any]) -> dict[str, Any]:
    """Return a name-invariant ontology shape with entity-membership counts."""
    kb = row["ontology_fol_structured"]
    edges = _inheritance_edges(row)
    nodes = {node for edge in edges for node in edge}
    hypothesis = kb.get("hypothesis", {})
    if hypothesis.get("subject"):
        nodes.add(str(hypothesis["subject"]).lower())
    if row.get("task") == "infer_subtype" and hypothesis.get("predicate"):
        nodes.add(str(hypothesis["predicate"]).lower())

    membership_counts: Counter[str] = Counter()
    for concepts in kb.get("membership", {}).values():
        for concept in concepts:
            concept_key = str(concept).lower()
            nodes.add(concept_key)
            membership_counts[concept_key] += 1

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    children = set()
    for child, parent in edges:
        children_by_parent[parent].append(child)
        children.add(child)

    roots = sorted(nodes - children)
    if not roots:
        roots = sorted(nodes)

    seen: set[str] = set()

    def node_form(node: str) -> str:
        if node in seen:
            return "cycle"
        seen.add(node)
        child_forms = sorted(node_form(child) for child in children_by_parent.get(node, []))
        seen.remove(node)
        return f"m{membership_counts[node]}[{','.join(child_forms)}]"

    root_forms = sorted(node_form(root) for root in roots)
    return {
        "task": row["task"],
        "height": int(row["height"]),
        "roots": root_forms,
    }


def canonical_topology_hash(row: dict[str, Any]) -> str:
    return stable_hash(canonical_topology(row), length=16)


def target_symbol(row: dict[str, Any]) -> str:
    """Return the main target symbol for lexical heldout diagnostics.

    For both in-scope tasks this is the hypothesis subject: the property task's
    target concept, or the subtype task's predicted subtype. Holding this
    symbol out globally tests whether a probe survives unseen target names
    without requiring a new generation run.
    """
    hypothesis = row["ontology_fol_structured"].get("hypothesis", {})
    subject = hypothesis.get("subject") or row.get("structural", {}).get("target_concept")
    if not subject:
        raise ValueError(f"row has no hypothesis subject or target_concept: {row.get('example_id')}")
    return str(subject).lower()


def _seed_for(seed: int, *parts: Any) -> int:
    return int(stable_hash([seed, *parts], length=8), 16)


def _stratified_row_split(records: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    assignments = {}
    by_label: dict[bool, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["is_correct_strong"]].append(record)

    for label, label_records in sorted(by_label.items()):
        rng = random.Random(_seed_for(seed, "s1", label, label_records[0]["model"], label_records[0]["task"], label_records[0]["height"]))
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n >= 3:
            n_val = max(1, round(n * SPLIT_FRACTIONS["val"]))
            n_test = max(1, round(n * SPLIT_FRACTIONS["test"]))
            n_train = max(1, n - n_val - n_test)
        else:
            n_train, n_val = n, 0
            n_test = 0
        for idx, record in enumerate(shuffled):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignments[record["row_id"]] = split
    return assignments


def _group_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n": len(records),
        "positive": sum(record["is_correct_strong"] for record in records),
        "negative": sum(not record["is_correct_strong"] for record in records),
    }


def _balanced_count_keys(records: list[dict[str, Any]]) -> list[str]:
    keys = ["n", "positive", "negative"]
    for height in sorted({record["height"] for record in records}):
        keys.extend(
            [
                f"h{height}_n",
                f"h{height}_positive",
                f"h{height}_negative",
            ]
        )
    return keys


def _balanced_group_counts(records: list[dict[str, Any]], keys: list[str]) -> dict[str, int]:
    counts = {key: 0 for key in keys}
    counts["n"] = len(records)
    counts["positive"] = sum(record["is_correct_strong"] for record in records)
    counts["negative"] = len(records) - counts["positive"]
    for record in records:
        height = record["height"]
        counts[f"h{height}_n"] += 1
        if record["is_correct_strong"]:
            counts[f"h{height}_positive"] += 1
        else:
            counts[f"h{height}_negative"] += 1
    return counts


def _score_group_candidate(
    split_counts: dict[str, dict[str, int]],
    split: str,
    group_count: dict[str, int],
    target_counts: dict[str, dict[str, float]],
    total: dict[str, int],
) -> float:
    score = 0.0
    for key in ("n", "positive", "negative"):
        denom = max(total[key], 1)
        candidate = split_counts[split][key] + group_count[key]
        score += abs(candidate - target_counts[split][key]) / denom
    return score


def _group_heldout_split(records: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["topology_hash"]].append(record)

    if len(groups) < 3:
        return {record["row_id"]: "train" for record in records}

    total = _group_counts(records)
    target_counts = {
        split: {key: total[key] * frac for key in ("n", "positive", "negative")}
        for split, frac in SPLIT_FRACTIONS.items()
    }
    split_counts = {split: {"n": 0, "positive": 0, "negative": 0} for split in SPLITS}

    group_items = list(groups.items())
    rng = random.Random(_seed_for(seed, "s2", records[0]["model"], records[0]["task"], records[0]["height"]))
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    group_assignment = {}
    for group_hash, group_records in group_items:
        group_count = _group_counts(group_records)
        best_split = min(
            SPLITS,
            key=lambda split: _score_group_candidate(
                split_counts,
                split,
                group_count,
                target_counts,
                total,
            ),
        )
        group_assignment[group_hash] = best_split
        for key in ("n", "positive", "negative"):
            split_counts[best_split][key] += group_count[key]

    return {record["row_id"]: group_assignment[record["topology_hash"]] for record in records}


def _target_symbol_heldout_split(records: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["target_symbol"]].append(record)

    if len(groups) < 3:
        return {record["row_id"]: "train" for record in records}

    keys = _balanced_count_keys(records)
    total = _balanced_group_counts(records, keys)
    target_counts = {
        split: {key: total[key] * frac for key in keys}
        for split, frac in SPLIT_FRACTIONS.items()
    }
    split_counts = {split: {key: 0 for key in keys} for split in SPLITS}

    group_items = list(groups.items())
    rng = random.Random(_seed_for(seed, "s3", records[0]["model"], records[0]["task"]))
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    group_assignment = {}
    for symbol, group_records in group_items:
        group_count = _balanced_group_counts(group_records, keys)
        def candidate_score(candidate_split: str) -> float:
            score = 0.0
            for split in SPLITS:
                for key in keys:
                    candidate_value = split_counts[split][key]
                    if split == candidate_split:
                        candidate_value += group_count[key]
                    score += abs(candidate_value - target_counts[split][key]) / max(total[key], 1)
            return score

        best_split = min(SPLITS, key=candidate_score)
        group_assignment[symbol] = best_split
        for key in keys:
            split_counts[best_split][key] += group_count[key]

    return {record["row_id"]: group_assignment[record["target_symbol"]] for record in records}


def make_split_assignments(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    grouped_by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["task"], record["height"])].append(record)
        record["target_symbol"] = target_symbol(record["row"])
        grouped_by_task[(record["model"], record["task"])].append(record)

    s1: dict[str, str] = {}
    s2: dict[str, str] = {}
    s3: dict[str, str] = {}
    for _cell, cell_records in sorted(grouped.items()):
        s1.update(_stratified_row_split(cell_records, seed=seed))
        s2.update(_group_heldout_split(cell_records, seed=seed))
    for _cell, cell_records in sorted(grouped_by_task.items()):
        s3.update(_target_symbol_heldout_split(cell_records, seed=seed))

    assignments = []
    for record in records:
        assignments.append(
            {
                "source_file": record["source_file"],
                "row_index": record["row_index"],
                "model": record["model"],
                "task": record["task"],
                "height": record["height"],
                "example_id": record["example_id"],
                "is_correct_strong": record["is_correct_strong"],
                "parse_failed": record["parse_failed"],
                "topology_hash": record["topology_hash"],
                "target_symbol": record["target_symbol"],
                "s1_split": s1[record["row_id"]],
                "s2_split": s2[record["row_id"]],
                "s3_split": s3[record["row_id"]],
            }
        )
    return assignments


def summarize_split_assignments(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(assignments),
        "families": {},
        "warnings": [],
    }
    for family, field in SPLIT_FAMILY_FIELDS.items():
        family_summary = {}
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in assignments:
            grouped[(row["model"], row["task"], row["height"])].append(row)
        for (model, task, height), rows in sorted(grouped.items()):
            split_counts: dict[str, dict[str, Any]] = {}
            for split in SPLITS:
                subset = [row for row in rows if row[field] == split]
                split_counts[split] = {
                    "n": len(subset),
                    "positive_n": sum(row["is_correct_strong"] for row in subset),
                    "negative_n": sum(not row["is_correct_strong"] for row in subset),
                    "parse_failed_n": sum(row["parse_failed"] for row in subset),
                    "non_parse_positive_n": sum(
                        row["is_correct_strong"] and not row["parse_failed"] for row in subset
                    ),
                    "non_parse_negative_n": sum(
                        (not row["is_correct_strong"]) and not row["parse_failed"]
                        for row in subset
                    ),
                    "topology_group_n": len({row["topology_hash"] for row in subset}),
                    "target_symbol_group_n": len({row.get("target_symbol") for row in subset}),
                }
            is_evaluable = all(
                split_counts[split]["non_parse_positive_n"] > 0
                and split_counts[split]["non_parse_negative_n"] > 0
                for split in SPLITS
            )
            if not is_evaluable:
                summary["warnings"].append(
                    {
                        "type": "split_not_evaluable",
                        "family": family,
                        "model": model,
                        "task": task,
                        "height": height,
                        "reason": "each train/val/test split needs at least one non-parse positive and negative row",
                    }
                )
            family_summary.setdefault(f"{model}__{task}", {})[f"h{height}"] = {
                "is_evaluable": is_evaluable,
                "split_counts": split_counts,
                "topology_group_n": len({row["topology_hash"] for row in rows}),
                "target_symbol_group_n": len({row.get("target_symbol") for row in rows}),
            }
        summary["families"][family] = family_summary
    return summary


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def read_split_assignments(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    assignments = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            assignments[(row["source_file"], int(row["row_index"]))] = row
    return assignments


def attach_splits(records: list[dict[str, Any]], assignments: dict[tuple[str, int], dict[str, Any]]) -> None:
    for record in records:
        split_row = assignments[(record["source_file"], record["row_index"])]
        for field in SPLIT_FAMILY_FIELDS.values():
            if field in split_row:
                record[field] = split_row[field]


def row_names(row: dict[str, Any]) -> set[str]:
    kb = row["ontology_fol_structured"]
    names = set()
    for entity, concepts in kb.get("membership", {}).items():
        names.add(str(entity).lower())
        names.update(str(concept).lower() for concept in concepts)
    for child, parents in kb.get("inheritance", {}).items():
        names.add(str(child).lower())
        names.update(str(parent).lower() for parent in parents)
    for section in ("properties", "negated_properties"):
        for subject, properties in kb.get(section, {}).items():
            names.add(str(subject).lower())
            names.update(str(prop).lower() for prop in properties)
    hypothesis = kb.get("hypothesis", {})
    for key in ("subject", "predicate"):
        if hypothesis.get(key):
            names.add(str(hypothesis[key]).lower())
    return names


def prompt_length_fallback(row: dict[str, Any]) -> int:
    text = f"{row.get('system_prompt', '')}\n\n{row.get('prompt_text', '')}"
    return len(re.findall(r"\S+", text))


def add_prompt_length_fallback(records: list[dict[str, Any]]) -> None:
    for record in records:
        record["prompt_token_count"] = prompt_length_fallback(record["row"])
        record["prompt_length_mode"] = "whitespace"


def add_prompt_token_counts(records: list[dict[str, Any]], *, hf_cache: Path | None) -> None:
    from transformers import AutoTokenizer

    from .messages import render_chat_text

    model_map = {
        "gemma3-27b": "google/gemma-3-27b-it",
        "gemma3-4b": "google/gemma-3-4b-it",
    }
    cache_dir = None
    if hf_cache is not None:
        cache_dir = str(hf_cache if hf_cache.name == "hub" else hf_cache / "hub")

    tokenizers = {}
    for record in records:
        hf_model = model_map.get(record["model"], record["model"])
        if hf_model not in tokenizers:
            kwargs: dict[str, Any] = {"local_files_only": True}
            if cache_dir is not None:
                kwargs["cache_dir"] = cache_dir
            tokenizers[hf_model] = AutoTokenizer.from_pretrained(hf_model, **kwargs)
        tokenizer = tokenizers[hf_model]
        row = record["row"]
        text = render_chat_text(
            tokenizer,
            system=row["system_prompt"],
            user=row["prompt_text"],
            model_name=record["model"],
            add_generation_prompt=True,
        )
        record["prompt_token_count"] = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        record["prompt_length_mode"] = "hf_tokenizer"


def _base_prompt_features(record: dict[str, Any]) -> list[float]:
    row = record["row"]
    structural = row.get("structural", {})
    return [
        float(record["height"]),
        float(record.get("prompt_token_count", prompt_length_fallback(row))),
        float(len(row.get("prompt_text", ""))),
        float(structural.get("num_theories_axioms", 0)),
        float(structural.get("num_observations", 0)),
        float(structural.get("num_direct_paths", 0)),
        float(structural.get("parent_salience", 0)),
    ]


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "b0_height":
        return ["height"]
    names = [
        "height",
        "prompt_token_count",
        "prompt_char_count",
        "num_theories_axioms",
        "num_observations",
        "num_direct_paths",
        "parent_salience",
    ]
    if feature_set == "b0_prompt":
        return names
    if feature_set == "b0_namefreq":
        return names + [
            "namefreq_pos_sum",
            "namefreq_neg_sum",
            "namefreq_pos_mean",
            "namefreq_neg_mean",
            "namefreq_pos_max",
            "namefreq_neg_max",
            "namefreq_distinct_names",
        ]
    raise ValueError(f"unknown feature set: {feature_set}")


def build_name_counts(train_records: list[dict[str, Any]]) -> dict[bool, Counter[str]]:
    counts = {False: Counter(), True: Counter()}
    for record in train_records:
        label = bool(record["is_correct_strong"])
        counts[label].update(row_names(record["row"]))
    return counts


def feature_vector(
    record: dict[str, Any],
    feature_set: str,
    *,
    name_counts: dict[bool, Counter[str]] | None = None,
) -> list[float]:
    if feature_set == "b0_height":
        return [float(record["height"])]
    features = _base_prompt_features(record)
    if feature_set == "b0_prompt":
        return features
    if feature_set != "b0_namefreq":
        raise ValueError(f"unknown feature set: {feature_set}")
    if name_counts is None:
        raise ValueError("b0_namefreq requires train-set name counts")
    names = row_names(record["row"])
    pos_counts = [name_counts[True][name] for name in names]
    neg_counts = [name_counts[False][name] for name in names]
    features.extend(
        [
            float(sum(pos_counts)),
            float(sum(neg_counts)),
            float(sum(pos_counts) / len(names)) if names else 0.0,
            float(sum(neg_counts) / len(names)) if names else 0.0,
            float(max(pos_counts)) if pos_counts else 0.0,
            float(max(neg_counts)) if neg_counts else 0.0,
            float(len(names)),
        ]
    )
    return features


def _safe_auc(y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, scores))


def _split_rows(records: list[dict[str, Any]], split_field: str, split: str) -> list[dict[str, Any]]:
    return [record for record in records if record[split_field] == split and not record["parse_failed"]]


def _predict_scores(model, rows: list[dict[str, Any]], feature_set: str, name_counts) -> list[float]:
    if not rows:
        return []
    x = [feature_vector(record, feature_set, name_counts=name_counts) for record in rows]
    return [float(score) for score in model.predict_proba(x)[:, 1]]


def _class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    positive = sum(record["is_correct_strong"] for record in rows)
    return {
        "n": len(rows),
        "positive_n": positive,
        "negative_n": len(rows) - positive,
    }


def _has_two_classes(rows: list[dict[str, Any]]) -> bool:
    return len({record["is_correct_strong"] for record in rows}) == 2


def _skip_result(status: str, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": status,
        "train": _class_counts(train_rows),
        "val": _class_counts(val_rows),
        "test": _class_counts(test_rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
    }


def train_metadata_baselines(
    records: list[dict[str, Any]],
    *,
    split_families: list[str] | None = None,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    families = split_families or list(SPLIT_FAMILY_FIELDS)
    unknown = sorted(set(families) - set(SPLIT_FAMILY_FIELDS))
    if unknown:
        raise ValueError(f"unknown split families: {unknown}")

    output: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_sets": {name: feature_names(name) for name in FEATURE_SETS},
        "split_families": families,
        "results": {},
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["task"])].append(record)

    for (model_name, task), group_records in sorted(grouped.items()):
        cell_key = f"{model_name}__{task}"
        output["results"][cell_key] = {}
        for family in families:
            split_field = SPLIT_FAMILY_FIELDS[family]
            train_rows = _split_rows(group_records, split_field, "train")
            val_rows = _split_rows(group_records, split_field, "val")
            test_rows = _split_rows(group_records, split_field, "test")
            output["results"][cell_key][family] = {}
            for feature_set in FEATURE_SETS:
                name_counts = build_name_counts(train_rows) if feature_set == "b0_namefreq" else None
                y_train = [int(record["is_correct_strong"]) for record in train_rows]
                if not _has_two_classes(train_rows):
                    output["results"][cell_key][family][feature_set] = _skip_result(
                        "skipped_one_class_train",
                        train_rows,
                        val_rows,
                        test_rows,
                    )
                    continue
                if not _has_two_classes(val_rows) or not _has_two_classes(test_rows):
                    output["results"][cell_key][family][feature_set] = _skip_result(
                        "skipped_no_evaluable_holdout",
                        train_rows,
                        val_rows,
                        test_rows,
                    )
                    continue

                x_train = [
                    feature_vector(record, feature_set, name_counts=name_counts)
                    for record in train_rows
                ]
                best = None
                for c_value in (0.01, 0.1, 1.0, 10.0):
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000),
                    )
                    model.fit(x_train, y_train)
                    val_scores = _predict_scores(model, val_rows, feature_set, name_counts)
                    val_auc = _safe_auc(
                        [int(record["is_correct_strong"]) for record in val_rows],
                        val_scores,
                    )
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
                test_scores = _predict_scores(model, test_rows, feature_set, name_counts)
                test_labels = [int(record["is_correct_strong"]) for record in test_rows]
                per_height = {}
                for height in sorted({record["height"] for record in test_rows}):
                    subset = [record for record in test_rows if record["height"] == height]
                    subset_scores = _predict_scores(model, subset, feature_set, name_counts)
                    per_height[f"h{height}"] = {
                        "n": len(subset),
                        "positive_n": sum(record["is_correct_strong"] for record in subset),
                        "auc": _safe_auc(
                            [int(record["is_correct_strong"]) for record in subset],
                            subset_scores,
                        ),
                    }
                output["results"][cell_key][family][feature_set] = {
                    "status": "ok",
                    "best_c": best["c"],
                    "val_auc": best["val_auc"],
                    "test_auc": _safe_auc(test_labels, test_scores),
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    "n_test": len(test_rows),
                    "train_positive_n": sum(record["is_correct_strong"] for record in train_rows),
                    "val_positive_n": sum(record["is_correct_strong"] for record in val_rows),
                    "test_positive_n": sum(record["is_correct_strong"] for record in test_rows),
                    "per_height": per_height,
                }
    return output


def summarize_b0_results(results: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_pre_output_baseline": {},
        "results": {},
    }
    for cell_key, by_family in results["results"].items():
        summary["results"][cell_key] = {}
        summary["best_pre_output_baseline"][cell_key] = {}
        for family, by_feature in by_family.items():
            ok_items = {
                feature: data
                for feature, data in by_feature.items()
                if data.get("status") == "ok" and data.get("test_auc") is not None
            }
            summary["results"][cell_key][family] = {
                feature: {
                    "test_auc": data.get("test_auc"),
                    "val_auc": data.get("val_auc"),
                    "n_train": data.get("n_train"),
                    "n_test": data.get("n_test"),
                }
                for feature, data in by_feature.items()
            }
            if ok_items:
                best_feature, best_data = max(ok_items.items(), key=lambda item: item[1]["test_auc"])
                summary["best_pre_output_baseline"][cell_key][family] = {
                    "feature_set": best_feature,
                    "test_auc": best_data["test_auc"],
                    "val_auc": best_data["val_auc"],
                }
            else:
                summary["best_pre_output_baseline"][cell_key][family] = None
    return summary
