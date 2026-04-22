"""Per-example JSONL export with structured FOL representations.

See BEHAVIORAL_DATA_PLAN.md Phase 2.2. One row per generated example, with
raw text, FOL string, and a structured FOL dict serialized from the existing
KnowledgeBase in beyond-deduction/benchmark/evaluate.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .bd_path import ensure_on_path

ensure_on_path()

from evaluate import (  # noqa: E402
    KnowledgeBase,
    parse_ground_truth,
    parse_hypothesis_structure,
)


TASK_CODE_TO_NAME = {
    "property": "infer_property",
    "ontology": "infer_subtype",
    "membership": "infer_membership",  # not in scope but kept for consistency
}


def _sorted_list(value: set[str]) -> list[str]:
    return sorted(value)


def kb_to_dict(kb: KnowledgeBase) -> dict[str, Any]:
    """Serialize KnowledgeBase internal state to JSON-compatible dict."""
    return {
        "membership": {k: _sorted_list(v) for k, v in kb.membership.items()},
        "inheritance": {k: _sorted_list(v) for k, v in kb.inheritance.items()},
        "properties": {k: _sorted_list(v) for k, v in kb.properties.items()},
        "negated_properties": {k: _sorted_list(v) for k, v in kb.negated_properties.items()},
    }


def structured_fol(theories_text: str, observations_text: str, hypothesis_text: str) -> dict[str, Any]:
    """Build the ontology_fol_structured dict.

    Schema matches the plan example (BEHAVIORAL_DATA_PLAN.md §2.2): a single
    unified KB holding entity-, concept-, and property-level facts from both
    theories and observations, alongside the structured hypothesis.

    Keys:
      membership          entity -> list[concept]          (from theories + observations)
      inheritance         concept -> list[parent concept]  (from theories)
      properties          subject -> list[property]        (positive)
      negated_properties  subject -> list[property]        (negated)
      hypothesis          {type, subject, predicate, negated}
    """
    kb = KnowledgeBase()
    kb.add_from_text(theories_text)
    kb.add_from_text(observations_text)
    kb_struct = kb_to_dict(kb)

    parsed = parse_hypothesis_structure(hypothesis_text)
    if parsed is None:
        hyp_struct: dict[str, Any] = {
            "type": "unparsed",
            "raw": hypothesis_text,
        }
    else:
        subj, pred = parsed
        negated = pred.startswith("not ")
        pred_clean = pred[len("not ") :] if negated else pred
        hyp_struct = {
            "type": "rule",
            "subject": subj,
            "predicate": pred_clean,
            "negated": negated,
        }

    kb_struct["hypothesis"] = hyp_struct
    return kb_struct


def build_row(
    *,
    example_id: str,
    task: str,
    height: int,
    model: str,
    prompt_text: str,
    system_prompt: str,
    ontology: Any,
    model_output: str,
    is_correct_strong: bool,
    is_correct_weak: bool,
    quality_score: float,
    parse_failed: bool,
    failure_mode: str | None,
    error_type: str | None,
    structural: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble a single JSONL row conforming to the plan's schema."""
    return {
        "example_id": example_id,
        "task": TASK_CODE_TO_NAME.get(task, task),
        "height": height,
        "model": model,
        "prompt_text": prompt_text,
        "system_prompt": system_prompt,
        "ground_truth": ontology.hypotheses,
        "model_output": model_output,
        "is_correct_strong": bool(is_correct_strong),
        "is_correct_weak": bool(is_correct_weak),
        "quality_score": float(quality_score),
        "parse_failed": bool(parse_failed),
        "failure_mode": failure_mode,
        "error_type": error_type,
        "structural": dict(structural),
        "ontology_raw": {
            "theories": ontology.theories,
            "observations": ontology.observations,
            "hypotheses": ontology.hypotheses,
        },
        "ontology_fol_string": {
            "theories": getattr(ontology, "fol_theories", None),
            "observations": getattr(ontology, "fol_observations", None),
            "hypotheses": getattr(ontology, "fol_hypotheses", None),
        },
        "ontology_fol_structured": structured_fol(
            ontology.theories, ontology.observations, ontology.hypotheses
        ),
    }


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
