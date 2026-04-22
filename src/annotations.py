"""Structural annotations for InAbHyD examples.

Per BEHAVIORAL_DATA_PLAN.md Phase 2.1, we compute per-example structural features
that may enable shortcut-based reasoning (e.g., has_direct_member). These are
attached to every row of the per-example JSONL so Teammate B can slice probe
analyses by structural confounds.

The core primitive is `KnowledgeBase.get_all_concepts_for_entity(entity)` from
beyond-deduction's evaluate.py, which returns (concept, proof_depth) tuples.
An entity is a direct member of target_concept iff proof_depth == 1.
"""

from __future__ import annotations

import re
from typing import Any

from .bd_path import ensure_on_path

ensure_on_path()

from evaluate import (  # noqa: E402  — path set up above
    KnowledgeBase,
    parse_ground_truth,
    parse_hypothesis_structure,
)


def _observation_entities(observations_text: str) -> set[str]:
    """Return the lowercased set of proper-noun entities mentioned in observations.

    The benchmark capitalizes entity names and leaves concepts lowercase, so the
    simple "first letter uppercase" rule is reliable here.
    """
    return {w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b", observations_text)}


def compute_structural_annotations(ontology: Any, task_type: str) -> dict[str, Any]:
    """Compute structural features for a generated Ontology example.

    Args:
        ontology: An Ontology instance from beyond-deduction.
        task_type: 'property' or 'ontology' (subtype). 'membership' is out of scope.

    Returns a dict with the fields documented in BEHAVIORAL_DATA_PLAN.md 2.1:
      target_concept, has_direct_member, num_direct_paths, parent_salience,
      num_theories_axioms, num_observations, tree_height.
      For task_type == 'ontology', also target_subtype and target_supertype.
    """
    gt_hyps = parse_ground_truth(ontology.hypotheses)
    if not gt_hyps:
        raise ValueError(
            f"parse_ground_truth returned no hypotheses for: {ontology.hypotheses!r}"
        )

    parsed = parse_hypothesis_structure(gt_hyps[0])
    if parsed is None:
        raise ValueError(f"parse_hypothesis_structure failed on: {gt_hyps[0]!r}")
    subject, predicate = parsed

    target_concept = subject

    kb = KnowledgeBase()
    kb.add_from_text(ontology.theories)

    observation_entities = _observation_entities(ontology.observations)

    num_direct_paths = 0
    for entity in observation_entities:
        for concept, depth in kb.get_all_concepts_for_entity(entity):
            if concept == target_concept and depth == 1:
                num_direct_paths += 1
                break

    has_direct_member = num_direct_paths > 0
    parent_salience = ontology.theories.lower().count(target_concept.lower())

    annotations: dict[str, Any] = {
        "target_concept": target_concept,
        "has_direct_member": has_direct_member,
        "num_direct_paths": num_direct_paths,
        "parent_salience": parent_salience,
        "num_theories_axioms": ontology.theories.count("."),
        "num_observations": ontology.observations.count("."),
        "tree_height": ontology.config.hops,
    }

    if task_type == "ontology":
        pred_cleaned = predicate[len("not ") :] if predicate.startswith("not ") else predicate
        annotations["target_subtype"] = target_concept
        annotations["target_supertype"] = pred_cleaned

    return annotations
