"""Locate and import the beyond-deduction `benchmark` package on sys.path.

The beyond-deduction repo is an external dependency that is NOT vendored into
this project. Set BD_PATH to point at a checkout of
https://github.com/<upstream>/beyond-deduction, or symlink it to
./third_party_beyond_deduction (default).
"""

import os
import sys
from pathlib import Path


def locate_beyond_deduction() -> Path:
    env = os.environ.get("BD_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent.parent
    candidates.append(here / "third_party_beyond_deduction")
    candidates.append(Path.home() / "beyond-deduction")
    for p in candidates:
        if (p / "benchmark" / "ontology.py").exists():
            return p
    raise RuntimeError(
        "Could not find beyond-deduction. Set BD_PATH env var or place a clone "
        "at ./third_party_beyond_deduction or ~/beyond-deduction."
    )


def ensure_on_path() -> Path:
    bd = locate_beyond_deduction()
    benchmark_dir = str(bd / "benchmark")
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    _apply_normalize_singular_patch()
    return bd


_NORM_PATCH_APPLIED = False


def _apply_normalize_singular_patch() -> None:
    """Fix an upstream bug in beyond-deduction/benchmark/evaluate.py.

    `normalize_to_singular` strips trailing 's' as a pluralization rule without
    a carve-out for proper nouns. Four entity names from morphology.py's
    `_available_entity_names` end in 's' and get mangled:

        Thomas    -> 'thoma'
        Charles   -> 'charle'
        James     -> 'jame'
        Nicholas  -> 'nichola'

    This leaks into `KnowledgeBase.add_fact`: when the subject no longer
    matches `KNOWN_ENTITIES`, the fact is stored as a concept-to-concept
    inheritance instead of an entity-to-concept membership, corrupting
    downstream structural annotations (has_direct_member, num_direct_paths,
    parent_salience) and the `ontology_fol_structured` dict. Scoring is
    unaffected because the same mangling is applied to both prediction and
    ground truth.

    We monkey-patch rather than editing the upstream file because
    `third_party_beyond_deduction` is a symlink that may back other work.
    The patch is idempotent.
    """
    global _NORM_PATCH_APPLIED
    if _NORM_PATCH_APPLIED:
        return

    import evaluate  # noqa: E402  — only resolvable after sys.path setup

    original = evaluate.normalize_to_singular
    entities = evaluate.KNOWN_ENTITIES

    def normalize_to_singular_patched(word: str) -> str:
        if word.lower().strip() in entities:
            return word.strip()
        return original(word)

    evaluate.normalize_to_singular = normalize_to_singular_patched
    _NORM_PATCH_APPLIED = True
