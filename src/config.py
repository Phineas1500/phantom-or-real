"""Project-wide constants matching BEHAVIORAL_DATA_PLAN.md.

Import from here rather than hard-coding magic values in scripts.
"""

from __future__ import annotations

import hashlib

# Phase 2.4: per-height example counts for the full run
HEIGHT_SAMPLE_SIZES: dict[int, int] = {1: 1000, 2: 2000, 3: 3000, 4: 5000}

# Pilot counts (Phase 1.3)
PILOT_PER_HEIGHT: int = 50

# Tasks in scope (membership is explicitly out of scope; see plan "Not in scope")
TASK_CODES: tuple[str, ...] = ("property", "ontology")
HEIGHTS: tuple[int, ...] = (1, 2, 3, 4)


# Phase 2.5: deterministic seed per (task, height).
#
# The shipped dataset under data/full was generated with an earlier `get_seed`
# that used Python's builtin hash(), which is randomized per-process (per
# PYTHONHASHSEED). That made `python -m src.generate_examples` silently
# non-reproducible across invocations — a real bug caught by external review.
#
# Resolution: the exact seeds used for each (task, height) in the shipped
# dataset are pinned below in SHIPPED_SEEDS. They were read back from the
# `seed` field stored in each data/full/examples_*.pkl. get_seed() returns
# those pinned values so `python -m src.generate_examples --counts full` now
# reproduces the shipped examples byte-for-byte.
#
# For any (task, height) not in SHIPPED_SEEDS, fall back to a stable md5-
# derived seed — deterministic across processes, not dependent on PYTHONHASHSEED.
SHIPPED_SEEDS: dict[tuple[str, int], int] = {
    ("property", 1): 1319552879,
    ("property", 2): 1052943057,
    ("property", 3):  223589659,
    ("property", 4):   14454230,
    ("ontology", 1): 1141115333,
    ("ontology", 2): 1841570772,
    ("ontology", 3):  237012278,
    ("ontology", 4): 1927955603,
}


def get_seed(task_type: str, height: int) -> int:
    if (task_type, height) in SHIPPED_SEEDS:
        return SHIPPED_SEEDS[(task_type, height)]
    digest = hashlib.md5(f"{task_type}_h{height}".encode()).hexdigest()
    return int(digest[:8], 16)


# System prompt — keep typo "assitant" to match beyond-deduction/benchmark/run_experiments.py
# and the v2 paper's generate.py (plan Implementation Notes / Test-time prompt format).
SYSTEM_PROMPT: str = (
    "You are a helpful assitant that performs abduction and induction reasoning.\n"
    "        Your job is to come up with hypotheses that explain observations with given theories. "
    "Each hypothesis should explain as many observations as possible.\n"
    "        You can come up with multiple hypotheses and each hypothesis should take one line "
    "with the format A is B or A is not B.\n"
    "    .   Only output final hypotheses.\n "
)


def make_user_prompt(ontology) -> str:
    """Mirror run_experiments.make_user_prompt so model input is byte-identical."""
    return (
        "Q: "
        + ontology.theories
        + " We observe that: "
        + ontology.observations
        + " Please come up with hypothesis to explain observations."
    )


# Project-specific Modal endpoints. Keys are the served-model-name registered by
# each deployment (--served-model-name in beyond-deduction/deployment/*.py),
# which is what you pass to --model on the inference script. Workspace prefix
# is hard-coded since this repo belongs to that workspace; swap if you fork.
MODAL_ENDPOINTS: dict[str, str] = {
    "gemma3-27b": "https://phineas1500--gemma3-27b-inference-serve.modal.run/v1",
    "gemma3-4b": "https://phineas1500--gemma3-4b-inference-serve.modal.run/v1",
}
