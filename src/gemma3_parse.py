"""Gemma 3-specific response parser (Phase 2.6).

The upstream `parse_hypotheses_from_response` in beyond-deduction/benchmark/
evaluate.py was built for Gemma 2 outputs and has two bugs that surface on
Gemma 3:

1. Its skip-pattern list includes `^(observation|...|so|...)` without a word
   boundary, so any response starting with "Sorple", "Sopranos", "South...",
   "sometime", etc. is silently dropped. We saw a 20% parse-failure rate on
   infer_subtype h=1 solely from this.

2. Gemma 3 often phrases generalizations as "Being X implies being Y" or
   "X implies Y" instead of "X is Y" / "X are Y". The upstream parser
   requires literal " is " / " are " in the candidate line, so these get
   discarded before normalization even runs.

We wrap the upstream parser: first try it, and if it returns nothing, try
our Gemma 3-aware path. We never suppress valid upstream hits.
"""

from __future__ import annotations

import re

from .bd_path import ensure_on_path

ensure_on_path()

from evaluate import (  # noqa: E402
    parse_hypotheses_from_response as _upstream_parse,
    parse_hypothesis_structure,
    extract_after_thinking,
)


_IMPLIES_BEING = re.compile(
    r"\bbeing\s+(?:a|an|the)?\s*([A-Za-z]+)\s+implies\s+being\s+(?:a|an|the)?\s*([A-Za-z]+)",
    re.IGNORECASE,
)
_IMPLIES_PLAIN = re.compile(
    r"\b(?:a|an|the)?\s*([A-Za-z]+)\s+implies\s+(?:a|an|the)?\s*([A-Za-z]+)",
    re.IGNORECASE,
)
# "Being dark is a property of being a chorper." → "Every chorper is dark"
_PROPERTY_OF_BEING = re.compile(
    r"\bbeing\s+(not\s+)?([A-Za-z]+)\s+is\s+(?:a\s+)?property\s+of\s+being\s+(?:a|an|the)?\s*([A-Za-z]+)",
    re.IGNORECASE,
)
# "Transparency is not a property of lompees." → "Every lompee is not transparent"
# Word-level sibling of _PROPERTY_OF_BEING; no "being" wrapper.
_PROPERTY_OF_BARE = re.compile(
    r"^\s*([A-Za-z]+)\s+is\s+(not\s+)?(?:a\s+)?property\s+of\s+(?:a|an|the)?\s*([A-Za-z]+)\s*\.?\s*$",
    re.IGNORECASE,
)
_EVERY_IS = re.compile(
    r"^\s*(?:every|each|all)\s+([A-Za-z]+)\s+(?:is|are)\s+(?:a|an|the)?\s*(not\s+)?([A-Za-z]+)\s*\.?\s*$",
    re.IGNORECASE,
)


def _rewrite_implies(line: str) -> str | None:
    """Convert Gemma 3 rephrasings to canonical 'Every X is Y' form."""
    # "X is a property of Y" / "X is not a property of Y"  (e.g. "Transparency is not a property of lompees.")
    m = _PROPERTY_OF_BARE.match(line)
    if m:
        prop, neg, concept = m.group(1), m.group(2), m.group(3)
        negation = "not " if neg else ""
        return f"Every {concept} is {negation}{prop}"
    # "Being X is a property of being Y"
    m = _PROPERTY_OF_BEING.search(line)
    if m:
        negation = "not " if m.group(1) else ""
        return f"Every {m.group(3)} is {negation}{m.group(2)}"
    m = _IMPLIES_BEING.search(line)
    if m:
        return f"Every {m.group(1)} is {m.group(2)}"
    m = _IMPLIES_PLAIN.search(line)
    if m:
        return f"Every {m.group(1)} is {m.group(2)}"
    return None


def _hedged(line: str) -> bool:
    """Model hedging: 'X is Y or X is not Y' or similar disjunction of opposites.

    We treat these as genuine model uncertainty, not parseable hypotheses.
    Detection is liberal — any 'or' in a line whose left and right sides each
    look like a hypothesis qualifies.
    """
    if " or " not in line.lower():
        return False
    if re.search(r"\bis\s+[\w]+\s+or\s+.+is\s+not\s+", line, re.IGNORECASE):
        return True
    if re.search(r"\bis\s+not\s+[\w]+\s+or\s+.+is\s+not\s+", line, re.IGNORECASE):
        return True
    if re.search(r"\bare\s+[\w]+\s+or\s+.+are\s+not\s+", line, re.IGNORECASE):
        return True
    return False


def _preprocess_line(line: str) -> str | None:
    """Return canonicalized line, or None to drop it.

    - Rewrites Gemma 3 rephrasings into 'Every X is (not) Y' form.
    - Drops hedging disjunctions (model saying 'X or not X').
    """
    if _hedged(line):
        return None
    rewritten = _rewrite_implies(line)
    return rewritten if rewritten is not None else line


_BAD_SKIP_PREFIXES = (
    "based on",
    "here are",
    "here is",
    "the following",
    "my hypothes",
    "final hypothes",
    "to explain",
    "observation",
    "therefore",
    "thus",
    "because",
    "since",
    "given",
)


def _should_skip(line: str) -> bool:
    low = line.lower().lstrip()
    for prefix in _BAD_SKIP_PREFIXES:
        if low.startswith(prefix + " ") or low == prefix or low.startswith(prefix + ":"):
            return True
    if low.startswith("**"):
        return True
    if re.search(r"hypothes[ie]s?.*:", low):
        return True
    return False


def _extract_candidate_hypotheses(line: str) -> list[str]:
    """Return candidate hypothesis strings from one output line.

    Includes the Gemma 3 'implies' rewrite when applicable.
    """
    rewritten = _rewrite_implies(line)
    if rewritten:
        return [rewritten]
    return []


def parse_hypotheses(response: str) -> list[str]:
    """Drop-in replacement with Gemma 3 coverage.

    Canonicalizes Gemma 3 rephrasings and drops hedging disjunctions *before*
    running the upstream parser, so 'Transparency is a property of lompees' and
    'Every lompee is transparent' both score identically.
    """
    if not response:
        return []

    content, had_thinking = extract_after_thinking(response)
    if content is None:
        return []
    response = content if content else response
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    preprocessed_lines: list[str] = []
    for raw_line in response.splitlines():
        processed = _preprocess_line(raw_line.strip())
        if processed is None:
            continue
        preprocessed_lines.append(processed)
    canonical = "\n".join(preprocessed_lines)

    hits = _upstream_parse(canonical)
    if hits:
        return hits

    out: list[str] = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or (line.startswith("*") and len(line) < 5):
            continue
        if _should_skip(line):
            continue

        line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)
        line = re.sub(r"^hypothesis\s*\d*\s*:?\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(final\s+)?hypothes[ie]s?\s*:?\s*", "", line, flags=re.IGNORECASE)

        for sep in (" because ", " since ", " as ", " given ", " due to "):
            low = line.lower()
            if sep in low:
                idx = low.find(sep)
                line = line[:idx].strip()
                break

        if " is " in line.lower() or " are " in line.lower():
            struct = parse_hypothesis_structure(line)
            if struct:
                subj, pred = struct
                if len(subj.split()) <= 3 and len(pred.split()) <= 3:
                    out.append(line)
                    continue

        for cand in _extract_candidate_hypotheses(line):
            struct = parse_hypothesis_structure(cand)
            if struct:
                subj, pred = struct
                if len(subj.split()) <= 3 and len(pred.split()) <= 3:
                    out.append(cand)

    return out
