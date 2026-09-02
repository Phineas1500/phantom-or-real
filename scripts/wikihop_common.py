"""Item W shared, dependency-free helpers (copied verbatim into the H100 job
containers): the std/closed prompt constructions (identical to the screening's)
and the pinned answer normalization. Registered: docs/causal_handle_directions.md
item W."""
import re
import unicodedata

SYSTEM = "You are a careful reading assistant. Answer concisely."


def std_closed_prompts(r):
    q = f"Based on the documents, what is the '{r['relation']}' of {r['subject']}?"
    cands = "\n".join(f"- {c}" for c in r["candidates"])
    return {
        "std": f"Documents:\n{r['docs']}\n\nQuestion: {q}\nCandidates:\n{cands}\n\nAnswer with exactly one candidate, nothing else.",
        "closed": f"Question: {q}\nCandidates:\n{cands}\n\nAnswer with exactly one candidate, nothing else.",
    }


def normalize_answer(s):
    """Pinned normalization (W0): first line, NFKC, lowercase, leading bullets/
    quotes and trailing quotes/asterisks/periods stripped, whitespace collapsed."""
    s = unicodedata.normalize("NFKC", s).strip().split("\n")[0].strip().lower()
    s = re.sub(r"^[\-\*•\s\"'`]+", "", s)
    s = re.sub(r"[\"'`\*\.\s]+$", "", s)
    return re.sub(r"\s+", " ", s)


def exact_match(output, gold):
    return normalize_answer(output) == normalize_answer(gold)


def contains_match(output, gold):
    g = normalize_answer(gold)
    return bool(g) and g in normalize_answer(output)


def hint_first_prompt(r, cand):
    """Item WH: the hint precedes the documents so mention positions are
    downstream of it (the screening's hint-after form cannot move them)."""
    return f"Hint: pay close attention to {cand}.\n\n" + std_closed_prompts(r)["std"]
