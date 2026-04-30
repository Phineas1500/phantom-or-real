"""Utilities for Stage 2 Phase B.3.b name-scramble diagnostics."""

from __future__ import annotations

import copy
import random
import re
from collections import defaultdict
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_NATURAL_NAME_CANDIDATES = [
    "amelia", "oliver", "charlotte", "liam", "sophia", "noah", "isabella", "elijah",
    "ava", "james", "mia", "henry", "luna", "lucas", "harper", "benjamin", "evelyn",
    "theodore", "abigail", "mateo", "ella", "jack", "scarlett", "sebastian", "aria",
    "owen", "hazel", "daniel", "camila", "leo", "chloe", "alexander", "penelope",
    "jackson", "violet", "levi", "riley", "mason", "nora", "ezra", "zoey", "asher",
    "mila", "ethan", "aurora", "hudson", "stella", "logan", "ellie", "luca", "paisley",
    "aiden", "nova", "samuel", "emilia", "joseph", "willow", "john", "emma", "david",
    "naomi", "wyatt", "elena", "matthew", "sarah", "julian", "alice", "dylan", "ruby",
    "grayson", "ivy", "isaac", "adeline", "gabriel", "lydia", "anthony", "madeline",
    "thomas", "josephine", "charles", "maya", "christopher", "delilah", "joshua", "maya",
    "andrew", "anna", "lincoln", "clara", "nathan", "lucy", "caleb", "iris", "ryan",
    "eva", "adrian", "leah", "isaiah", "julia", "aaron", "natalie", "eli", "quinn",
]


def _tokenize_symbols(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def extract_symbols_from_structured_fol(ontology_fol_structured: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    membership = ontology_fol_structured.get("membership", {})
    for ent, concepts in membership.items():
        out.add(str(ent).lower())
        out.update(str(c).lower() for c in concepts)
    inheritance = ontology_fol_structured.get("inheritance", {})
    for child, parents in inheritance.items():
        out.add(str(child).lower())
        out.update(str(p).lower() for p in parents)
    for key in ("properties", "negated_properties"):
        section = ontology_fol_structured.get(key, {})
        for subject, props in section.items():
            out.add(str(subject).lower())
            out.update(str(p).lower() for p in props)
    hyp = ontology_fol_structured.get("hypothesis", {})
    if isinstance(hyp, dict):
        for key in ("subject", "predicate"):
            if hyp.get(key):
                out.add(str(hyp[key]).lower())
    return out


def extract_row_symbols(row: dict[str, Any]) -> set[str]:
    symbols = extract_symbols_from_structured_fol(row.get("ontology_fol_structured", {}))
    structural = row.get("structural", {})
    if structural.get("target_concept"):
        symbols.add(str(structural["target_concept"]).lower())
    return symbols


def _random_nonce(rng: random.Random, min_len: int = 4, max_len: int = 9) -> str:
    syllables = [
        "ba", "be", "bi", "bo", "bu", "ca", "ce", "ci", "co", "cu", "da", "de", "di", "do", "du",
        "fa", "fe", "fi", "fo", "fu", "ga", "ge", "gi", "go", "gu", "ka", "ke", "ki", "ko", "ku",
        "la", "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu",
        "pa", "pe", "pi", "po", "pu", "ra", "re", "ri", "ro", "ru", "sa", "se", "si", "so", "su",
        "ta", "te", "ti", "to", "tu", "va", "ve", "vi", "vo", "vu", "za", "ze", "zi", "zo", "zu",
    ]
    target_len = rng.randint(min_len, max_len)
    out = []
    while len("".join(out)) < target_len:
        out.append(rng.choice(syllables))
    nonce = "".join(out)[:target_len]
    if not re.fullmatch(r"[a-z]+", nonce):
        nonce = re.sub(r"[^a-z]", "", nonce) or "nona"
    return nonce


def build_nonce_pool(symbols: set[str], *, seed: int, multiplier: int = 6) -> dict[str, Any]:
    rng = random.Random(seed)
    by_len: dict[int, set[str]] = defaultdict(set)
    for sym in symbols:
        if len(sym) < 2:
            continue
        target_n = max(40, multiplier)
        while len(by_len[len(sym)]) < target_n:
            cand = _random_nonce(rng, min_len=max(3, len(sym) - 1), max_len=len(sym) + 1)
            if cand != sym and cand not in symbols:
                by_len[len(sym)].add(cand)
    return {
        "seed": seed,
        "by_length": {str(k): sorted(v) for k, v in sorted(by_len.items())},
    }


def build_natural_pool(used_symbols: set[str]) -> list[str]:
    out = []
    used = {s.lower() for s in used_symbols}
    for name in _NATURAL_NAME_CANDIDATES:
        if name not in used:
            out.append(name)
    return sorted(set(out))


def _choose_replacement_from_len_pool(
    *,
    original: str,
    by_length: dict[str, list[str]],
    used_targets: set[str],
    forbidden: set[str],
    rng: random.Random,
) -> str:
    candidate_lengths = [len(original), len(original) - 1, len(original) + 1]
    candidates: list[str] = []
    for L in candidate_lengths:
        if L <= 1:
            continue
        candidates.extend(by_length.get(str(L), []))
    candidates = [c for c in candidates if c not in used_targets and c not in forbidden and c != original]
    if not candidates:
        # guaranteed fallback
        base = _random_nonce(rng, min_len=max(3, len(original) - 1), max_len=len(original) + 1)
        idx = 1
        while base in used_targets or base in forbidden or base == original:
            base = f"{base}{idx}"
            idx += 1
        return base
    return rng.choice(candidates)


def build_name_mapping(
    row: dict[str, Any],
    *,
    condition: str,
    nonce_pool: dict[str, Any],
    natural_pool: list[str],
    global_forbidden: set[str],
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    symbols = sorted(extract_row_symbols(row))
    if not symbols:
        return {}
    mapping: dict[str, str] = {}
    used_targets: set[str] = set()
    for symbol in symbols:
        if condition == "nonce":
            replacement = _choose_replacement_from_len_pool(
                original=symbol,
                by_length=nonce_pool.get("by_length", {}),
                used_targets=used_targets,
                forbidden=global_forbidden,
                rng=rng,
            )
        elif condition == "natural":
            choices = [
                c for c in natural_pool
                if c not in used_targets and c not in global_forbidden and c != symbol
            ]
            if not choices:
                replacement = _choose_replacement_from_len_pool(
                    original=symbol,
                    by_length=nonce_pool.get("by_length", {}),
                    used_targets=used_targets,
                    forbidden=global_forbidden,
                    rng=rng,
                )
            else:
                replacement = rng.choice(choices)
        else:
            raise ValueError(f"unknown condition: {condition}")
        mapping[symbol] = replacement
        used_targets.add(replacement)
    return mapping


def _replace_in_string(text: str, mapping: dict[str, str]) -> str:
    out = text
    # longest-first avoids partial replacement collisions.
    for src in sorted(mapping, key=len, reverse=True):
        dst = mapping[src]
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return out


def _replace_recursive(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_in_string(value, mapping)
    if isinstance(value, list):
        return [_replace_recursive(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _replace_recursive(v, mapping) for k, v in value.items()}
    return value


def apply_name_mapping(row: dict[str, Any], mapping: dict[str, str], *, condition: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out = _replace_recursive(out, mapping)
    out["namescramble"] = {
        "condition": condition,
        "mapping": mapping,
        "source_example_id": row.get("example_id"),
    }
    out["source_example_id"] = row.get("example_id")
    out["example_id"] = f"{row['example_id']}.{condition}"
    return out
