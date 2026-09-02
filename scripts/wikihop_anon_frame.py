#!/usr/bin/env python3
"""Anonymized-entity WikiHop frame (item WA). For each row: the entity set is
the candidates plus the query subject; an entity is a NAME if it is not
numeric and its document occurrences are predominantly capitalized (title
case) — common-noun candidates ('stadium', 'work') and years are left alone.
Every name entity is replaced, consistently across documents / candidates /
answer / subject, by a pseudonym generated from a seeded syllable grammar
(case-matched, whole-word). Rows are kept only if the GOLD is a name entity
(so memory cannot answer). Output schema = the W0 frame's, plus
'anon_map' (original -> pseudonym) and 'n_renamed'."""
from __future__ import annotations
import argparse, gzip, json, random, re
from pathlib import Path

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "kr", "tr", "sl", "th", "vr"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "ou", "ia"]
CODAS = ["", "n", "r", "l", "s", "th", "nd", "rk", "m", "x"]


def pseudonym(rng: random.Random, n_words: int) -> str:
    words = []
    for _ in range(n_words):
        k = rng.choice([2, 2, 3])
        w = "".join(rng.choice(ONSETS) + rng.choice(NUCLEI) + (rng.choice(CODAS) if j == k - 1 else "") for j in range(k))
        words.append(w.capitalize())
    return " ".join(words)


def is_name(entity: str, docs: str) -> bool:
    if not entity.strip() or re.fullmatch(r"[\d\s.,'/-]+", entity):
        return False
    occ = re.findall(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", docs, re.IGNORECASE)
    if not occ:
        return False
    cap = sum(o[0].isupper() for o in occ)
    return cap / len(occ) >= 0.6


def match_case(src: str, pseud: str) -> str:
    if src.isupper():
        return pseud.upper()
    if src[0].islower():
        return pseud.lower()
    return pseud


def anonymize(row: dict, seed: int) -> dict | None:
    rng = random.Random(f"{seed}:{row['id']}")
    docs = row["docs"]
    entities = sorted(set(row["candidates"]) | {row["subject"]}, key=lambda e: -len(e))
    names = [e for e in entities if is_name(e, docs)]
    if row["answer"] not in names:
        return None
    amap = {e: pseudonym(rng, max(1, min(3, len(e.split())))) for e in names}
    def repl_text(text: str) -> str:
        for e in names:  # longest first: avoids partial overwrite of multiword names
            text = re.sub(r"(?<!\w)" + re.escape(e) + r"(?!\w)", lambda m: match_case(m.group(0), amap[e]), text, flags=re.IGNORECASE)
        return text
    out = dict(row)
    out["docs"] = repl_text(docs)
    out["candidates"] = [amap.get(c, c).lower() if c in amap else c for c in row["candidates"]]
    out["answer"] = amap[row["answer"]].lower()
    out["subject"] = amap.get(row["subject"], row["subject"]).lower() if row["subject"] in amap else row["subject"]
    if not re.search(r"(?<!\w)" + re.escape(out["answer"]) + r"(?!\w)", out["docs"], re.IGNORECASE):
        return None  # gold only occurred inside a longer renamed entity
    out["anon_map"] = amap
    out["n_renamed"] = len(names)
    out["id"] = row["id"] + "_anon"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="a real-entity frame (W0 schema)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260831)
    args = p.parse_args()
    n_in, n_out = 0, 0
    with gzip.open(args.out, "wt") as f:
        for line in gzip.open(args.source, "rt"):
            r = json.loads(line)
            n_in += 1
            a = anonymize(r, args.seed)
            if a is None:
                continue
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"{n_in} rows in → {n_out} anonymized rows (gold is a name entity) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
