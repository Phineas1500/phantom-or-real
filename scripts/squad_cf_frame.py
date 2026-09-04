#!/usr/bin/env python3
"""Item WS frame: counterfactual SQuAD in the WikiHop frame schema — a second
knowledge-conflict construction. From the WP′ SQuAD frame, rows whose answer
is a capitalized named entity (no numbers, no dates); the answer is replaced
at every whole-word mention of the paragraph (title included) by a seeded
answer of another row from the same shape bucket (word count capped at 3,
whether it contains a digit), never already present in the paragraph or the
question. The document's answer (the substitute) is the gold; the original
answer is the memory candidate; candidates = the gold, the memory candidate
and up to 18 sentence-bounded paragraph spans (the NQ-Swap enumerator)."""
from __future__ import annotations
import argparse, collections, gzip, json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hotpot_frame import spans, whole_word  # noqa: E402
from nqswap_frame import LOCAL_STOP, NUMERIC_OR_DATE  # noqa: E402


def bucket(a):
    return (min(len(a.split()), 3), any(ch.isdigit() for ch in a))


def replace_all(text, old, new):
    return re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", new, text, flags=re.IGNORECASE)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("results/loop_screen/squad_input.jsonl.gz"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260899)
    p.add_argument("--max-candidates", type=int, default=20)
    args = p.parse_args()
    rows = [json.loads(l) for l in gzip.open(args.src, "rt")]
    named = [r for r in rows if not NUMERIC_OR_DATE.search(r["answer_original"]) and r["answer_original"][:1].isupper() and len(r["answer_original"].split()) <= 4]
    by_bucket = collections.defaultdict(list)
    for r in named:
        by_bucket[bucket(r["answer_original"])].append(r["answer_original"])
    out_rows, stats = [], collections.Counter()
    for k, r in enumerate(named):
        org = r["answer_original"]; rng = random.Random(args.seed + k)
        pool = [a for a in by_bucket[bucket(org)] if a.lower() != org.lower() and not whole_word(a, r["docs"]) and not whole_word(a, r["question"])]
        if not pool:
            stats["skip_no_substitute"] += 1; continue
        sub = rng.choice(sorted(set(pool)))
        docs = replace_all(r["docs"], org, sub)
        if whole_word(org, docs) or not whole_word(sub, docs):
            stats["skip_replacement"] += 1; continue
        counts = collections.Counter()
        for sent in re.split(r"(?<=[\.\!\?;:])\s+", docs):
            counts.update(spans(sent))
        cands = {}
        for c, n in counts.most_common():
            kk = c.lower()
            if kk in cands or kk == sub.lower() or kk == org.lower() or kk in r["question"].lower() or len(kk) < 2 or not whole_word(c, docs):
                continue
            if len(kk.split()) > 4 or kk.endswith("'s") or kk in LOCAL_STOP or re.fullmatch(r"0+|[\d,\.]{1,1}", kk):
                continue
            cands[kk] = kk
            if len(cands) >= args.max_candidates - 2:
                break
        if len(cands) < 3:
            stats["skip_few_candidates"] += 1; continue
        cand_list = [sub.lower(), org.lower()] + list(cands.values())
        random.Random(args.seed + 7 * k).shuffle(cand_list)
        out_rows.append({"id": r["id"] + "_cf", "ds": "squad_cf", "squad_id": r["squad_id"], "docs": docs, "question": r["question"], "relation": "", "subject": "",
                         "query": r["question"], "candidates": cand_list, "answer": sub.lower(), "answer_original": org.lower(), "answer_original_cased": org,
                         "answer_cased": sub, "org_context": r["docs"]})
        stats["eligible"] += 1
    with gzip.open(args.out, "wt") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ncand = [len(r["candidates"]) for r in out_rows]
    print(json.dumps({"stats": dict(stats), "named_source_rows": len(named), "written": len(out_rows), "candidates_mean": sum(ncand) / len(ncand), "candidates_min": min(ncand),
                      "doc_chars_mean": sum(len(r["docs"]) for r in out_rows) / len(out_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
