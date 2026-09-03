#!/usr/bin/env python3
"""Item WP′ frame: SQuAD v1.1 validation in the WikiHop frame schema.
docs = "Title: paragraph"; question = the free-form question; candidates
= the paragraph's capitalized spans, numbers and years (the same
enumerator as the HotpotQA frame), lowercased like WikiHop's, deduplicated,
capped at --max-candidates with the answer always present and a seeded
order. Rows are kept only if the (first) answer is an entity-like span —
a capitalized span, a number, or a year — that occurs as a whole-word span
of the paragraph, so the candidate set is of the answer's kind. Seeded
draw of --n rows among the eligible."""
from __future__ import annotations
import argparse, collections, gzip, json, random, re, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hotpot_frame import spans, whole_word  # noqa: E402


def entity_like(ans):
    if re.fullmatch(r"[\d,\.\-–]+(\s?(%|percent|million|billion|thousand|km|miles?|years?))?", ans):
        return True
    words = ans.split()
    caps = sum(1 for w in words if w[:1].isupper() or w[:1].isdigit())
    return caps >= max(1, len(words) - 2) and len(words) <= 6


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260877)
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--max-candidates", type=int, default=20)
    args = p.parse_args()
    t = pq.read_table(args.parquet).to_pylist()
    eligible, stats = [], collections.Counter()
    for i, ex in enumerate(t):
        ans = ex["answers"]["text"][0].strip().strip(".,;")
        docs = f"{ex['title'].replace('_', ' ')}: {ex['context']}"
        if not entity_like(ans):
            stats["skip_answer_form"] += 1; continue
        if not whole_word(ans, docs):
            stats["skip_answer_not_in_docs"] += 1; continue
        counts = spans(ex["context"])
        for m in re.findall(r"(?<![\w\.])\d[\d,\.]*(?![\w])", ex["context"]):
            counts[m.strip(".,")] += 1
        cands = {}
        for c, n in counts.most_common():
            k = c.lower()
            if k in cands or k == ans.lower() or k in ex["question"].lower() or len(k) < 2:
                continue
            if not whole_word(c, docs):
                continue
            cands[k] = k
            if len(cands) >= args.max_candidates - 1:
                break
        if len(cands) < 4:
            stats["skip_few_candidates"] += 1; continue
        cand_list = [ans.lower()] + list(cands.values())
        random.Random(args.seed + i).shuffle(cand_list)
        eligible.append({"id": f"SQ_dev_{i}", "ds": "squad", "squad_id": ex["id"], "docs": docs, "question": ex["question"],
                         "relation": "", "subject": "", "query": ex["question"], "candidates": cand_list, "answer": ans.lower(),
                         "answer_original": ans, "all_answers": sorted(set(a.strip().strip(".,;").lower() for a in ex["answers"]["text"]))})
        stats["eligible"] += 1
    draw = sorted(random.Random(args.seed).sample(range(len(eligible)), min(args.n, len(eligible))))
    with gzip.open(args.out, "wt") as f:
        for j in draw:
            f.write(json.dumps(eligible[j], ensure_ascii=False) + "\n")
    ncand = [len(eligible[j]["candidates"]) for j in draw]
    print(json.dumps({"stats": dict(stats), "drawn": len(draw), "candidates_mean": sum(ncand) / len(ncand), "candidates_min": min(ncand),
                      "doc_chars_mean": sum(len(eligible[j]["docs"]) for j in draw) / len(draw)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
