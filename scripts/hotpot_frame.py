#!/usr/bin/env python3
"""Item WP frame: HotpotQA (distractor setting, validation) in the WikiHop
frame schema, so every WikiHop job runs unchanged. docs = the ten
paragraphs ("Title: sentences", blank-line separated); question = the
free-form question (the prompt builder uses it when present); candidates
= spans enumerated from the paragraphs — paragraph titles, capitalized
multi-word spans, four-digit years and numbers — deduplicated, capped at
--max-candidates with the answer always included and the order seeded.
Rows are kept only if the answer is not yes/no, is at most --max-answer-
words long, and occurs as a whole-word span in the paragraphs (the
write's address). Seeded draw of --n rows among the eligible."""
from __future__ import annotations
import argparse, collections, gzip, json, random, re
from pathlib import Path
import pyarrow.parquet as pq

STOP_CAP = {"The", "A", "An", "In", "On", "At", "It", "He", "She", "They", "This", "That", "These", "Those", "His", "Her", "Its", "Their",
            "There", "When", "Where", "Which", "Who", "What", "How", "Why", "After", "Before", "During", "However", "Although", "Both",
            "Also", "But", "And", "Or", "As", "By", "For", "From", "With", "Of", "To", "Is", "Was", "Are", "Were", "Has", "Had", "Have",
            "Not", "No", "Yes", "One", "Two", "Three", "First", "Second", "Third", "New", "Later", "Since", "Until", "While", "Some", "Many",
            "Most", "Other", "Another", "Each", "Several", "All", "Following", "According", "Because", "Despite", "Under", "Over", "Between"}
CONNECT = {"of", "the", "de", "la", "du", "and", "for", "on", "in", "at", "von", "van", "der", "del", "di", "da", "le", "y", "&"}


def whole_word(cand, text):
    return re.search(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE) is not None


def spans(text):
    out = collections.Counter()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\.\-]*|[^\sA-Za-z0-9]", text)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t[0].isupper() and t not in STOP_CAP:
            j = i + 1; run = [t]
            while j < len(tokens):
                u = tokens[j]
                if u[0].isupper() and u not in STOP_CAP:
                    run.append(u); j += 1
                elif u in CONNECT and j + 1 < len(tokens) and tokens[j + 1][0].isupper() and tokens[j + 1] not in STOP_CAP:
                    run.append(u); j += 1
                else:
                    break
            s = " ".join(run).strip(".-'")
            if len(s) >= 2:
                out[s] += 1
            i = j
        else:
            i += 1
    for y in re.findall(r"(?<!\d)(1[0-9]{3}|20[0-2][0-9])(?!\d)", text):
        out[y] += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260870)
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--max-doc-chars", type=int, default=14000)
    p.add_argument("--max-candidates", type=int, default=40)
    p.add_argument("--max-answer-words", type=int, default=6)
    args = p.parse_args()
    t = pq.read_table(args.parquet).to_pylist()
    eligible, stats = [], collections.Counter()
    for i, ex in enumerate(t):
        ans = ex["answer"].strip()
        if ans.lower() in ("yes", "no") or len(ans.split()) > args.max_answer_words:
            stats["skip_answer_form"] += 1; continue
        paras = ["%s: %s" % (ti, "".join(sents).strip()) for ti, sents in zip(ex["context"]["title"], ex["context"]["sentences"])]
        docs = "\n\n".join(paras)
        if len(docs) > args.max_doc_chars:
            stats["skip_long"] += 1; continue
        if not whole_word(ans, docs):
            stats["skip_answer_not_in_docs"] += 1; continue
        counts = spans(docs)
        for ti in ex["context"]["title"]:
            counts[ti] += 3
        cands = {}
        for c, n in counts.most_common():
            k = c.lower()
            if k in cands or k == ans.lower() or k in ex["question"].lower():
                continue
            if not whole_word(c, docs):
                continue
            cands[k] = c
            if len(cands) >= args.max_candidates - 1:
                break
        cand_list = [ans] + list(cands.values())
        random.Random(args.seed + i).shuffle(cand_list)
        eligible.append({"id": f"HP_dev_{i}", "ds": "hotpot", "hotpot_id": ex["id"], "type": ex["type"], "docs": docs,
                         "question": ex["question"], "relation": "", "subject": "", "query": ex["question"],
                         "candidates": cand_list, "answer": ans})
        stats["eligible"] += 1
    draw = sorted(random.Random(args.seed).sample(range(len(eligible)), min(args.n, len(eligible))))
    with gzip.open(args.out, "wt") as f:
        for j in draw:
            f.write(json.dumps(eligible[j], ensure_ascii=False) + "\n")
    ncand = [len(eligible[j]["candidates"]) for j in draw]
    types = collections.Counter(eligible[j]["type"] for j in draw)
    print(json.dumps({"stats": dict(stats), "drawn": len(draw), "candidates_mean": sum(ncand) / len(ncand), "candidates_min": min(ncand), "candidates_max": max(ncand),
                      "types": dict(types), "doc_chars_mean": sum(len(eligible[j]["docs"]) for j in draw) / len(draw)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
