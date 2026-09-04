#!/usr/bin/env python3
"""Item WK frame: NQ-Swap (Longpre et al. 2021) in the WikiHop frame schema —
the knowledge-conflict regime. Each Natural Questions row has its answer
entity in the Wikipedia paragraph replaced by a type-matched entity; the
document's answer (the substituted one) is the gold, the original answer is
what the model's memory says. One substitution per question (seeded), rows
kept only if the substituted answer is a whole-word span of the substituted
context and the original answer is not (so the memory candidate has no
address in the document). Candidates = capitalized spans, numbers and years
of the substituted context (the HotpotQA enumerator; paragraph contexts
only, spans of at most four words), lowercased, plus the
original answer as the memory candidate, capped at --max-candidates with the
gold always present and a seeded order. Fields: question, docs, candidates,
answer (substituted), answer_original, org_context."""
from __future__ import annotations
import argparse, collections, gzip, json, random, re, sys
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hotpot_frame import spans, whole_word  # noqa: E402

NUMERIC_OR_DATE = re.compile(r"^[\d,\.\s]+$|\b(1[0-9]{3}|20[0-2][0-9])\b|january|february|march|april|may|june|july|august|september|october|november|december|"
                             r"^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|[a-z]+teen|[a-z]+ty|hundred|thousand|million|billion)$", re.I)
LOCAL_STOP = {"about", "during", "while", "after", "before", "however", "although", "historically", "north", "south", "east", "west", "northern", "southern",
              "eastern", "western", "later", "today", "currently", "originally", "initially", "unlike", "like", "since", "though", "meanwhile", "instead", "thus", "then",
              "there", "here", "it", "its", "they", "their", "he", "she", "his", "her", "we", "you", "i", "a", "an", "the", "census", "population"}


def clean(ctx):
    return re.sub(r"\s+", " ", re.sub(r"</?(P|Td|Tr|Th|Table|Ul|Li|Dd|Dl|Dt|H\d)>", " ", ctx)).replace(" ,", ",").replace(" .", ".").replace(" 's", "'s").strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--max-doc-chars", type=int, default=6000)
    p.add_argument("--named-only", type=int, default=1, help="keep only rows whose original and substituted answers are named entities (no numbers, dates)")
    args = p.parse_args()
    dev = pq.read_table(hf_hub_download("pminervini/NQ-Swap", "data/dev-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    by_q = collections.defaultdict(list)
    for r in dev:
        by_q[r["question"]].append(r)
    eligible, stats = [], collections.Counter()
    for qi, (q, subs) in enumerate(sorted(by_q.items())):
        rng = random.Random(args.seed + qi); rng.shuffle(subs)
        for r in subs:
            org, sub = r["org_answer"][0].strip(), r["sub_answer"][0].strip()
            if not r["sub_context"].lstrip().startswith("<P>") or "<Table>" in r["sub_context"] or "<Tr>" in r["sub_context"]:
                stats["skip_non_paragraph"] += 1; continue
            docs = clean(r["sub_context"])
            if len(docs) > args.max_doc_chars:
                stats["skip_long"] += 1; continue
            if org.lower() == sub.lower() or not whole_word(sub, docs) or whole_word(org, docs) or len(sub) < 2:
                stats["skip_substitution"] += 1; continue
            if args.named_only and (NUMERIC_OR_DATE.search(sub) or NUMERIC_OR_DATE.search(org)):
                stats["skip_numeric_or_date_answer"] += 1; continue
            counts = collections.Counter()
            for sent in re.split(r"(?<=[\.\!\?;])\s+", docs):
                counts.update(spans(sent))
            for m in re.findall(r"(?<![\w\.])\d[\d,\.]*(?![\w])", docs):
                counts[m.strip(".,")] += 1
            cands = {}
            for c, n in counts.most_common():
                k = c.lower()
                if k in cands or k == sub.lower() or k == org.lower() or k in q.lower() or len(k) < 2 or not whole_word(c, docs):
                    continue
                if len(k.split()) > 4 or k.endswith("'s") or k in LOCAL_STOP or re.fullmatch(r"0+|[\d,\.]{1,1}", k):
                    continue
                cands[k] = k
                if len(cands) >= args.max_candidates - 2:
                    break
            if len(cands) < 3:
                stats["skip_few_candidates"] += 1; continue
            cand_list = [sub.lower(), org.lower()] + list(cands.values())
            random.Random(args.seed + 7 * qi).shuffle(cand_list)
            eligible.append({"id": f"NQS_{qi}", "ds": "nqswap", "docs": docs, "question": q[0].upper() + q[1:] + ("" if q.endswith("?") else "?"),
                             "relation": "", "subject": "", "query": q, "candidates": cand_list, "answer": sub.lower(), "answer_original": org.lower(),
                             "answer_original_cased": org, "answer_cased": sub, "org_context": clean(r["org_context"])})
            stats["eligible"] += 1
            break
        else:
            stats["skip_question_no_valid_sub"] += 1
    draw = sorted(random.Random(args.seed).sample(range(len(eligible)), min(args.n, len(eligible))))
    with gzip.open(args.out, "wt") as f:
        for j in draw:
            f.write(json.dumps(eligible[j], ensure_ascii=False) + "\n")
    ncand = [len(eligible[j]["candidates"]) for j in draw]
    kind = collections.Counter("numeric" if re.fullmatch(r"[\d,\.\s]+", eligible[j]["answer"]) else "date" if re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b|january|february|march|april|may|june|july|august|september|october|november|december", eligible[j]["answer"]) else "named" for j in draw)
    print(json.dumps({"stats": dict(stats), "questions": len(by_q), "drawn": len(draw), "candidates_mean": sum(ncand) / len(ncand), "candidates_min": min(ncand),
                      "doc_chars_mean": sum(len(eligible[j]["docs"]) for j in draw) / len(draw), "answer_kinds": dict(kind)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
