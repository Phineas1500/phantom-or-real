#!/usr/bin/env python3
"""Item WM frame: a haystack of tables. Each WikiTableQuestions example whose
answer is a unique table cell becomes a document made of its own table plus
seeded distractor tables from other examples (up to --max-doc-chars), tables
shuffled, rendered as markdown; the question is prefixed with the target
table's column names so the target is identifiable. Per-row character spans
are recorded for addressing, and a label-free BM25 retriever over rows
(header + cells) ranks every row against the question (top-5 kept, gold
rank recorded). Candidates are empty: the std prompt is free-form."""
from __future__ import annotations
import argparse, collections, csv, gzip, json, math, random, re
from pathlib import Path


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .")


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def load_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows = [[re.sub(r"\s+", " ", c).strip() for c in r] for r in rows if r]
    return rows[0], rows[1:]


def render(j, header, rows):
    lines = [f"### Table {j} (columns: {', '.join(header)})", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    spans = []
    text = "\n".join(lines) + "\n"
    for ri, r in enumerate(rows):
        line = "| " + " | ".join(r) + " |"
        spans.append((ri, len(text), len(text) + len(line)))
        text += line + "\n"
    return text, spans


def bm25(query, docs, k1=1.5, b=0.75):
    N = len(docs); avg = sum(len(d) for d in docs) / max(N, 1)
    df = collections.Counter(t for d in docs for t in set(d))
    out = []
    for d in docs:
        tf = collections.Counter(d); s = 0.0
        for t in query:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wtq", type=Path, required=True, help="WikiTableQuestions directory (compact release)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--max-doc-chars", type=int, default=18000)
    p.add_argument("--max-tables", type=int, default=24)
    args = p.parse_args()
    ex = []
    for split in ("training.tsv", "pristine-unseen-tables.tsv"):
        with open(args.wtq / "data" / split, encoding="utf-8") as f:
            rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            ex += list(rd)
    tables = {}
    stats = collections.Counter(); elig = []
    for e in ex:
        ctx = e["context"]
        if ctx not in tables:
            try:
                tables[ctx] = load_table(args.wtq / ctx)
            except Exception:
                tables[ctx] = None
        if tables[ctx] is None:
            stats["skip_table_load"] += 1; continue
        header, rows = tables[ctx]
        if "|" in e["targetValue"]:
            stats["skip_multi_answer"] += 1; continue
        ans = e["targetValue"].strip()
        if not ans or len(ans.split()) > 6 or len(rows) < 4 or len(rows) > 40 or len(header) < 2 or len(header) > 8 or any(len(r) != len(header) for r in rows):
            stats["skip_shape"] += 1; continue
        hits = [(ri, ci) for ri, r in enumerate(rows) for ci, c in enumerate(r) if norm(c) == norm(ans)]
        if len(hits) != 1:
            stats["skip_not_unique_cell"] += 1; continue
        if norm(ans) in norm(e["utterance"]) or len(norm(ans)) < 2:
            stats["skip_answer_in_question"] += 1; continue
        elig.append((e, ctx, hits[0][0]))
        stats["eligible"] += 1
    rng = random.Random(args.seed); rng.shuffle(elig)
    all_ctx = sorted({c for c in tables if tables[c] is not None})
    out_rows = []
    for k, (e, ctx, gold_row) in enumerate(elig):
        if len(out_rows) >= args.n:
            break
        header, rows = tables[ctx]; r = random.Random(args.seed + k)
        chosen = [ctx]; total = len(render(0, header, rows)[0])
        pool = [c for c in all_ctx if c != ctx]; r.shuffle(pool)
        for c in pool:
            h2, r2 = tables[c]
            if len(r2) > 40 or len(h2) > 8 or any(len(x) != len(h2) for x in r2):
                continue
            if any(norm(cell) == norm(e["targetValue"]) for row in r2 for cell in row):
                continue
            t = len(render(0, h2, r2)[0])
            if total + t > args.max_doc_chars or len(chosen) >= args.max_tables:
                break
            chosen.append(c); total += t
        if len(chosen) < 6:
            stats["skip_few_distractors"] += 1; continue
        r.shuffle(chosen)
        docs, spans, gold_t = "", [], None
        for j, c in enumerate(chosen):
            h, rr = tables[c]
            text, sp = render(j, h, rr)
            for ri, a, b in sp:
                spans.append({"t": j, "r": ri, "start": len(docs) + a, "end": len(docs) + b})
            if c == ctx:
                gold_t = j
            docs += text + "\n"
        q = f"In the table whose columns are {', '.join(header)}: {e['utterance'].strip()}"
        row_docs = []
        for s in spans:
            h, rr = tables[chosen[s["t"]]]
            row_docs.append(toks(" ".join(h) + " " + " ".join(rr[s["r"]])))
        scores = bm25(toks(e["utterance"]), row_docs)
        order = sorted(range(len(spans)), key=lambda i: -scores[i])
        gold_idx = next(i for i, s in enumerate(spans) if s["t"] == gold_t and s["r"] == gold_row)
        table_docs = [toks(" ".join(tables[c][0]) * 3 + " " + " ".join(" ".join(x) for x in tables[c][1])) for c in chosen]
        tscores = bm25(toks(q), table_docs); torder = sorted(range(len(chosen)), key=lambda i: -tscores[i])
        top_t = torder[0]
        in_top = [i for i, sp in enumerate(spans) if sp["t"] == top_t]
        in_top_order = sorted(in_top, key=lambda i: -scores[i])
        out_rows.append({"id": f"WTQ_{e['id']}", "ds": "wtq_haystack", "docs": docs.strip(), "question": q, "candidates": [], "answer": norm(e["targetValue"]),
                         "answer_cased": e["targetValue"].strip(), "relation": "", "subject": "", "query": q, "gold_table": gold_t, "gold_row": gold_row,
                         "gold_span_index": gold_idx, "spans": spans, "retrieved": order[:5], "gold_rank": order.index(gold_idx) + 1, "n_tables": len(chosen),
                         "retrieved_tables": torder[:3], "gold_table_rank": torder.index(gold_t) + 1, "retrieved_rows_in_top_table": in_top_order[:3],
                         "gold_row_rank_in_top_table": (in_top_order.index(gold_idx) + 1) if gold_idx in in_top_order else None,
                         "n_rows_total": len(spans), "doc_chars": len(docs), "source_table": ctx})
    with gzip.open(args.out, "wt") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ranks = [x["gold_rank"] for x in out_rows]; tr = [x["gold_table_rank"] for x in out_rows]; rr = [x["gold_row_rank_in_top_table"] for x in out_rows]
    print(json.dumps({"stats": dict(stats), "written": len(out_rows), "doc_chars_mean": sum(x["doc_chars"] for x in out_rows) / len(out_rows), "tables_mean": sum(x["n_tables"] for x in out_rows) / len(out_rows),
                      "rows_mean": sum(x["n_rows_total"] for x in out_rows) / len(out_rows), "bm25_gold_rank_le": {k: sum(r <= k for r in ranks) / len(ranks) for k in (1, 3, 5, 10)},
                      "table_bm25_gold_rank_le1": sum(r == 1 for r in tr) / len(tr), "row_in_top_table_gold_rank_le": {k: sum(1 for r in rr if r is not None and r <= k) / len(rr) for k in (1, 3, 5)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
