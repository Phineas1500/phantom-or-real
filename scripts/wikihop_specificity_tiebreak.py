#!/usr/bin/env python3
"""Descriptive (post-hoc, unregistered): a blind specificity tie-break for
the output-first selector. Among branches tied at the top answers-fired
rate, drop any candidate that a GPT judge calls a PARENT (broader,
containing entity) of another tied candidate; pick the unique survivor,
else fall back to the gauge tie-break among survivors. Judged pairs are
cached in docs/wikihop_tied_pair_relations.json. Reports the paired change
vs the plain output-first pick on every frozen-write setting on disk,
including the correct-row (collateral) sets."""
from __future__ import annotations
import argparse, asyncio, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wikihop_w1_gates import boot_ci  # noqa: E402
from wikihop_loop_descriptives import RS, load, rows_from, select, frame  # noqa: E402
from wikihop_judge_relations import judge, LABELS  # noqa: E402

SETTINGS = [
    ("WX real (59)", ["wikihop_wx_a.jsonl", "wikihop_wx_b.jsonl"], "wikihop_fresh_input.jsonl.gz", "primary_L38", "failures"),
    ("WA rider anon (60)", ["wikihop_wa_frozen_a.jsonl", "wikihop_wa_frozen_b.jsonl"], "wikihop_anon_input.jsonl.gz", "second_L38", "failures"),
    ("WO anon (47)", ["wikihop_wo_a.jsonl", "wikihop_wo_b.jsonl"], "wikihop_anon2_input.jsonl.gz", "second_L38", "failures"),
    ("WD unrepairable anon (100)", ["wikihop_wd_y1.jsonl", "wikihop_wd_y2.jsonl"], "wikihop_anon2_input.jsonl.gz", "second_L38", "failures"),
    ("WE unrepairable real (100)", ["wikihop_we_y1.jsonl", "wikihop_we_y2.jsonl"], "wikihop_fresh_input.jsonl.gz", "primary_L38", "failures"),
    ("WD correct anon (60)", ["wikihop_wd_c.jsonl"], "wikihop_anon2_input.jsonl.gz", "second_L38", "correct"),
    ("WE correct real (60)", ["wikihop_we_c.jsonl"], "wikihop_fresh_input.jsonl.gz", "primary_L38", "correct"),
]


def tied_set(row):
    c = row["cands"]; top = max(v["af"] for v in c.values())
    return ([x for x in c if c[x]["af"] == top] if top > 0 else []), top


async def main_async(args):
    cache = json.load(open(args.cache)) if args.cache.exists() else {}
    data = {}
    todo = {}
    for name, files, fpath, tie_key, kind in SETTINGS:
        rows = rows_from(load([RS / f for f in files]), tie_key=tie_key); fr = frame(RS / fpath)
        data[name] = (rows, fr, kind)
        for i, row in rows.items():
            tied, top = tied_set(row)
            if len(tied) < 2:
                continue
            f = fr[i]; rev = {v.lower(): k for k, v in (f.get("anon_map") or {}).items()}
            for a, b in combinations(sorted(tied), 2):
                key = f"{f['relation']}||{rev.get(a, a)}||{rev.get(b, b)}"
                if key not in cache:
                    todo[key] = {"relation": f["relation"], "subject": rev.get(f["subject"].lower(), f["subject"]), "gold": rev.get(b, b), "selected": rev.get(a, a)}
    if todo and not args.no_judge:
        from openai import AsyncOpenAI
        from src.env_loader import get_openai_gpt_credentials, load_env
        load_env(); base_url, key = get_openai_gpt_credentials()
        client = AsyncOpenAI(base_url=base_url, api_key=key); sem = asyncio.Semaphore(16)
        keys = list(todo)
        labels = await asyncio.gather(*[judge(client, args.model, sem, todo[k]) for k in keys])
        for k, lab in zip(keys, labels):
            cache[k] = lab
        args.cache.write_text(json.dumps(cache, indent=1) + "\n")
        print(f"judged {len(keys)} new tied pairs (a relative to b: PARENT means a contains b)")
    def parents(f, tied):
        rev = {v.lower(): k for k, v in (f.get("anon_map") or {}).items()}
        drop = set()
        for a, b in combinations(sorted(tied), 2):
            lab = cache.get(f"{f['relation']}||{rev.get(a, a)}||{rev.get(b, b)}")
            if lab == "PARENT":
                drop.add(a)
            elif lab == "CHILD":
                drop.add(b)
        return drop
    print("| setting | rows | tied rows | rows changed | plain output-first | specificity tie-break | paired Δ [95% CI] |")
    print("|---|---|---|---|---|---|---|")
    out = {}
    for name, (rows, fr, kind) in data.items():
        plain, spec, n_tied, n_changed = [], [], 0, 0
        for i, row in rows.items():
            c = row["cands"]; sel = select(c); tied, top = tied_set(row)
            pick = sel
            if len(tied) >= 2:
                n_tied += 1
                keep = [x for x in tied if x not in parents(fr[i], tied)] or tied
                pick = keep[0] if len(keep) == 1 else max(keep, key=lambda x: c[x]["gauge"])
            n_changed += int(pick != sel)
            plain.append(c[sel]["corr"]); spec.append(c[pick]["corr"])
        d = np.array(spec) - np.array(plain); ci = boot_ci(list(d))
        out[name] = {"n": len(rows), "tied_rows": n_tied, "changed": n_changed, "plain": float(np.mean(plain)), "specificity": float(np.mean(spec)), "paired": float(d.mean()), "ci": ci, "kind": kind}
        print(f"| {name} | {len(rows)} | {n_tied} | {n_changed} | {np.mean(plain):.3f} | {np.mean(spec):.3f} | {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    args.out.write_text(json.dumps(out, indent=1) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("docs/wikihop_tied_pair_relations.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_specificity_tiebreak.json"))
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--no-judge", action="store_true")
    asyncio.run(main_async(p.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
