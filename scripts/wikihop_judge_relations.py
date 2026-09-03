#!/usr/bin/env python3
"""Descriptive: label the relation between the loop's wrong pick and the
gold answer with a GPT judge (PARENT / CHILD / ALIAS / SIBLING / UNRELATED),
for every wrong selection listed in docs/wikihop_loop_descriptives.json.
Credentials come from src.env_loader (OPENAI_API_KEY_GPT), never printed."""
from __future__ import annotations
import argparse, asyncio, gzip, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai import AsyncOpenAI  # noqa: E402
from src.env_loader import get_openai_gpt_credentials, load_env  # noqa: E402

PROMPT = """A reading-comprehension question asks for the "{relation}" of "{subject}".
The correct answer is: {gold}
A model answered instead: {selected}

Classify the model's answer relative to the correct answer with exactly one label:
PARENT — a broader entity that contains or subsumes the correct answer (a country for a city or region, a higher taxon for a lower one, a parent company or league for a member, a franchise for one entry, a continent for a country).
CHILD — narrower: a part, subtype, member, or sub-entity of the correct answer.
ALIAS — the same entity under another name, spelling, or near-synonym.
SIBLING — the same kind of entity at the same level, but a different one (another city, another band, another year, another person).
UNRELATED — none of the above.

Reply with the label only."""
LABELS = ["PARENT", "CHILD", "ALIAS", "SIBLING", "UNRELATED"]
FRAMES = {"WX real text (59, frozen)": "wikihop_fresh_input.jsonl.gz", "WA rider anonymized (60, frozen)": "wikihop_anon_input.jsonl.gz",
          "WO anonymized (47, frozen)": "wikihop_anon2_input.jsonl.gz", "WD unrepairable (100, frozen)": "wikihop_anon2_input.jsonl.gz"}


def frame(path):
    return {r["id"]: r for r in (json.loads(l) for l in gzip.open(path, "rt"))}


async def judge(client, model, sem, item):
    async with sem:
        for _ in range(3):
            try:
                c = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": PROMPT.format(**item)}], max_completion_tokens=256)
                text = (c.choices[0].message.content or "").strip().upper()
                for lab in LABELS:
                    if lab in text:
                        return lab
                return "UNPARSED"
            except Exception:
                await asyncio.sleep(2)
        return "ERROR"


async def run(args):
    load_env()
    base_url, key = get_openai_gpt_credentials()
    client = AsyncOpenAI(base_url=base_url, api_key=key); sem = asyncio.Semaphore(args.concurrency)
    d = json.load(open(args.descriptives))["selector_failures"]
    items = []
    for name, b in d.items():
        fr = frame(Path("results/loop_screen") / FRAMES[name])
        for w in b["wrong_selections_all"]:
            f = fr[w["id"]]; rev = {v.lower(): k for k, v in (f.get("anon_map") or {}).items()}
            subj = rev.get(f["subject"].lower(), f["subject"])
            items.append({"setting": name, "id": w["id"], "relation": f["relation"], "subject": subj, "gold": w["gold"], "selected": w["selected"],
                          "string_kind": w["kind"], "af_sel": w["af_sel"], "af_gold": w["af_gold"]})
    labels = await asyncio.gather(*[judge(client, args.model, sem, it) for it in items])
    for it, lab in zip(items, labels):
        it["relation_label"] = lab
    args.out.write_text(json.dumps(items, indent=1) + "\n")
    print(f"judged {len(items)} wrong selections with {args.model}")
    print("| setting | n | PARENT | CHILD | ALIAS | SIBLING | UNRELATED | other |")
    print("|---|---|---|---|---|---|---|---|")
    for name in d:
        xs = [it["relation_label"] for it in items if it["setting"] == name]; c = Counter(xs); n = len(xs)
        other = n - sum(c[l] for l in LABELS)
        print(f"| {name} | {n} | " + " | ".join(f"{c[l]} ({c[l]/n:.0%})" for l in LABELS) + f" | {other} |")
    allc = Counter(it["relation_label"] for it in items); n = len(items)
    print("| all | " + str(n) + " | " + " | ".join(f"{allc[l]} ({allc[l]/n:.0%})" for l in LABELS) + f" | {n - sum(allc[l] for l in LABELS)} |")
    tied = [it for it in items if it["af_sel"] == it["af_gold"] and it["af_sel"] > 0]
    tc = Counter(it["relation_label"] for it in tied)
    print(f"tied-at-top wrong picks ({len(tied)}): " + ", ".join(f"{l} {tc[l]} ({tc[l]/len(tied):.0%})" for l in LABELS))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--descriptives", type=Path, default=Path("docs/wikihop_loop_descriptives.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wrong_selection_relations.json"))
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=16)
    asyncio.run(run(p.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
