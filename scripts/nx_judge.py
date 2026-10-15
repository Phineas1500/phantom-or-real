#!/usr/bin/env python3
"""Item NX stage 4 — the explanation protocol (registered: item NX).
Three SEPARATE calls per neuron, none seeing another's inputs beyond what is
listed: (1) explain: the top-20 contexts (firing token in [[ ]]) -> one
sentence; (2) test: only that sentence -> 10 positives, 10 near misses
(same surface words or topic, property absent), 10 distant texts; (3) bin:
only the sentence -> one of the pre-named bins.
Usage: nx_judge.py --mined nx_mined.json --out docs/nx_explanations.json
       [--tests-out nx_tests.json]   (the score-job input)"""
from __future__ import annotations
import argparse, asyncio, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai import AsyncOpenAI  # noqa: E402
from src.env_loader import get_openai_gpt_credentials, load_env  # noqa: E402

BINS = ["entity / copy", "document reference", "question relation", "answer format", "other"]

EXPLAIN = """Below are the 20 text snippets on which one neuron of a language model is most active. In each snippet the token where it is most active is marked with [[ ]].

{contexts}

In ONE sentence, say what this neuron responds to. Be specific enough that someone could write new text that makes it fire, and new text that is similar on the surface but does not. Reply with the sentence only."""

TEST = """A neuron in a language model is described as follows:

"{explanation}"

Write 30 short texts (one to three sentences each) to test this description:
- 10 POSITIVE texts that, according to the description, should make the neuron fire;
- 10 NEAR-MISS texts that share surface features with the positives (same words, topic or style) but lack the property the description names, so it should NOT fire;
- 10 DISTANT texts on unrelated subjects.
Reply with JSON only: {{"positive": [...], "near_miss": [...], "distant": [...]}}"""

BIN = """A neuron in a language model is described as follows:

"{explanation}"

Which ONE category fits the description best?
- entity / copy: tracks a specific entity, name or span, or copying a token from context
- document reference: refers to documents, passages, sources or where something is stated
- question relation: the relation or attribute being asked about (e.g. birthplace, publisher, country)
- answer format: output formatting, answer phrasing, list or candidate structure
- other: anything else
Reply with the category name only."""


async def ask(client, sem, model, prompt, max_tokens=4000):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                         max_completion_tokens=max_tokens)
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt + 1)
                print(f"retry after {type(e).__name__}", flush=True)


def parse_tests(text):
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0)) if m else {}
    return {k: [str(x) for x in d.get(k, [])][:10] for k in ("positive", "near_miss", "distant")}


async def one(client, sem, model, n):
    ctx = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(n["top_contexts"]))
    expl = (await ask(client, sem, model, EXPLAIN.format(contexts=ctx))).strip().split("\n")[0].strip()
    tests = parse_tests(await ask(client, sem, model, TEST.format(explanation=expl), max_tokens=8000))
    b = (await ask(client, sem, model, BIN.format(explanation=expl), max_tokens=1000)).strip().lower()
    b = next((x for x in BINS if x.split(" /")[0] in b), "other")
    return {"set": n["set"], "layer": n["layer"], "neuron": n["neuron"], "pole": n["pole"], "fire_scale": n["fire_scale"],
            "explanation": expl, "bin": b, "tests": tests}


async def run(args):
    load_env()
    base_url, key = get_openai_gpt_credentials()
    client = AsyncOpenAI(base_url=base_url, api_key=key)
    sem = asyncio.Semaphore(args.concurrency)
    mined = json.loads(args.mined.read_text())["neurons"]
    out = await asyncio.gather(*[one(client, sem, args.model, n) for n in mined])
    await client.close()
    args.out.write_text(json.dumps({"model": args.model, "neurons": out}, indent=1) + "\n")
    if args.tests_out:
        args.tests_out.write_text(json.dumps({"tests": [{"key": f"{n['set']}:{n['layer']}:{n['neuron']}", "texts": n["tests"]} for n in out]}) + "\n")
    short = sum(min(len(v) for v in n["tests"].values()) < 10 for n in out)
    print(f"{len(out)} neurons explained; {short} with fewer than 10 texts in some kind")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mined", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("docs/nx_explanations.json"))
    p.add_argument("--tests-out", type=Path)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=16)
    asyncio.run(run(p.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
