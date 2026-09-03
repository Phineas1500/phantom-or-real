import asyncio, json, sys, random
from collections import Counter
from pathlib import Path
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from wikihop_judge_relations import judge, FRAMES, frame, LABELS
from wikihop_loop_descriptives import RS, load, rows_from
from openai import AsyncOpenAI
from src.env_loader import get_openai_gpt_credentials, load_env
SETS = {"WX real text (59, frozen)": ["wikihop_wx_a.jsonl", "wikihop_wx_b.jsonl"], "WA rider anonymized (60, frozen)": ["wikihop_wa_frozen_a.jsonl", "wikihop_wa_frozen_b.jsonl"],
        "WO anonymized (47, frozen)": ["wikihop_wo_a.jsonl", "wikihop_wo_b.jsonl"], "WD unrepairable (100, frozen)": ["wikihop_wd_y1.jsonl", "wikihop_wd_y2.jsonl"]}
async def main():
    load_env(); base_url, key = get_openai_gpt_credentials()
    client = AsyncOpenAI(base_url=base_url, api_key=key); sem = asyncio.Semaphore(16)
    pool = []
    for name, files in SETS.items():
        rows = rows_from(load([RS / f for f in files])); fr = frame(RS / FRAMES[name])
        for i, row in rows.items():
            f = fr[i]; rev = {v.lower(): k for k, v in (f.get("anon_map") or {}).items()}
            gold = [c for c in row["cands"] if row["cands"][c]["gold"]]
            if not gold: continue
            for c in row["cands"]:
                if c != gold[0]:
                    pool.append({"setting": name, "id": i, "relation": f["relation"], "subject": rev.get(f["subject"].lower(), f["subject"]),
                                 "gold": rev.get(f["answer"], f["answer"]), "selected": rev.get(c, c)})
    random.Random(20260849).shuffle(pool); sample = pool[:200]
    labels = await asyncio.gather(*[judge(client, "gpt-5.4-mini", sem, it) for it in sample])
    for it, lab in zip(sample, labels): it["relation_label"] = lab
    Path("docs/wikihop_nongold_relation_base_rate.json").write_text(json.dumps({"n_pool": len(pool), "sample": sample}, indent=1) + "\n")
    c = Counter(labels); n = len(labels)
    print(f"non-gold candidate pool {len(pool)}; judged sample {n}: " + ", ".join(f"{l} {c[l]} ({c[l]/n:.0%})" for l in LABELS) + f"; other {n - sum(c[l] for l in LABELS)}")
asyncio.run(main())
