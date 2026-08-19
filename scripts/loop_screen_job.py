"""Loop-port screening (registration-free reconnaissance) — givemeanode H100
batch entrypoint. Gemma-3-27B behavioral gates on WikiHop + RACE-high
(300 seeded rows each): four arms x k=8 at temp 0.7 —
  std    : documents + question + candidate list (the dataset's task)
  free   : documents + question, no candidates (recognition-gap analog)
  hint   : std + a hint naming the gold candidate (G0 hint-repair analog)
  closed : question + candidates, NO documents (contamination sniff)
Gates computed locally: baseline band, self-consistency flatness on
failures, hint gap, closed-book accuracy."""
import gzip
import json
import os
from collections import defaultdict

SYSTEM = "You are a careful reading assistant. Answer concisely."


def prompts_for(r):
    docs = r["docs"]
    if r["ds"] == "wikihop":
        q = f"Based on the documents, what is the '{r['relation']}' of {r['subject']}?"
        cands = "\n".join(f"- {c}" for c in r["candidates"])
        return {
            "std": f"Documents:\n{docs}\n\nQuestion: {q}\nCandidates:\n{cands}\n\nAnswer with exactly one candidate, nothing else.",
            "free": f"Documents:\n{docs}\n\nQuestion: {q}\n\nAnswer with a single entity name, nothing else.",
            "hint": f"Documents:\n{docs}\n\nQuestion: {q}\nCandidates:\n{cands}\n\nHint: pay close attention to {r['answer']}.\nAnswer with exactly one candidate, nothing else.",
            "closed": f"Question: {q}\nCandidates:\n{cands}\n\nAnswer with exactly one candidate, nothing else.",
        }
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(r["candidates"]))
    q = r["question"]
    gold_text = r["candidates"][letters.index(r["answer"])]
    return {
        "std": f"Article:\n{docs}\n\nQuestion: {q}\nOptions:\n{opts}\n\nAnswer with the letter (A/B/C/D) only.",
        "free": f"Article:\n{docs}\n\nQuestion: {q}\n\nAnswer in one short phrase.",
        "hint": f"Article:\n{docs}\n\nQuestion: {q}\nOptions:\n{opts}\n\nHint: consider \"{gold_text}\".\nAnswer with the letter (A/B/C/D) only.",
        "closed": f"Question: {q}\nOptions:\n{opts}\n\nAnswer with the letter (A/B/C/D) only.",
    }


def main():
    rows = [json.loads(l) for l in gzip.open("/app/screen_input.jsonl.gz", "rt")]
    print(f"loaded {len(rows)} rows", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-3-27b-it", max_model_len=8192, gpu_memory_utilization=0.92)
    arms = ["std", "free", "hint", "closed"]
    convs, meta = [], []
    for r in rows:
        ps = prompts_for(r)
        for arm in arms:
            convs.append([{"role": "user", "content": SYSTEM + "\n\n" + ps[arm]}])
            meta.append((r["id"], r["ds"], arm))
    sp = SamplingParams(n=8, temperature=0.7, max_tokens=32, seed=20260819)
    outs = llm.chat(convs, sp)
    print("generation done", flush=True)
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    n = 0
    with open(os.path.join(outdir, "loop_screen.jsonl"), "w") as f:
        for (rid, ds, arm), o in zip(meta, outs):
            for s, c in enumerate(o.outputs):
                f.write(json.dumps({"id": rid, "ds": ds, "arm": arm,
                                    "sample_index": s, "model_output": c.text},
                                   ensure_ascii=False) + "\n")
                n += 1
    rp = os.environ.get("GMN_RESULT_PATH")
    summary = {"n_generations": n, "n_rows": len(rows), "arms": arms}
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
