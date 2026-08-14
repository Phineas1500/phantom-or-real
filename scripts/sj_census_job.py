"""SJ census, offline-batch form (registration-free reconnaissance) — the
givemeanode job entrypoint. Same protocol as scripts/sj_census.py: frozen SJ
question, k=4 at temp 0.7, majority vote; OC x SJ census. Runs entirely
in-job on one H100. Input: /app/sj_census_input.jsonl.gz (built locally by
scripts/sj_census.py's select_rows at seed 20260812)."""
import gzip
import json
import os
from collections import Counter

SJ_QUESTION = os.environ.get(
    "SJ_QUESTION",
    "Look back at the answer you just gave. Was your final proposed hypothesis "
    "exactly correct? Reply with a single word: yes or no.",
)


def parse_sj(text):
    word = text.strip().lower().strip(".,!\"' \n")
    if word.startswith("yes"):
        return "yes"
    if word.startswith("no"):
        return "no"
    return None


def main():
    rows = [json.loads(l) for l in gzip.open("/app/sj_census_input.jsonl.gz", "rt")]
    print(f"loaded {len(rows)} rows", flush=True)

    convs = [[
        {"role": "user", "content": r["system_prompt"] + "\n\n" + r["prompt_text"]},
        {"role": "assistant", "content": r["model_output"]},
        {"role": "user", "content": SJ_QUESTION},
    ] for r in rows]

    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-3-27b-it", max_model_len=8192, gpu_memory_utilization=0.92)
    sp = SamplingParams(n=4, temperature=0.7, max_tokens=8, seed=20260812)
    outs = llm.chat(convs, sp)
    print("generation done", flush=True)

    out_rows = []
    for r, o in zip(rows, outs):
        samples = [c.text for c in o.outputs]
        parsed = [parse_sj(s) for s in samples]
        votes = Counter(p for p in parsed if p)
        sj = "yes" if votes.get("yes", 0) > votes.get("no", 0) else ("no" if votes.get("no", 0) > votes.get("yes", 0) else "tie")
        out_rows.append({
            "source_row_index": r["row_index"], "height": r["height"],
            "is_correct_strong": bool(r["is_correct_strong"]),
            "sj_majority": sj, "sj_samples": samples, "sj_parsed": parsed,
            "n_parsed": sum(1 for p in parsed if p),
            "unanimous": len({p for p in parsed if p}) == 1 and any(parsed),
        })

    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    with open(os.path.join(outdir, "sj_census_2k.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cells = Counter((r["is_correct_strong"], r["sj_majority"]) for r in out_rows)
    n = len(out_rows)
    summary = {
        "n_rows": n, "samples_per_row": 4, "temperature": 0.7, "seed": 20260812,
        "sj_question": SJ_QUESTION, "protocol": "offline_vllm_batch_h100",
        "census": {f"oc_{oc}__sj_{sj}": cells.get((oc, sj), 0) for oc in (True, False) for sj in ("yes", "no", "tie")},
        "conflict_cells": {
            "confident_wrong_oc_false_sj_yes": cells.get((False, "yes"), 0),
            "unconfident_right_oc_true_sj_no": cells.get((True, "no"), 0),
        },
        "sj_yes_rate_overall": sum(1 for r in out_rows if r["sj_majority"] == "yes") / n,
        "unanimous_rate": sum(1 for r in out_rows if r["unanimous"]) / n,
        "parse_ok_rate": sum(r["n_parsed"] for r in out_rows) / (n * 4),
        "by_height": {str(h): {f"oc_{oc}__sj_{sj}": c for (oc, sj), c in
                      Counter((r["is_correct_strong"], r["sj_majority"]) for r in out_rows if r["height"] == h).items()}
                      for h in (3, 4)},
    }
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary["census"], indent=1), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
