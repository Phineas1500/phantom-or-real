"""Qwen3.8-27B gauge hunt (givemeanode H100 batch job entrypoint).

Registration-free reconnaissance, step past the day-one smoke: does the
correctness gauge exist in the DeltaNet/attention hybrid, and at which depths?
Three phases in one job:
  1. labels  — unhinted k=1 at temp 0.7 on 2,000 seeded stage-1 property rows
               (h3/h4, seed 20260815), the row-level correctness labels for
               probe training (scored locally with the house parser);
  2. hinted  — hint_concept_first k=4 on the first 500 rows, the G0-scale
               behavioral gate (hint lift with real CIs);
  3. capture — one forward pass per unhinted prompt with output_hidden_states,
               saving the final-prompt-token vector at every layer (fp16,
               [n_layers+1, n_rows, hidden]) for the local AUC-by-depth ladder.
Plain transformers bf16 (pinned vLLM/TL stacks predate the hybrid class);
thinking disabled throughout, matching smoke v2.
"""
import gzip
import json
import os
import time


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("GAUGE_MODEL", "Qwen/Qwen3.8-27B")
    rows = [json.loads(l) for l in gzip.open("/app/gauge_input.jsonl.gz", "rt")]
    hinted_rows = [r for r in rows if "hinted_prompt" in r]
    print(f"loaded {len(rows)} rows ({len(hinted_rows)} hinted); model={model_id}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    t_load = time.time() - t0
    print(f"model loaded in {t_load:.0f}s; class={model.__class__.__name__}", flush=True)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    max_new = int(os.environ.get("GAUGE_MAX_NEW", "640"))

    def build(row, arm):
        user = row["prompt_text"] if arm == "unhinted" else row["hinted_prompt"]
        msgs = [{"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": user}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            return tok.apply_chat_template(msgs, **kwargs)

    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    counts = {}

    def generate(subset, arm, k, batch, seed_off, fname):
        n = 0
        t1 = time.time()
        with open(os.path.join(outdir, fname), "w") as fout:
            for i in range(0, len(subset), batch):
                chunk = subset[i:i + batch]
                prompts = [build(r, arm) for r in chunk]
                enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
                torch.manual_seed(20260815 + i * 10007 + seed_off)
                with torch.inference_mode():
                    gen = model.generate(
                        **enc, do_sample=True, temperature=0.7, num_return_sequences=k,
                        max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                    )
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for j, r in enumerate(chunk):
                    for s in range(k):
                        fout.write(json.dumps({
                            "row_index": r["row_index"], "height": r["height"],
                            "arm": arm, "sample_index": s,
                            "model_output": texts[j * k + s],
                        }, ensure_ascii=False) + "\n")
                        n += 1
                fout.flush()
                if (i // batch) % 10 == 0:
                    print(f"{fname} {i + len(chunk)}/{len(subset)} rows, {time.time()-t1:.0f}s", flush=True)
        counts[fname] = n
        print(f"{fname} done: {n} gens in {time.time()-t1:.0f}s", flush=True)

    generate(rows, "unhinted", 1, 16, 0, "gauge_labels.jsonl")
    generate(hinted_rows, "hinted", 4, 8, 101, "gauge_hinted.jsonl")

    t2 = time.time()
    cap_batch = 4
    caps, lens = [], []
    for i in range(0, len(rows), cap_batch):
        chunk = rows[i:i + cap_batch]
        prompts = [build(r, "unhinted") for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        caps.append(torch.stack([h[:, -1, :] for h in out.hidden_states])
                    .to(torch.float16).cpu().numpy())
        lens.extend(int(m.sum()) for m in enc["attention_mask"])
        del out
        if (i // cap_batch) % 50 == 0:
            print(f"capture {i + len(chunk)}/{len(rows)}, {time.time()-t2:.0f}s", flush=True)
    hidden = np.concatenate(caps, axis=1)
    np.save(os.path.join(outdir, "gauge_hidden_final.npy"), hidden)
    with open(os.path.join(outdir, "gauge_capture_meta.json"), "w") as f:
        json.dump({"shape": list(hidden.shape),
                   "order": [r["row_index"] for r in rows],
                   "prompt_token_lens": lens,
                   "position": "final_prompt_token", "dtype": "float16"}, f)
    print(f"capture done: {hidden.shape} in {time.time()-t2:.0f}s", flush=True)

    summary = {"model": model_id, "model_class": model.__class__.__name__,
               "n_rows": len(rows), "counts": counts,
               "capture_shape": list(hidden.shape),
               "load_seconds": round(t_load), "total_seconds": round(time.time() - t0)}
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
