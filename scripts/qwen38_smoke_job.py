"""Qwen3.8-27B day-one smoke test (givemeanode H100 batch job entrypoint).

Pure generation: unhinted + concept-hinted arms on 100 stage-1 property rows
(h3/h4, seed 20260814), k=4 at temp 0.7, plain transformers (the hybrid
DeltaNet class is unsupported by our pinned vLLM/TransformerLens stacks).
Raw generations ship back as the artifact; scoring happens locally with the
house parser. Reconnaissance only — no registration, no lane implications
(Qwen3.8's native lane, if the bridge proceeds, is this one).
"""
import gzip
import json
import os
import time


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3.8-27B")
    rows = [json.loads(l) for l in gzip.open("/app/smoke_input.jsonl.gz", "rt")]
    print(f"loaded {len(rows)} rows; model={model_id}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    print(f"model loaded in {time.time()-t0:.0f}s; class={model.__class__.__name__}", flush=True)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    no_think = os.environ.get("SMOKE_NO_THINK", "1") == "1"
    max_new = int(os.environ.get("SMOKE_MAX_NEW", "640"))

    def build(row, arm):
        user = row["prompt_text"] if arm == "unhinted" else row["hinted_prompt"]
        msgs = [{"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": user}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if no_think:
            try:
                return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
            except TypeError:
                pass
        return tok.apply_chat_template(msgs, **kwargs)

    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    n_out = 0
    t1 = time.time()
    with open(os.path.join(outdir, "qwen38_smoke.jsonl"), "w") as fout:
        for arm in ("unhinted", "hinted"):
            for i in range(0, len(rows), 8):
                chunk = rows[i:i + 8]
                prompts = [build(r, arm) for r in chunk]
                enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
                torch.manual_seed(20260814 + i * 10007 + (0 if arm == "unhinted" else 101))
                with torch.inference_mode():
                    gen = model.generate(
                        **enc, do_sample=True, temperature=0.7, num_return_sequences=4,
                        max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                    )
                new = gen[:, enc["input_ids"].shape[1]:]
                texts = tok.batch_decode(new, skip_special_tokens=True)
                for j, r in enumerate(chunk):
                    for s in range(4):
                        fout.write(json.dumps({
                            "row_index": r["row_index"], "height": r["height"],
                            "arm": arm, "sample_index": s,
                            "model_output": texts[j * 4 + s],
                        }, ensure_ascii=False) + "\n")
                        n_out += 1
                fout.flush()
                print(f"{arm} {i + len(chunk)}/{len(rows)} rows, {time.time()-t1:.0f}s", flush=True)

    summary = {"n_generations": n_out, "model": model_id,
               "model_class": model.__class__.__name__,
               "load_seconds": round(t1 - t0), "gen_seconds": round(time.time() - t1), "no_think": no_think, "max_new_tokens": max_new}
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
