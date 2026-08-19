"""Item W0 (registered) — WikiHop grading + capture, givemeanode H100 entrypoint.

Mode via W0_MODE env:
  grade   : vLLM — std k=8 + closed k=8 on the 800-row frame (seed 20260821);
            prompts identical to the screening's constructions.
  capture : HF transformers — one forward per row on the std prompt; captures
            final-token states at layers {38,43,48,53} and per-candidate
            mention-mean states (+ position counts) at L30.
Registered: docs/causal_handle_directions.md item W.
"""
import gzip
import json
import os
import re
import time

SYSTEM = "You are a careful reading assistant. Answer concisely."
CAP_LAYERS = [38, 43, 48, 53]
WRITE_LAYER = 30


def std_closed_prompts(r):
    q = f"Based on the documents, what is the '{r['relation']}' of {r['subject']}?"
    cands = "\n".join(f"- {c}" for c in r["candidates"])
    return {
        "std": f"Documents:\n{r['docs']}\n\nQuestion: {q}\nCandidates:\n{cands}\n\nAnswer with exactly one candidate, nothing else.",
        "closed": f"Question: {q}\nCandidates:\n{cands}\n\nAnswer with exactly one candidate, nothing else.",
    }


def main():
    mode = os.environ.get("W0_MODE", "grade")
    rows = [json.loads(l) for l in gzip.open("/app/wikihop_port_input.jsonl.gz", "rt")]
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    print(f"mode={mode} rows={len(rows)}", flush=True)
    t0 = time.time()

    if mode == "grade":
        from vllm import LLM, SamplingParams
        llm = LLM(model="google/gemma-3-27b-it", max_model_len=8192, gpu_memory_utilization=0.92)
        convs, meta = [], []
        for r in rows:
            ps = std_closed_prompts(r)
            for arm in ("std", "closed"):
                convs.append([{"role": "user", "content": SYSTEM + "\n\n" + ps[arm]}])
                meta.append((r["id"], arm))
        outs = llm.chat(convs, SamplingParams(n=8, temperature=0.7, max_tokens=32, seed=20260821))
        n = 0
        with open(os.path.join(outdir, "wikihop_w0_grades.jsonl"), "w") as f:
            for (rid, arm), o in zip(meta, outs):
                for s, c in enumerate(o.outputs):
                    f.write(json.dumps({"id": rid, "arm": arm, "sample_index": s,
                                        "model_output": c.text}, ensure_ascii=False) + "\n")
                    n += 1
        summary = {"mode": mode, "n_generations": n, "seconds": round(time.time() - t0)}
    else:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-27b-it", torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        layers_mod = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
        print(f"model loaded ({time.time()-t0:.0f}s); n_layers={len(layers_mod)}", flush=True)

        final_states = {L: [] for L in CAP_LAYERS}
        cand_vectors, manifest = [], []
        for n_done, r in enumerate(rows):
            prompt = SYSTEM + "\n\n" + std_closed_prompts(r)["std"]
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            cand_positions = {}
            for cand in r["candidates"]:
                spans = [m.span() for m in re.finditer(re.escape(cand), text, re.IGNORECASE)]
                pos = sorted({i for (a, b) in spans for i, (s0, s1) in enumerate(offsets)
                              if s0 < b and s1 > a and s1 > s0})
                if pos:
                    cand_positions[cand] = pos
            store = {}
            def hook_for(L):
                def fn(_m, _i, out):
                    h = out[0] if isinstance(out, tuple) else out
                    store[L] = h[0].detach().float().cpu()
                    return out
                return fn
            handles = [layers_mod[L].register_forward_hook(hook_for(L))
                       for L in set(CAP_LAYERS + [WRITE_LAYER])]
            try:
                with torch.inference_mode():
                    model(input_ids=torch.tensor([ids], device=next(model.parameters()).device),
                          use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            for L in CAP_LAYERS:
                final_states[L].append(store[L][-1].numpy().astype(np.float16))
            row_cands = []
            for cand, pos in cand_positions.items():
                vec = store[WRITE_LAYER][pos].mean(dim=0).numpy().astype(np.float16)
                row_cands.append({"candidate": cand, "n_positions": len(pos),
                                  "vec_index": len(cand_vectors)})
                cand_vectors.append(vec)
            manifest.append({"id": r["id"], "answer": r["answer"], "n_tokens": len(ids),
                             "candidates": row_cands})
            if n_done % 50 == 0:
                print(f"{n_done + 1}/{len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
        arrays = {f"L{L}_final": np.stack(final_states[L]) for L in CAP_LAYERS}
        arrays["cand_L30"] = np.stack(cand_vectors)
        np.savez(os.path.join(outdir, "wikihop_w0_capture.npz"), **arrays)
        with open(os.path.join(outdir, "wikihop_w0_capture_manifest.json"), "w") as f:
            json.dump({"rows": manifest, "cap_layers": CAP_LAYERS,
                       "write_layer": WRITE_LAYER}, f)
        summary = {"mode": mode, "n_rows": len(manifest),
                   "n_cand_vectors": len(cand_vectors), "seconds": round(time.time() - t0)}

    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
