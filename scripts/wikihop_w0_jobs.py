"""Item W0 (registered) — WikiHop grading + capture, givemeanode H100 entrypoint.

Mode via W0_MODE env:
  grade   : vLLM — std k=8 + closed k=8 on the 800-row frame (seed 20260821);
            prompts identical to the screening's constructions.
  hint    : vLLM — hint-first TEXT arm (gold named before the documents) k=8
            on the rows listed under W0_ROWS_KEY (default w2_pool) of
            /app/wikihop_w0_pinned.json; item WR stage 1 (seed 20260824).
  capture : HF transformers — one forward per row on the std prompt; captures
            final-token states at layers {38,43,48,53} and per-candidate
            mention-mean states (+ position counts + mean per-position norm)
            at L30, plus per-row per-dim mean squares of the L30 states (all
            positions excl. BOS / candidate positions) so massive-dim-excluded
            norms are computable offline. Stored float32 (v4+): the v3 float16
            store overflowed on Gemma's large-magnitude residual dimensions.
            v5: candidate mentions are whole-word matches (v3/v4 substring
            matching hit 'hawaii' inside 'hawaiian', 'ship' in 'relationship').
Registered: docs/causal_handle_directions.md item W.
"""
import gzip
import json
import os
import re
import time

SYSTEM = "You are a careful reading assistant. Answer concisely."
CAP_LAYERS = [int(x) for x in os.environ.get("W0_CAP_LAYERS", "38,43,48,53").split(",") if x.strip()]
WRITE_LAYERS = [int(x) for x in os.environ.get("W0_WRITE_LAYERS", "30").split(",") if x.strip()]
WRITE_LAYER = WRITE_LAYERS[0]
MODEL = os.environ.get("W0_MODEL", "google/gemma-3-27b-it")
CT_KW = {"enable_thinking": False} if "qwen" in MODEL.lower() else {}


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

    if mode == "hint":
        import sys
        sys.path.insert(0, "/app")
        from wikihop_common import hint_first_prompt
        from vllm import LLM, SamplingParams
        pins = json.load(open("/app/wikihop_w0_pinned.json"))
        ids = pins["pools"][os.environ.get("W0_ROWS_KEY", "w2_pool")]
        by_id = {r["id"]: r for r in rows}
        seed = int(os.environ.get("W0_SEED", "20260824"))
        llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.92)
        convs = [[{"role": "user", "content": SYSTEM + "\n\n" + hint_first_prompt(by_id[i], by_id[i]["answer"])}] for i in ids]
        outs = llm.chat(convs, SamplingParams(n=8, temperature=0.7, max_tokens=32, seed=seed), chat_template_kwargs=CT_KW or None)
        n = 0
        with open(os.path.join(outdir, "wikihop_hint_grades.jsonl"), "w") as f:
            for rid, o in zip(ids, outs):
                for s_i, c in enumerate(o.outputs):
                    f.write(json.dumps({"id": rid, "arm": "hint_first", "sample_index": s_i,
                                        "model_output": c.text}, ensure_ascii=False) + "\n")
                    n += 1
        summary = {"mode": mode, "n_rows": len(ids), "n_generations": n, "seed": seed, "seconds": round(time.time() - t0)}
    elif mode in ("grade", "grade_hint"):
        from vllm import LLM, SamplingParams
        if mode == "grade_hint":
            import sys
            sys.path.insert(0, "/app")
            from wikihop_common import hint_first_prompt
        llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.92)
        convs, meta = [], []
        for r in rows:
            ps = std_closed_prompts(r)
            if mode == "grade_hint":
                ps["hint_first"] = hint_first_prompt(r, r["answer"])
            for arm in ps:
                convs.append([{"role": "user", "content": SYSTEM + "\n\n" + ps[arm]}])
                meta.append((r["id"], arm))
        outs = llm.chat(convs, SamplingParams(n=8, temperature=0.7, max_tokens=32,
                                              seed=int(os.environ.get("W0_SEED", "20260821"))), chat_template_kwargs=CT_KW or None)
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
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        layers_mod = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
        print(f"model loaded ({time.time()-t0:.0f}s); n_layers={len(layers_mod)}", flush=True)

        final_states = {L: [] for L in CAP_LAYERS}
        cand_vectors = {L: [] for L in WRITE_LAYERS}
        manifest = []
        sq_mean_all = {L: [] for L in WRITE_LAYERS}
        sq_mean_cand = {L: [] for L in WRITE_LAYERS}
        for n_done, r in enumerate(rows):
            prompt = SYSTEM + "\n\n" + std_closed_prompts(r)["std"]
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CT_KW)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            cand_positions = {}
            for cand in r["candidates"]:
                spans = [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE)]
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
                       for L in set(CAP_LAYERS + WRITE_LAYERS)]
            try:
                with torch.inference_mode():
                    model(input_ids=torch.tensor([ids], device=next(model.parameters()).device),
                          use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            for L in CAP_LAYERS:
                final_states[L].append(store[L][-1].numpy().astype(np.float32))
            all_pos = sorted({p for ps in cand_positions.values() for p in ps})
            pos_norms = {L: store[L].norm(dim=-1).numpy() for L in WRITE_LAYERS}
            for L in WRITE_LAYERS:
                sq_mean_all[L].append((store[L][1:] ** 2).mean(dim=0).numpy().astype(np.float32))
                sq_mean_cand[L].append((store[L][all_pos] ** 2).mean(dim=0).numpy().astype(np.float32))
            row_cands = []
            for cand, pos in cand_positions.items():
                entry = {"candidate": cand, "n_positions": len(pos),
                         "vec_index": len(cand_vectors[WRITE_LAYER]),
                         "mean_position_norm": float(pos_norms[WRITE_LAYER][pos].mean()),
                         "mean_position_norm_by_layer": {str(L): float(pos_norms[L][pos].mean()) for L in WRITE_LAYERS}}
                for L in WRITE_LAYERS:
                    cand_vectors[L].append(store[L][pos].mean(dim=0).numpy().astype(np.float32))
                row_cands.append(entry)
            manifest.append({"id": r["id"], "answer": r["answer"], "n_tokens": len(ids),
                             "mean_position_norm_excl_bos": float(pos_norms[WRITE_LAYER][1:].mean()),
                             "candidates": row_cands})
            if n_done % 50 == 0:
                print(f"{n_done + 1}/{len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
        arrays = {f"L{L}_final": np.stack(final_states[L]) for L in CAP_LAYERS}
        for L in WRITE_LAYERS:
            arrays[f"cand_L{L}"] = np.stack(cand_vectors[L])
            arrays[f"L{L}_sq_mean_all"] = np.stack(sq_mean_all[L])
            arrays[f"L{L}_sq_mean_cand"] = np.stack(sq_mean_cand[L])
        np.savez(os.path.join(outdir, "wikihop_w0_capture.npz"), **arrays)
        with open(os.path.join(outdir, "wikihop_w0_capture_manifest.json"), "w") as f:
            json.dump({"rows": manifest, "cap_layers": CAP_LAYERS,
                       "write_layer": WRITE_LAYER, "write_layers": WRITE_LAYERS, "dtype": "float32",
                       "addressing": "case-insensitive whole-word (no \\w on either side) span match via offset mapping"}, f)
        summary = {"mode": mode, "n_rows": len(manifest), "write_layers": WRITE_LAYERS,
                   "n_cand_vectors": len(cand_vectors[WRITE_LAYER]), "seconds": round(time.time() - t0)}

    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
