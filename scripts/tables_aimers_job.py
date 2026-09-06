#!/usr/bin/env python3
"""Item WN job: label-free aimers on a haystack of tables, no steering. Per
row (free-form std prompt, Gemma-3 eager attention): (a) one forward with
attention capture — for every layer the final-token attention over keys,
aggregated into per-row-span mass (mean over heads per layer; and per head
for the global layers); (b) self-selection — the model copies the row it
thinks holds the answer (greedy, 64 tokens); (c) contrastive perplexity —
per-token logp of the documents with the question first minus without,
aggregated per span. Outputs wn.jsonl (per row: spans' masses, selection
text, CP per span) for the reader to rank."""
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import torch

APP = os.environ.get("WN_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, std_closed_prompts  # noqa: E402

MODEL = os.environ.get("WN_MODEL", "google/gemma-3-27b-it")
SELECT = ("\n\nWhich single table row contains the answer to the question? Copy that row exactly as it appears, "
          "starting with '|' and including every cell, and nothing else.")


def layers_of(model):
    try:
        return model.model.language_model.layers
    except AttributeError:
        return model.model.layers


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp"); fake = os.environ.get("WN_FAKE_MODEL") == "1"
    job = os.environ.get("WN_JOB", "A"); max_rows = int(os.environ.get("WN_MAX_ROWS", "100000"))
    pins = json.load(open(os.path.join(APP, os.environ.get("WN_PINS_FILE", "wn_pinned.json"))))
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line); frame[r["id"]] = r
    row_ids = pins["jobs"][job][:max_rows]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(62, 64, len(tok)).float(); dev = "cpu"
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"); model.eval(); dev = next(model.parameters()).device
    layers = layers_of(model); n_layers = len(layers)
    cfg = getattr(getattr(model, "config", None), "text_config", getattr(model, "config", None))
    pattern = getattr(cfg, "layer_types", None)
    global_layers = [i for i, t in enumerate(pattern) if "full" in str(t) or "global" in str(t)] if pattern else [i for i in range(n_layers) if (i + 1) % 6 == 0]
    print(f"rows={len(row_ids)} job={job} n_layers={n_layers} global_layers={global_layers}", flush=True)
    eos_ids = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<end_of_turn>")] if tok.convert_tokens_to_ids("<end_of_turn>") is not None else [])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def render(user_text):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user_text}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return text, enc["input_ids"], enc["offset_mapping"]

    attn_store = {}
    def attn_hook(L):
        def fn(_m, _i, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                w = out[1]
                attn_store[L] = w[0, :, -1, :].detach().float().cpu()  # heads x keys, final query
            return out
        return fn
    handles = [] if fake else [layers[L].self_attn.register_forward_hook(attn_hook(L)) for L in range(n_layers)]

    def forward_attn(ids):
        attn_store.clear()
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        return dict(attn_store)

    def forward_logp(ids):
        with torch.inference_mode():
            if fake:
                model(torch.tensor([ids])); return -np.random.default_rng(len(ids)).random(len(ids) - 1) * 5
            lg = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits
            lp = torch.log_softmax(lg[0].float(), -1)
            return lp[torch.arange(len(ids) - 1, device=lp.device), torch.tensor(ids[1:], device=lp.device)].cpu().numpy()  # logp of token p given prefix, index p-1

    def generate(ids, max_new=64):
        with torch.inference_mode():
            if fake:
                model(torch.tensor([ids])); return "| 3 | Spain (ESP) | 1 | 2 | 0 | 3 |"
            out = model.generate(input_ids=torch.tensor([ids], device=dev), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev), do_sample=False, max_new_tokens=max_new, eos_token_id=eos_ids, pad_token_id=pad_id)
        return tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()

    fout = open(os.path.join(outdir, "wn.jsonl"), "w")
    for ri, rid in enumerate(row_ids):
        r = frame[rid]; ps = std_closed_prompts(r)
        text_s, ids_s, off_s = render(ps["std"]); doc_start = text_s.find(r["docs"][:80]); assert doc_start >= 0
        span_tok = []
        for sp in r["spans"]:
            a, b = doc_start + sp["start"], doc_start + sp["end"]
            span_tok.append([j for j, (x, y) in enumerate(off_s) if x >= a and y <= b and y > x])
        rec = {"id": rid, "n_tokens": len(ids_s), "n_spans": len(r["spans"]), "gold_span_index": r["gold_span_index"], "span_token_counts": [len(t) for t in span_tok],
               "span_rel_pos": [float(np.mean(t)) / len(ids_s) if t else None for t in span_tok]}
        if not fake:
            A = forward_attn(ids_s)  # {L: heads x keys}
            layer_mean = np.zeros((n_layers, len(span_tok)), dtype=np.float32); glob = {}
            for L, w in A.items():
                wm = w.mean(0).numpy()  # keys
                layer_mean[L] = [float(wm[t].sum()) if t else 0.0 for t in span_tok]
                if L in global_layers:
                    glob[L] = [[float(w[h][t].sum()) if t else 0.0 for t in span_tok] for h in range(w.shape[0])]
            rec["attn_layer_mean"] = layer_mean.round(6).tolist(); rec["attn_global_heads"] = {str(L): np.round(np.array(v), 6).tolist() for L, v in glob.items()}
            del A
        else:
            rec["attn_layer_mean"] = [[0.0] * len(span_tok)] * 2; rec["attn_global_heads"] = {}
        # self-selection
        sel_text = generate(render(ps["std"] + SELECT)[1]); rec["selection"] = sel_text
        # contrastive perplexity: docs with question first vs docs alone (teacher-forced user content)
        q = r["question"]; body_q = f"Question: {q}\n\nDocuments:\n{r['docs']}"; body_n = f"Documents:\n{r['docs']}"
        cp = []
        for body in (body_q, body_n):
            text_b, ids_b, off_b = render(body); ds = text_b.find(r["docs"][:80]); lp = forward_logp(ids_b)
            per_span = []
            for sp in r["spans"]:
                a, b = ds + sp["start"], ds + sp["end"]; toks = [j for j, (x, y) in enumerate(off_b) if x >= a and y <= b and y > x and j >= 1]
                per_span.append(float(np.mean([lp[j - 1] for j in toks])) if toks else 0.0)
            cp.append(per_span)
        rec["cp_with_q"] = np.round(cp[0], 4).tolist(); rec["cp_without_q"] = np.round(cp[1], 4).tolist()
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        if not fake:
            torch.cuda.empty_cache()
        print(f"row {ri+1}/{len(row_ids)} {rid}: {len(ids_s)} tokens, {len(span_tok)} spans, selection={sel_text[:60]!r} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    for h in handles:
        h.remove()
    summary = {"n_rows": len(row_ids), "job": job, "global_layers": global_layers, "seconds": round(time.time() - t0)}
    with open(os.path.join(outdir, "wn_summary.json"), "w") as f:
        json.dump(summary, f)
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
