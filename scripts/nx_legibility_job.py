#!/usr/bin/env python3
"""Item NX stage 4 — legibility of the neurons named by stages 1 and 3.
givemeanode H100 entrypoint; neurons come from /app/nx_neurons.json
({"neurons": [{"set", "layer", "neuron", "pole"}]}).

NX4_MODE=mine: the registered mining corpus (the WF frame's documents plus
the first 2,000 documents of HuggingFaceFW/fineweb sample-10BT, streaming
train order; each truncated to 512 tokens, BOS, no chat template). Per
neuron: max over tokens of pole * a (a = down_proj input), the top-20
texts with the firing token marked in a +-32-token window, and the mean
top-20 value (the fire scale).
NX4_MODE=score: /app/nx_tests.json ({"tests": [{"key", "texts": {kind:
[...]}}]}, key = set:layer:neuron) — max pole * a per test text."""
from __future__ import annotations
import gzip, heapq, json, os, sys, time
import torch

APP = os.environ.get("NX_APP_DIR", "/app")
MODEL = os.environ.get("NX_MODEL", "google/gemma-3-27b-it")


class Stop(Exception):
    pass


def layers_of(model):
    return model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    mode = os.environ.get("NX4_MODE", "mine")
    n_web = int(os.environ.get("NX4_N_WEB", "2000"))
    max_tok = int(os.environ.get("NX4_MAX_TOKENS", "512"))
    top_n = int(os.environ.get("NX4_TOP", "20"))
    bs = int(os.environ.get("NX4_BATCH", "8"))
    neurons = json.load(open(os.path.join(APP, "nx_neurons.json")))["neurons"]
    by_layer = {}
    for n in neurons:
        by_layer.setdefault(n["layer"], []).append(n)
    max_layer = max(by_layer)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    if os.environ.get("NX_FAKE_MODEL") == "1":
        sys.path.insert(0, APP)
        from fake_gemma import FakeGemma
        model = FakeGemma(n_layers=40, d=64, vocab=len(tok), inter=128, n_heads=4, head_dim=8)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    layers = layers_of(model)
    dev = next(model.parameters()).device
    idx = {L: torch.tensor([n["neuron"] for n in ns], device=dev) for L, ns in by_layer.items()}
    pole = {L: torch.tensor([float(n["pole"]) for n in ns], device=dev) for L, ns in by_layer.items()}
    print(f"mode={mode} neurons={len(neurons)} layers={sorted(by_layer)} ({time.time()-t0:.0f}s)", flush=True)

    def acts(texts):
        """Per text, per layer: (n_tokens, n_neurons_in_layer) pole-signed activations (BOS excluded)."""
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tok, add_special_tokens=True)
        ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        store = {}

        def grab(L):
            def fn(_m, args):
                store[L] = (args[0][:, :, idx[L]].float() * pole[L]).cpu()
                if L == max_layer:
                    raise Stop
            return fn
        hs = [layers[L].mlp.down_proj.register_forward_pre_hook(grab(L)) for L in by_layer]
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        out = []
        m = mask.cpu().bool()
        for b in range(len(texts)):
            keep = m[b].clone()
            keep[(keep.cumsum(0) == 1)] = False
            out.append(({L: store[L][b][keep] for L in by_layer}, enc["input_ids"][b][keep]))
        return out

    tok.padding_side = "right"
    if mode == "mine":
        texts = []
        for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
            texts.append(("wf", json.loads(line)["docs"]))
        if n_web > 0:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
            for i, ex in enumerate(ds):
                if i >= n_web:
                    break
                texts.append(("web", ex["text"]))
        print(f"corpus {len(texts)} texts ({time.time()-t0:.0f}s)", flush=True)
        heaps = {(n["set"], n["layer"], n["neuron"]): [] for n in neurons}
        for s in range(0, len(texts), bs):
            chunk = texts[s:s + bs]
            for bi, ((src, _), (per_layer, ids)) in enumerate(zip(chunk, acts([t for _, t in chunk]))):
                for L, ns in by_layer.items():
                    a = per_layer[L]
                    if a.shape[0] == 0:
                        continue
                    vals, pos = a.max(0)
                    for j, n in enumerate(ns):
                        key = (n["set"], L, n["neuron"])
                        item = (float(vals[j]), s + bi, int(pos[j]), src)
                        h = heaps[key]
                        if len(h) < top_n:
                            heapq.heappush(h, item + (ids,))
                        elif item[0] > h[0][0]:
                            heapq.heapreplace(h, item + (ids,))
            if (s // bs) % 50 == 0:
                print(f"  {s + len(chunk)}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
        out = []
        for n in neurons:
            key = (n["set"], n["layer"], n["neuron"])
            tops = sorted(heaps[key], key=lambda x: -x[0])
            ctxs = []
            for v, _, p, src, ids in tops:
                lo, hi = max(0, p - 32), min(len(ids), p + 33)
                ctxs.append({"value": v, "source": src,
                             "text": tok.decode(ids[lo:p]) + " [[" + tok.decode(ids[p:p + 1]) + "]] " + tok.decode(ids[p + 1:hi])})
            out.append({**n, "fire_scale": sum(c["value"] for c in ctxs) / max(len(ctxs), 1), "top_contexts": ctxs})
        result = {"mode": "mine", "n_texts": len(texts), "n_neurons": len(out), "seconds": round(time.time() - t0)}
        json.dump({"neurons": out, **result}, open(os.path.join(outdir, "nx_mined.json"), "w"))
    else:
        tests = json.load(open(os.path.join(APP, "nx_tests.json")))["tests"]
        col = {}
        for L, ns in by_layer.items():
            for j, n in enumerate(ns):
                col[f"{n['set']}:{L}:{n['neuron']}"] = (L, j)
        scored = []
        for t in tests:
            L, j = col[t["key"]]
            rec = {"key": t["key"], "maxact": {}}
            for kind, texts in t["texts"].items():
                vals = []
                for s in range(0, len(texts), bs):
                    for per_layer, _ in acts(texts[s:s + bs]):
                        a = per_layer[L][:, j]
                        vals.append(float(a.max()) if a.numel() else float("-inf"))
                rec["maxact"][kind] = vals
            scored.append(rec)
        result = {"mode": "score", "n_tests": len(scored), "seconds": round(time.time() - t0)}
        json.dump({"scored": scored, **result}, open(os.path.join(outdir, "nx_scored.json"), "w"))
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        json.dump(result, open(rp, "w"))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
