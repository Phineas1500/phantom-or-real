#!/usr/bin/env python3
"""Item WZ job: label-free detectors from the literature, scored on a frame.
For every row in the targets file: (a) P(True) of the model's own modal
answer (Kadavath et al. 2022) — the verification prompt names the proposed
answer and the score is logp(True) − logp(False) for the first verdict
token; (b) hidden states at the answer tokens of the modal answer (Orgad et
al. 2024), teacher-forced as the assistant turn, at layers WZ_LAYERS: mean
over the answer tokens, the last answer token, and the final prompt token.
For rows flagged `blind` in the targets file, per candidate: P(True) and the
context-aware-decoding pair (Shi et al. 2023): logp(candidate | documents)
and logp(candidate | question only) under teacher forcing. Outputs
wikihop_wz_scores.jsonl (scalars) and wikihop_wz_states.npz (float16)."""
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

APP = os.environ.get("WZ_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, std_closed_prompts  # noqa: E402

MODEL = os.environ.get("WZ_MODEL", "google/gemma-3-27b-it")
IS_QWEN = "qwen" in MODEL.lower()
CT_KW = {"enable_thinking": False} if IS_QWEN else {}
LAYERS = [int(x) for x in os.environ.get("WZ_LAYERS", "20,30,38,43,48,53").split(",")]
VERIFY = "\n\nProposed answer: {ans}\nIs the proposed answer correct according to the documents? Reply with exactly one word: True or False."


def layers_of(model):
    m = model
    for name in ("model", "language_model", "layers"):
        m = getattr(m, name, None) or getattr(getattr(m, "model", m), name, None) or m
    return m if isinstance(m, torch.nn.ModuleList) else model.model.language_model.layers


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    fake = os.environ.get("WZ_FAKE_MODEL") == "1"
    max_rows = int(os.environ.get("WZ_MAX_ROWS", "100000"))
    targets = json.load(open(os.path.join(APP, os.environ.get("WZ_TARGETS", "wz_targets.json"))))
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    row_ids = [i for i in targets if i in frame][:max_rows]
    print(f"rows={len(row_ids)} blind={sum(targets[i].get('blind', False) for i in row_ids)} layers={LAYERS} model={MODEL} fake={fake}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(62, 64, len(tok)).float()
        dev = "cpu"
        def logits_of(ids):
            h = model(ids)
            return h @ model.emb.weight.T
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
        model.eval()
        dev = next(model.parameters()).device
        def logits_of(ids):
            return model(input_ids=ids, use_cache=False).logits
    layers = layers_of(model)
    true_id = tok.encode("True", add_special_tokens=False)[0]
    false_id = tok.encode("False", add_special_tokens=False)[0]
    print(f"model loaded ({time.time()-t0:.0f}s) n_layers={len(layers)} true_id={true_id} false_id={false_id}", flush=True)

    store = {}
    def cap(L):
        def fn(_m, _i, out):
            store[L] = hidden_of(out)[0].detach().float().cpu()
            return out
        return fn
    handles = [layers[L].register_forward_hook(cap(L)) for L in LAYERS if L < len(layers)]

    def forward(ids):
        store.clear()
        with torch.inference_mode():
            lg = logits_of(torch.tensor([ids], device=dev))
        return torch.log_softmax(lg[0].float(), -1).cpu()

    def render_gen(user):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user}], tokenize=False, add_generation_prompt=True, **CT_KW)
        return tok(text, add_special_tokens=False)["input_ids"]

    def render_tf(user, answer):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user}, {"role": "assistant", "content": answer}], tokenize=False, add_generation_prompt=False, **CT_KW)
        start = text.rfind(answer)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        pos = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if a >= start and b <= start + len(answer) and b > a]
        return enc["input_ids"], pos

    def p_true(user, answer):
        ids = render_gen(user + VERIFY.format(ans=answer))
        lp = forward(ids)[-1]
        return float(lp[true_id] - lp[false_id]), float(lp[true_id])

    def tf_logp(user, answer):
        ids, pos = render_tf(user, answer)
        if not pos:
            return float("nan"), []
        lp = forward(ids)
        return float(sum(lp[p - 1, ids[p]] for p in pos)), pos

    fout = open(os.path.join(outdir, "wikihop_wz_scores.jsonl"), "w")
    states = {f"L{L}_{k}": [] for L in LAYERS for k in ("mean", "last", "prompt")}
    state_ids = []
    for ri, rid in enumerate(row_ids):
        r = frame[rid]; t = targets[rid]; ps = std_closed_prompts(r); modal = t["modal"]
        rec = {"id": rid, "modal": modal, "blind": bool(t.get("blind", False))}
        rec["p_true_modal"], rec["logp_true_modal"] = p_true(ps["std"], modal)
        ids, pos = render_tf(ps["std"], modal)
        if pos:
            forward(ids)
            for L in LAYERS:
                if L in store:
                    h = store[L]
                    states[f"L{L}_mean"].append(h[pos].mean(0).numpy().astype(np.float16))
                    states[f"L{L}_last"].append(h[pos[-1]].numpy().astype(np.float16))
                    states[f"L{L}_prompt"].append(h[pos[0] - 1].numpy().astype(np.float16))
            state_ids.append(rid)
        rec["n_answer_tokens"] = len(pos)
        if rec["blind"]:
            cands = t.get("candidates") or r["candidates"]
            per = {}
            for c in cands:
                pt, _ = p_true(ps["std"], c)
                l_ctx, _ = tf_logp(ps["std"], c)
                l_no, _ = tf_logp(ps["closed"], c)
                per[c] = {"p_true": pt, "logp_ctx": l_ctx, "logp_noctx": l_no}
            rec["candidates"] = per
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if ri % 25 == 0 or ri == len(row_ids) - 1:
            print(f"row {ri+1}/{len(row_ids)} {rid}: p_true(modal)={rec['p_true_modal']:+.2f} tokens={len(pos)} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    for h in handles:
        h.remove()
    np.savez_compressed(os.path.join(outdir, "wikihop_wz_states.npz"), ids=np.array(state_ids), **{k: np.stack(v) for k, v in states.items() if v})
    summary = {"n_rows": len(row_ids), "n_states": len(state_ids), "layers": LAYERS, "model": MODEL, "seconds": round(time.time() - t0)}
    with open(os.path.join(outdir, "wikihop_wz_summary.json"), "w") as f:
        json.dump(summary, f)
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
