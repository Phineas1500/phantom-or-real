#!/usr/bin/env python3
"""Item WT — what the frozen hint-delta write does to attention. For each
test row of the WX cross-fit pins (A and B; donors give the frozen
direction exactly as in the WH job), one eager-attention prefill with no
write and one per candidate (gold + the non-gold candidates WX fired at
1x) with the frozen write at WT_WRITE_LAYER x WT_RUNG at that candidate's
whole-word mentions. Records, per layer, the final prompt token's
attention mass (mean and max over heads) onto the written span, the gold
span, and the single most-attended position. Prefill only; no sampling.
Env: WT_MODEL, WT_WRITE_LAYER (30), WT_RUNG (2.0), WT_MAX_TOKENS (1600),
WT_PINS_FILE, WT_JOBS ("A,B"), WT_WX_RECORDS, WT_MAX_ROWS, WT_FAKE_MODEL,
WT_APP_DIR."""
from __future__ import annotations
import gzip, json, os, re, sys, time
import numpy as np
import torch

APP = os.environ.get("WT_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, hint_first_prompt, std_closed_prompts  # noqa: E402

MODEL = os.environ.get("WT_MODEL", "google/gemma-3-27b-it")
IS_QWEN = "qwen" in MODEL.lower()
CT_KW = {"enable_thinking": False} if IS_QWEN else {}


def layers_of(model):
    return model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def with_hidden(out, h):
    return (h, *out[1:]) if isinstance(out, tuple) else h


def mention_positions(text, offsets, cand):
    spans = [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE)]
    return sorted({i for (a, b) in spans for i, (s0, s1) in enumerate(offsets) if s0 < b and s1 > a and s1 > s0})


class PerPositionWriter:
    def __init__(self, mat, positions):
        self.mat, self.positions, self.prefill_calls, self.positions_written = mat, positions, 0, 0

    def __call__(self, _m, _i, out):
        h = hidden_of(out)
        if h.shape[1] <= 1:
            return out
        m = self.mat.to(h.device, h.dtype)
        n = 0
        for j, pp in enumerate(self.positions):
            if pp < h.shape[1]:
                h[:, pp, :] += m[j]
                n += 1
        self.prefill_calls += 1
        self.positions_written += n
        return with_hidden(out, h)


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    write_layer = int(os.environ.get("WT_WRITE_LAYER", "30"))
    rung = float(os.environ.get("WT_RUNG", "2.0"))
    max_tokens = int(os.environ.get("WT_MAX_TOKENS", "1600"))
    max_rows = int(os.environ.get("WT_MAX_ROWS", "10000"))
    fake = os.environ.get("WT_FAKE_MODEL") == "1"
    pins = json.load(open(os.path.join(APP, os.environ.get("WT_PINS_FILE", "wikihop_wx_pinned.json"))))
    jobs = [j for j in os.environ.get("WT_JOBS", "A,B").split(",") if j]
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    fired_nongold = {}
    for f in os.environ.get("WT_WX_RECORDS", "wikihop_wx_a.jsonl,wikihop_wx_b.jsonl").split(","):
        for line in open(os.path.join(APP, f.strip())):
            r = json.loads(line)
            if r["condition"] == "delta_write" and r["rung"] == 1.0 and not r["fired_is_gold"]:
                fired_nongold.setdefault(r["id"], set()).add(r["fired_candidate"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(n_layers=int(os.environ.get("WT_FAKE_LAYERS", "62")), d=int(os.environ.get("WT_FAKE_D", "256")), vocab=len(tok))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    model.eval()
    layers = layers_of(model)
    dev = next(model.parameters()).device
    print(f"model loaded ({time.time()-t0:.0f}s) n_layers={len(layers)} write=L{write_layer} rung={rung} max_tokens={max_tokens}", flush=True)

    def render(user_text):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user_text}], tokenize=False, add_generation_prompt=True, **CT_KW)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return text, enc["input_ids"], enc["offset_mapping"]

    def states_at(ids, positions):
        store = {}
        def cap(_m, _i, out):
            store["h"] = hidden_of(out)[0, positions].detach().float().cpu()
            return out
        h = layers[write_layer].register_forward_hook(cap)
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            h.remove()
        return store["h"]

    def attention_rows(ids, writer=None):
        handles = [layers[write_layer].register_forward_hook(writer)] if writer is not None else []
        try:
            with torch.inference_mode():
                if fake:
                    T = len(ids)
                    atts = tuple(torch.softmax(torch.randn(1, 2, 1, T), dim=-1) for _ in range(len(layers)))
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                else:
                    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, output_attentions=True)
                    atts = tuple(a[:, :, -1:, :].detach().float().cpu() for a in out.attentions)
                    del out
        finally:
            for h in handles:
                h.remove()
        if not fake:
            torch.cuda.empty_cache()
        return [a[0, :, -1, :] for a in atts]

    def masses(rows, written, gold_pos):
        per = {}
        for L, a in enumerate(rows):
            w = a[:, written].sum(-1) if written else torch.zeros(a.shape[0])
            g = a[:, gold_pos].sum(-1) if gold_pos else torch.zeros(a.shape[0])
            top = a.argmax(-1)
            per[str(L)] = {"written_mean": float(w.mean()), "written_max": float(w.max()), "gold_mean": float(g.mean()), "gold_max": float(g.max()),
                           "top_in_written": float(np.mean([int(int(t) in set(written)) for t in top.tolist()])) if written else 0.0,
                           "mass_last_1024": float(a[:, -1024:].sum(-1).mean())}
        return per

    n_rec = 0
    fout = open(os.path.join(outdir, "wikihop_wt.jsonl"), "w")
    summary = {"jobs": {}, "write_layer": write_layer, "rung": rung, "max_tokens": max_tokens, "model": MODEL, "seed": int(os.environ.get("WT_SEED", "20260864")), "vector_mode": os.environ.get("WT_VECTOR_MODE", "hint")}
    for job in jobs:
        test_ids = pins["jobs"][job]["test_rows"][:max_rows]
        donor_ids = pins["jobs"][job]["donor_rows"]
        acc, n_pos, norms, n_donors = None, 0, [], 0
        for did in donor_ids:
            r = frame[did]; gold = r["answer"]
            text_s, ids_s, off_s = render(std_closed_prompts(r)["std"]); pos_s = mention_positions(text_s, off_s, gold)
            text_h, ids_h, off_h = render(hint_first_prompt(r, gold)); pos_h_all = mention_positions(text_h, off_h, gold)
            pos_h = pos_h_all[len(pos_h_all) - len(pos_s):] if len(pos_h_all) > len(pos_s) else None
            if not pos_s or pos_h is None or len(pos_h) != len(pos_s) or any(ids_h[a] != ids_s[b] for a, b in zip(pos_h, pos_s)):
                continue
            d = (states_at(ids_h, pos_h) - states_at(ids_s, pos_s)).numpy().astype(np.float64)
            acc = d.sum(axis=0) if acc is None else acc + d.sum(axis=0); n_pos += len(pos_s); norms.extend(np.linalg.norm(d, axis=1).tolist()); n_donors += 1
        mean_delta = acc / n_pos
        unit, norm_target = mean_delta / max(np.linalg.norm(mean_delta), 1e-8), float(np.mean(norms))
        vector_mode = os.environ.get("WT_VECTOR_MODE", "hint")
        if vector_mode == "flip":
            unit = -unit
        elif vector_mode == "random":
            v = np.random.default_rng(int(os.environ.get("WT_VECTOR_SEED", "20260867"))).standard_normal(unit.shape[0])
            unit = v / np.linalg.norm(v)
        print(f"job {job}: frozen write from {n_donors} donors / {n_pos} positions, norm target {norm_target:.1f} ({time.time()-t0:.0f}s)", flush=True)
        n_rows_used, n_skipped_long = 0, 0
        for rid in test_ids:
            r = frame[rid]; gold = r["answer"]
            text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
            if len(ids_s) > max_tokens:
                n_skipped_long += 1
                continue
            gold_pos = mention_positions(text_s, off_s, gold)
            cands = [gold] + sorted(c for c in fired_nongold.get(rid, set()) if c != gold)
            cand_pos = {c: mention_positions(text_s, off_s, c) for c in cands}
            cand_pos = {c: p for c, p in cand_pos.items() if p}
            if gold not in cand_pos:
                continue
            base_rows = attention_rows(ids_s)
            rec = {"job": job, "id": rid, "condition": "none", "fired_candidate": None, "fired_is_gold": None, "n_tokens": len(ids_s),
                   "n_written_positions": 0, "n_gold_positions": len(gold_pos), "per_layer": masses(base_rows, [], gold_pos),
                   "candidate_masses": {c: masses(base_rows, pos, gold_pos) for c, pos in cand_pos.items()}}
            fout.write(json.dumps(rec) + "\n"); n_rec += 1
            for c, pos in cand_pos.items():
                mat = torch.from_numpy(np.tile(unit.astype(np.float32) * norm_target * rung, (len(pos), 1)))
                w = PerPositionWriter(mat, pos)
                rows_w = attention_rows(ids_s, writer=w)
                rec = {"job": job, "id": rid, "condition": "write", "fired_candidate": c, "fired_is_gold": c == gold, "n_tokens": len(ids_s),
                       "n_written_positions": len(pos), "n_gold_positions": len(gold_pos), "hook_prefill_calls": w.prefill_calls,
                       "hook_positions_written": w.positions_written, "per_layer": masses(rows_w, pos, gold_pos)}
                fout.write(json.dumps(rec) + "\n"); n_rec += 1
            n_rows_used += 1
            print(f"  row {rid}: {len(ids_s)} tokens, {len(cand_pos)} candidates ({time.time()-t0:.0f}s)", flush=True)
        summary["jobs"][job] = {"n_donors": n_donors, "norm_target": norm_target, "n_rows_used": n_rows_used, "n_skipped_long": n_skipped_long, "vector_mode": vector_mode}
    fout.close()
    summary["n_records"] = n_rec; summary["seconds"] = round(time.time() - t0)
    json.dump(summary, open(os.path.join(outdir, "wikihop_wt_summary.json"), "w"))
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        json.dump(summary, open(rp, "w"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
