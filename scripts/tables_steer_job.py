#!/usr/bin/env python3
"""Item WW job: PASTA-style attention steering on the table haystack, aimed by
the model's own retrieval heads. Per row: one unsteered eager forward reads
the 14 pinned heads' attention from the final prompt token over the rows
(aim = top-3 rows by mean mass; gate = peak share); then k = 8 samples per
arm. Steering: in the pinned heads only, for queries after the documents,
add log(alpha) to the attention logits of every key outside the highlighted
rows (PASTA's alpha-scaling of non-highlighted tokens), applied on prefill
and every decode step. Delivery: the bias must fire on every steered chunk
and the pinned heads' highlight share at the final prompt token must rise."""
import gzip
import json
import os
import random
import sys
import time

import numpy as np
import torch

APP = os.environ.get("WW_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, normalize_answer, std_closed_prompts  # noqa: E402

MODEL = os.environ.get("WW_MODEL", "google/gemma-3-27b-it")
STEER = {"active": False, "measure": False, "layer_heads": {}, "bias": None, "q_start": 0, "fired": 0, "rows": {}}


def install_patch():
    import transformers.models.gemma3.modeling_gemma3 as mg
    repeat_kv = mg.repeat_kv

    def patched(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kwargs):
        if scaling is None:
            scaling = module.head_dim ** -0.5
        key_states = repeat_kv(key, module.num_key_value_groups); value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)).mul_(scaling)
        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap
        if attention_mask is not None:
            attn_weights.add_(attention_mask[:, :, :, : key_states.shape[-2]].to(attn_weights.dtype))
        L = module.layer_idx; hs = STEER["layer_heads"].get(L)
        if hs and STEER["active"]:
            q_len, k_len = attn_weights.shape[-2], attn_weights.shape[-1]; q0 = k_len - q_len; i0 = max(0, STEER["q_start"] - q0)
            if i0 < q_len:
                b = STEER["bias"]
                bias = b[:k_len] if k_len <= b.shape[0] else torch.cat([b, b.new_full((k_len - b.shape[0],), float(STEER["log_alpha"]))])
                attn_weights[:, hs, i0:, :] = attn_weights[:, hs, i0:, :] + bias.to(attn_weights.dtype)
                STEER["fired"] += 1
        for h0 in range(0, attn_weights.shape[1], 8):  # per-head-chunk float32 softmax: same numerics, a quarter of the transient
            attn_weights[:, h0:h0 + 8] = torch.nn.functional.softmax(attn_weights[:, h0:h0 + 8], dim=-1, dtype=torch.float32).to(query.dtype)
        if hs and (STEER["measure"] or STEER["active"]) and attn_weights.shape[-2] > 1:
            STEER["rows"][L] = attn_weights[0, hs, -1, :].detach().float().cpu()
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, attn_weights

    mg.eager_attention_forward = patched
    try:
        from transformers.modeling_utils import AttentionInterface
        AttentionInterface.register("eager", patched)
    except Exception as e:
        print(f"AttentionInterface.register skipped: {e}", flush=True)


def layers_of(model):
    try:
        return model.model.language_model.layers
    except AttributeError:
        return model.model.layers


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp"); fake = os.environ.get("WW_FAKE_MODEL") == "1"
    job = os.environ.get("WW_JOB", "FA"); max_rows = int(os.environ.get("WW_MAX_ROWS", "100000"))
    seed0 = int(os.environ.get("WW_SEED", "20260929")); k_gen = int(os.environ.get("WW_K", "8")); k_chunk = int(os.environ.get("WW_K_CHUNK", "2")); max_new = int(os.environ.get("WW_MAX_NEW_TOKENS", "32"))
    arms = [a for a in os.environ.get("WW_ARMS", "baseline,steer_gold:0.01,steer_gold:0.1,steer_heads3:0.01,steer_heads3:0.1,steer_control:0.01,quote_heads3").split(",") if a]
    pins = json.load(open(os.path.join(APP, os.environ.get("WW_PINS_FILE", "ww_pinned.json"))))
    heads = [(int(L), int(h)) for L, h in pins["heads"]]; layer_heads = {}
    for L, h in heads:
        layer_heads.setdefault(L, []).append(h)
    STEER["layer_heads"] = layer_heads
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line); frame[r["id"]] = r
    row_ids = pins["jobs"][job]["rows"][:max_rows]
    print(f"rows={len(row_ids)} job={job} arms={arms} heads={heads} k={k_gen} chunk={k_chunk}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(62, 64, len(tok)).float(); dev = "cpu"
    else:
        install_patch()
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"); model.eval(); dev = next(model.parameters()).device
    eos_ids = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<end_of_turn>")] if tok.convert_tokens_to_ids("<end_of_turn>") is not None else [])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)

    def render(user_text):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user_text}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return text, enc["input_ids"], enc["offset_mapping"]

    def generate(ids, k, seed):
        outs = []
        for c0 in range(0, k, k_chunk):
            kc = min(k_chunk, k - c0); torch.manual_seed(seed + c0)
            with torch.inference_mode():
                if fake:
                    out = model.generate(torch.tensor([ids]), max_new_tokens=4, num_return_sequences=kc, do_sample=True, temperature=0.7, eos_token_id=eos_ids, pad_token_id=pad_id) if hasattr(model, "generate") else None
                    outs += ["fake answer"] * kc; continue
                out = model.generate(input_ids=torch.tensor([ids], device=dev), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev), do_sample=True, temperature=0.7,
                                     max_new_tokens=max_new, num_return_sequences=kc, eos_token_id=eos_ids, pad_token_id=pad_id)
            outs += [tok.decode(seq, skip_special_tokens=True).strip() for seq in out[:, len(ids):].detach().cpu().tolist()]
            del out; torch.cuda.empty_cache()
        return outs

    fout = open(os.path.join(outdir, "ww.jsonl"), "w"); n_steered = 0; n_delivered = 0

    def emit(rid, ri, arm, outputs, gold, extra):
        for s, o in enumerate(outputs):
            rec = {"id": rid, "row_ordinal": ri, "arm": arm, "sample_index": s, "model_output": o, "normalized_output": normalize_answer(o), "correct": normalize_answer(o) == normalize_answer(gold)}
            rec.update(extra); fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

    def share_of(rows_dict, positions):
        if not rows_dict:
            return None
        idx = torch.tensor(sorted(positions)); vals = []
        for L, hs in layer_heads.items():
            if L in rows_dict:
                w = rows_dict[L]; vals += [float(w[j, idx].sum()) for j in range(w.shape[0])]
        return float(np.mean(vals)) if vals else None

    for ri, rid in enumerate(row_ids):
        r = frame[rid]; gold = r["answer"]; ps = std_closed_prompts(r); row_seed = seed0 + ri * 10007; rng = random.Random(row_seed)
        text_s, ids_s, off_s = render(ps["std"]); doc_start = text_s.find(r["docs"][:80]); assert doc_start >= 0; doc_end = doc_start + len(r["docs"])
        q_start = next(j for j, (x, y) in enumerate(off_s) if x >= doc_end)
        span_tok = []
        for sp in r["spans"]:
            a, b = doc_start + sp["start"], doc_start + sp["end"]; span_tok.append([j for j, (x, y) in enumerate(off_s) if x >= a and y <= b and y > x])
        gsi = r["gold_span_index"]
        # aim: unsteered forward, pinned heads' attention from the final prompt token
        STEER.update(active=False, measure=True, rows={})
        if not fake:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids_s], device=dev), use_cache=False)
        STEER["measure"] = False; rows_before = dict(STEER["rows"])
        if rows_before:
            per_head = np.concatenate([rows_before[L].numpy() for L in sorted(rows_before)], 0)  # heads x keys
            mass = np.array([[float(per_head[j, t].sum()) if t else 0.0 for t in span_tok] for j in range(per_head.shape[0])]); m = mass.mean(0)
            order = np.argsort(-m, kind="stable"); top3 = [int(i) for i in order[:3]]; peak = float(m.max() / m.sum()) if m.sum() > 0 else 0.0; gold_rank = int(np.where(order == gsi)[0][0]) + 1
        else:
            top3 = [gsi, (gsi + 1) % len(span_tok), (gsi + 2) % len(span_tok)]; peak = 0.5; gold_rank = 1
        non_gold_tables = [t for t in range(r["n_tables"]) if t != r["gold_table"]]; ctrl_t = rng.choice(non_gold_tables)
        ctrl_rows = rng.sample([i for i, sp in enumerate(r["spans"]) if sp["t"] == ctrl_t], min(3, sum(1 for sp in r["spans"] if sp["t"] == ctrl_t)))
        base_extra = {"gold_span_index": gsi, "aim_top3": top3, "aim_hits_gold": gsi in top3, "aim_gold_rank": gold_rank, "peak_share": round(peak, 4), "n_spans": len(span_tok), "q_start": q_start, "n_tokens": len(ids_s), "control_rows": ctrl_rows}
        for ai, arm in enumerate(arms):
            name, alpha = (arm.split(":")[0], float(arm.split(":")[1])) if ":" in arm else (arm, None)
            seed = row_seed + 100 + ai * 10; extra = dict(base_extra, alpha=alpha)
            if name == "baseline":
                emit(rid, ri, arm, generate(ids_s, k_gen, seed), gold, extra); continue
            if name == "quote_heads3":
                quoted = "\n".join(r["docs"][r["spans"][i]["start"]:r["spans"][i]["end"]] for i in top3)
                _, ids_q, _ = render(f"Possibly relevant rows:\n{quoted}\n\n" + ps["std"]); emit(rid, ri, arm, generate(ids_q, k_gen, seed), gold, extra); continue
            target = {"steer_gold": [gsi], "steer_heads3": top3, "steer_control": ctrl_rows}[name]
            positions = sorted({j for i in target for j in span_tok[i]}); extra.update(target_rows=target, target_hits_gold=gsi in target, n_positions=len(positions))
            bias = torch.full((len(ids_s),), float(np.log(alpha)), dtype=torch.float32, device=dev); bias[positions] = 0.0
            STEER.update(active=True, bias=bias, q_start=q_start, log_alpha=float(np.log(alpha)), fired=0, rows={})
            try:
                outs = generate(ids_s, k_gen, seed)
            finally:
                STEER["active"] = False
            rows_after = dict(STEER["rows"]); n_steered += 1
            sb, sa = share_of(rows_before, positions), share_of(rows_after, positions)
            extra.update(bias_fired=STEER["fired"], highlight_share_before=None if sb is None else round(sb, 4), highlight_share_after=None if sa is None else round(sa, 4))
            if not fake and (STEER["fired"] < 1 or sa is None or sb is None):
                raise RuntimeError(f"{rid}/{arm}: steering did not fire — execution-invalid")
            if sa is not None and sb is not None and sa > sb:
                n_delivered += 1
            emit(rid, ri, arm, outs, gold, extra)
        print(f"row {ri+1}/{len(row_ids)} {rid}: {len(ids_s)} tok, aim top3={top3} gold={gsi} hit={gsi in top3} peak={peak:.3f} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    summary = {"n_rows": len(row_ids), "job": job, "arms": arms, "steered_row_arms": n_steered, "delivered_row_arms": n_delivered, "seconds": round(time.time() - t0)}
    for path in [os.path.join(outdir, "ww_summary.json")] + ([os.environ["GMN_RESULT_PATH"]] if os.environ.get("GMN_RESULT_PATH") else []):
        with open(path, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
