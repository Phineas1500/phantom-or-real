"""Item WH (registered) — hint-delta oracle write on the reading-driven slice,
givemeanode H100 entrypoint. Per row: std baseline k=8; hint-first TEXT arms
(gold k=8, 3 seeded non-gold k=4); per-position L30 hint-delta writes at the
fired candidate's whole-word mentions x rungs {0.5,1,2} (gold k=4, non-gold
k=4); W0's L38 gauge read under every write; hook counters inside generate.
Registered: docs/causal_handle_directions.md item WH.
"""
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import torch

APP = os.environ.get("WH_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, hint_first_prompt, normalize_answer, std_closed_prompts  # noqa: E402

MODEL = "google/gemma-3-27b-it"


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
        self.mat = mat
        self.positions = positions
        self.prefill_calls = 0
        self.positions_written = 0

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
    seed0 = int(os.environ.get("WH_SEED", "20260823"))
    write_layer = int(os.environ.get("WH_WRITE_LAYER", "30"))
    rungs = [float(x) for x in os.environ.get("WH_RUNGS", "0.5,1.0,2.0").split(",")]
    n_nongold = int(os.environ.get("WH_NONGOLD", "3"))
    k_base, k_text_gold, k_text_non, k_write = (int(os.environ.get(k, d)) for k, d in
                                                (("WH_K_BASE", "8"), ("WH_K_TEXT_GOLD", "8"), ("WH_K_TEXT_NON", "4"), ("WH_K_WRITE", "4")))
    max_new = int(os.environ.get("WH_MAX_NEW_TOKENS", "32"))
    fake = os.environ.get("WH_FAKE_MODEL") == "1"
    max_rows = int(os.environ.get("WH_MAX_ROWS", "10000"))
    loop = os.environ.get("WH_LOOP", "0") == "1"
    loop_rung = float(os.environ.get("WH_LOOP_RUNG", "2.0"))
    text_nongold = os.environ.get("WH_TEXT_NONGOLD", "0" if loop else "1") == "1"
    shard_index, shard_count = int(os.environ.get("WH_SHARD_INDEX", "0")), int(os.environ.get("WH_SHARD_COUNT", "1"))

    pins = json.load(open(os.path.join(APP, os.environ.get("WH_PINS_FILE", "wikihop_wh_pinned.json"))))
    rows_key = os.environ.get("WH_ROWS_KEY", "wh_rows")
    z = np.load(os.path.join(APP, "wikihop_w0_pinned.npz"))
    gauge_layer = int(os.environ.get("WH_GAUGE_LAYER", int(z["gauge_layer"][0])))
    gw, gb, gmean = (z[f"gauge_w_L{gauge_layer}"].astype(np.float64), float(z[f"gauge_b_L{gauge_layer}"][0]),
                     z[f"gauge_mean_L{gauge_layer}"].astype(np.float64))
    base_norm = float(z["base_norm"][0])
    assert gauge_layer > write_layer
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    row_ids = pins[rows_key][:max_rows][shard_index::shard_count]
    print(f"rows={len(row_ids)} ({rows_key} shard {shard_index}/{shard_count}) write=L{write_layer} gauge=L{gauge_layer} rungs={rungs} "
          f"loop={loop} loop_rung={loop_rung} text_nongold={text_nongold} base_norm(W0)={base_norm:.1f}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(n_layers=62, d=len(gw), vocab=len(tok))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="sdpa")
    model.eval()
    layers = layers_of(model)
    dev = next(model.parameters()).device
    eos_ids = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")} - {None})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"model loaded ({time.time()-t0:.0f}s) n_layers={len(layers)}", flush=True)

    def render(user_text):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user_text}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return text, enc["input_ids"], enc["offset_mapping"]

    def forward(ids, writer=None, keep_positions=None):
        """Returns (gauge-layer final-token state, write-layer states at keep_positions or all)."""
        store = {}
        def cap_gauge(_m, _i, out):
            store["g"] = hidden_of(out)[0, -1].detach().float().cpu().numpy().astype(np.float64)
            return out
        def cap_write(_m, _i, out):
            h = hidden_of(out)[0].detach().float().cpu()
            store["w"] = h if keep_positions is None else h[keep_positions]
            return out
        handles = []
        if writer is not None:
            handles.append(layers[write_layer].register_forward_hook(writer))
        handles.append(layers[write_layer].register_forward_hook(cap_write))
        handles.append(layers[gauge_layer].register_forward_hook(cap_gauge))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return store["g"], store["w"]

    def gauge_score(x):
        return float(gw @ (x - gmean) + gb)

    def generate(ids, k, seed, writer=None):
        handles = [layers[write_layer].register_forward_hook(writer)] if writer is not None else []
        try:
            torch.manual_seed(seed)
            with torch.inference_mode():
                out = model.generate(input_ids=torch.tensor([ids], device=dev),
                                     attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev),
                                     do_sample=True, temperature=0.7, max_new_tokens=max_new,
                                     num_return_sequences=k, eos_token_id=eos_ids, pad_token_id=pad_id)
        finally:
            for h in handles:
                h.remove()
        return [tok.decode(seq, skip_special_tokens=True).strip() for seq in out[:, len(ids):].detach().cpu().tolist()]

    fout = open(os.path.join(outdir, "wikihop_wh.jsonl"), "w")
    row_summaries, n_fired, skipped = [], 0, []
    for ri, rid in enumerate(row_ids):
        r = frame[rid]
        gold = r["answer"]
        text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
        row_seed = seed0 + (ri * shard_count + shard_index) * 10007
        g_state, h_std_all = forward(ids_s)
        b_score = gauge_score(g_state)
        cands_pos = {c: mention_positions(text_s, off_s, c) for c in r["candidates"]}
        cands_pos = {c: p for c, p in cands_pos.items() if p}
        assert gold in cands_pos, f"{rid}: gold has no mentions"
        rng = np.random.default_rng(row_seed)
        nongold_all = sorted(c for c in cands_pos if c != gold)
        nongold = [nongold_all[i] for i in rng.choice(len(nongold_all), size=min(n_nongold, len(nongold_all)), replace=False)]

        def emit(condition, rung, fired, outputs, gscore, writer=None, gauge_writer=None, delta_norm=None, npos=None, sample_offset=0):
            nonlocal n_fired
            for s, o in enumerate(outputs):
                rec = {"id": rid, "row_ordinal": ri, "condition": condition, "rung": rung, "fired_candidate": fired,
                       "fired_is_gold": None if fired is None else fired == gold, "sample_index": sample_offset + s,
                       "model_output": o, "normalized_output": normalize_answer(o),
                       "correct": normalize_answer(o) == normalize_answer(gold),
                       "answers_fired": None if fired is None else normalize_answer(o) == normalize_answer(fired),
                       "gauge_score": gscore, "base_gauge_score": b_score, "n_fired_positions": npos,
                       "delta_mean_position_norm": delta_norm,
                       "hook_prefill_calls": None if writer is None else writer.prefill_calls,
                       "hook_positions_written": None if writer is None else writer.positions_written,
                       "gauge_forward_hook_calls": None if gauge_writer is None else gauge_writer.prefill_calls,
                       "write_layer": write_layer, "gauge_layer": gauge_layer}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if writer is not None:
                n_fired += 1
                if writer.prefill_calls < 1 or writer.positions_written < 1:
                    raise RuntimeError(f"{rid}/{condition}/{fired}: write hook did not fire inside generate — execution-invalid")

        base_out = []
        for chunk in range(0, k_base, k_write):
            base_out += generate(ids_s, min(k_write, k_base - chunk), row_seed + chunk)
        emit("baseline", None, None, base_out, b_score)

        row_deltas = {}
        fired_list = [gold] + (sorted(c for c in cands_pos if c != gold) if loop else nongold)
        for ci, cand in enumerate(fired_list):
            is_gold = cand == gold
            seeded = cand in nongold
            cand_rungs = rungs if (is_gold or seeded) else []
            if loop and loop_rung not in cand_rungs:
                cand_rungs = cand_rungs + [loop_rung]
            text_h, ids_h, off_h = render(hint_first_prompt(r, cand))
            pos_h_all = mention_positions(text_h, off_h, cand)
            pos_s = cands_pos[cand]
            pos_h = pos_h_all[len(pos_h_all) - len(pos_s):] if len(pos_h_all) > len(pos_s) else None
            if pos_h is None or len(pos_h) != len(pos_s) or any(ids_h[a] != ids_s[b] for a, b in zip(pos_h, pos_s)):
                skipped.append({"id": rid, "candidate": cand, "n_std": len(pos_s), "n_hint": len(pos_h_all)})
                print(f"  skip {rid}/{cand!r}: mention pairing failed (std {len(pos_s)}, hint {len(pos_h_all)})", flush=True)
                continue
            _, h_hint = forward(ids_h, keep_positions=pos_h)
            delta = (h_hint - h_std_all[pos_s]).numpy().astype(np.float32)
            dnorm = float(np.linalg.norm(delta, axis=1).mean())
            row_deltas[cand] = {"n_positions": len(pos_s), "mean_position_norm": dnorm,
                                "norm_over_w0_base": dnorm / base_norm}
            k_text = k_text_gold if is_gold else (k_text_non if (text_nongold and seeded) else 0)
            text_out = []
            for chunk in range(0, k_text, k_write):
                text_out += generate(ids_h, min(k_write, k_text - chunk), row_seed + 500 + ci * 50 + chunk)
            if text_out:
                emit("text_hint", None, cand, text_out, None, delta_norm=dnorm, npos=len(pos_s))
            else:
                row_deltas[cand]["text_arm"] = False
            for gi, rung in enumerate(cand_rungs):
                mat = torch.from_numpy(delta * rung)
                w_g = PerPositionWriter(mat, pos_s)
                s_state, _ = forward(ids_s, writer=w_g)
                w = PerPositionWriter(mat, pos_s)
                outs = generate(ids_s, k_write, row_seed + 1000 + ci * 100 + gi * 10, writer=w)
                emit("delta_write", rung, cand, outs, gauge_score(s_state), writer=w, gauge_writer=w_g,
                     delta_norm=dnorm * rung, npos=len(pos_s))
        fout.flush()
        row_summaries.append({"id": rid, "n_tokens": len(ids_s), "n_candidates": len(cands_pos), "gold_positions": len(cands_pos[gold]),
                              "nongold_fired": nongold, "base_gauge_score": b_score, "deltas": row_deltas})
        print(f"row {ri+1}/{len(row_ids)} {rid}: {len(cands_pos)} cands, {len(ids_s)} tokens, gold pos {len(cands_pos[gold])}, "
              f"|δ_gold| {row_deltas.get(gold, {}).get('mean_position_norm', float('nan')):.0f} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    summary = {"n_rows": len(row_ids), "rows_key": rows_key, "shard_index": shard_index, "shard_count": shard_count,
               "loop": loop, "loop_rung": loop_rung, "n_fired_branches": n_fired, "write_layer": write_layer, "gauge_layer": gauge_layer,
               "rungs": rungs, "seed": seed0, "w0_base_norm": base_norm, "skipped": skipped, "rows": row_summaries,
               "seconds": round(time.time() - t0)}
    with open(os.path.join(outdir, "wikihop_wh_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "rows"}, f)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
