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

MODEL = os.environ.get("WH_MODEL", "google/gemma-3-27b-it")
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
    loop = (os.environ.get("WH_LOOP", "0") == "1" or os.environ.get("WH_WX_JOB") is not None) and os.environ.get("WH_LOOP_OFF") != "1"
    loop_rung = float(os.environ.get("WH_LOOP_RUNG", "2.0"))
    text_nongold = os.environ.get("WH_TEXT_NONGOLD", "0" if loop else "1") == "1"
    shard_index, shard_count = int(os.environ.get("WH_SHARD_INDEX", "0")), int(os.environ.get("WH_SHARD_COUNT", "1"))
    frozen_job = os.environ.get("WH_WX_JOB")  # "A"/"B": frozen-write mode (item WX); test/donor rows from pins["jobs"]

    pins = json.load(open(os.path.join(APP, os.environ.get("WH_PINS_FILE", "wikihop_wh_pinned.json"))))
    rows_key = os.environ.get("WH_ROWS_KEY", "wh_rows")
    z = np.load(os.path.join(APP, "wikihop_w0_pinned.npz"))
    gauge_layer = int(os.environ.get("WH_GAUGE_LAYER", int(z["gauge_layer"][0])))
    gw, gb, gmean = (z[f"gauge_w_L{gauge_layer}"].astype(np.float64), float(z[f"gauge_b_L{gauge_layer}"][0]),
                     z[f"gauge_mean_L{gauge_layer}"].astype(np.float64))
    extra_gauges = {}  # key -> (layer, w, b, mean); scored alongside the primary in the same forward
    for L in [int(x) for x in os.environ.get("WH_GAUGE_LAYERS", "").split(",") if x.strip()]:
        if f"gauge_w_L{L}" in z:
            extra_gauges[f"primary_L{L}"] = (L, z[f"gauge_w_L{L}"].astype(np.float64), float(z[f"gauge_b_L{L}"][0]), z[f"gauge_mean_L{L}"].astype(np.float64))
    g2 = os.environ.get("WH_GAUGE2_NPZ")
    if g2:
        z2 = np.load(os.path.join(APP, g2))
        for L in [int(x) for x in os.environ.get("WH_GAUGE2_LAYERS", "38").split(",") if x.strip()]:
            extra_gauges[f"second_L{L}"] = (L, z2[f"gauge_w_L{L}"].astype(np.float64), float(z2[f"gauge_b_L{L}"][0]), z2[f"gauge_mean_L{L}"].astype(np.float64))
    extra_layers = sorted({v[0] for v in extra_gauges.values()} - {gauge_layer})
    base_norm = float(z["base_norm"][0])
    assert gauge_layer > write_layer
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    if frozen_job:
        row_ids = pins["jobs"][frozen_job]["test_rows"][:max_rows]
        donor_ids = pins["jobs"][frozen_job]["donor_rows"][:max_rows]
    else:
        row_ids = pins[rows_key][:max_rows][shard_index::shard_count]
        donor_ids = []
    print(f"extra gauges: {sorted(extra_gauges)}", flush=True)
    print(f"rows={len(row_ids)} ({rows_key} shard {shard_index}/{shard_count}) write=L{write_layer} gauge=L{gauge_layer} rungs={rungs} "
          f"loop={loop} loop_rung={loop_rung} text_nongold={text_nongold} base_norm(W0)={base_norm:.1f}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(n_layers=int(os.environ.get("WH_FAKE_LAYERS", "62")), d=len(gw), vocab=len(tok))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="sdpa")
    model.eval()
    layers = layers_of(model)
    dev = next(model.parameters()).device
    turn_end = tok.convert_tokens_to_ids("<|im_end|>" if IS_QWEN else "<end_of_turn>")
    eos_ids = sorted({tok.eos_token_id, turn_end} - {None, tok.unk_token_id})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"model loaded ({time.time()-t0:.0f}s) n_layers={len(layers)}", flush=True)

    def render(user_text):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user_text}],
                                       tokenize=False, add_generation_prompt=True, **CT_KW)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return text, enc["input_ids"], enc["offset_mapping"]

    def forward(ids, writer=None, keep_positions=None):
        """Returns (gauge-layer final-token state, write-layer states at keep_positions or all)."""
        store = {}
        def cap_gauge(_m, _i, out):
            store["g"] = hidden_of(out)[0, -1].detach().float().cpu().numpy().astype(np.float64)
            return out
        def cap_extra(L):
            def fn(_m, _i, out):
                store[f"x{L}"] = hidden_of(out)[0, -1].detach().float().cpu().numpy().astype(np.float64)
                return out
            return fn
        def cap_write(_m, _i, out):
            h = hidden_of(out)[0].detach().float().cpu()
            store["w"] = h if keep_positions is None else h[keep_positions]
            return out
        handles = []
        if writer is not None:
            handles.append(layers[write_layer].register_forward_hook(writer))
        handles.append(layers[write_layer].register_forward_hook(cap_write))
        handles.append(layers[gauge_layer].register_forward_hook(cap_gauge))
        for L in extra_layers:
            handles.append(layers[L].register_forward_hook(cap_extra(L)))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        states = {gauge_layer: store["g"], **{L: store[f"x{L}"] for L in extra_layers}}
        store["extra_scores"] = {k: float(w @ (states[L] - m) + b) for k, (L, w, b, m) in extra_gauges.items()}
        last_extra["scores"] = store["extra_scores"]
        last_extra["states"] = {L: states[L].astype(np.float32) for L in sorted(states)}
        return store["g"], store["w"]

    last_extra = {"scores": {}}

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

    frozen = None
    if frozen_job:
        acc, n_pos, norms, n_donors = None, 0, [], 0
        for did in donor_ids:
            r = frame[did]
            gold = r["answer"]
            text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
            pos_s = mention_positions(text_s, off_s, gold)
            text_h, ids_h, off_h = render(hint_first_prompt(r, gold))
            pos_h_all = mention_positions(text_h, off_h, gold)
            pos_h = pos_h_all[len(pos_h_all) - len(pos_s):] if len(pos_h_all) > len(pos_s) else None
            if not pos_s or pos_h is None or len(pos_h) != len(pos_s) or any(ids_h[a] != ids_s[b] for a, b in zip(pos_h, pos_s)):
                print(f"  donor skip {did}", flush=True)
                continue
            _, h_std = forward(ids_s, keep_positions=pos_s)
            _, h_hint = forward(ids_h, keep_positions=pos_h)
            d = (h_hint - h_std).numpy().astype(np.float64)
            acc = d.sum(axis=0) if acc is None else acc + d.sum(axis=0)
            n_pos += len(pos_s)
            norms.extend(np.linalg.norm(d, axis=1).tolist())
            n_donors += 1
        mean_delta = acc / n_pos
        frozen = {"unit": mean_delta / max(np.linalg.norm(mean_delta), 1e-8), "norm_target": float(np.mean(norms)),
                  "n_donors": n_donors, "n_positions": n_pos, "mean_delta_norm": float(np.linalg.norm(mean_delta)),
                  "mean_position_norm": float(np.mean(norms))}
        print(f"frozen write from {n_donors} donors / {n_pos} positions: |mean δ| {frozen['mean_delta_norm']:.1f}, "
              f"norm target (donor mean per-position |δ|) {frozen['norm_target']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    if os.environ.get("WH_CAPTURE_ONLY", "0") == "1":
        assert frozen is not None, "capture-only mode is defined for the frozen-write (WX) design"
        cap_layers = sorted({gauge_layer, *extra_layers})
        branch_states, manifest, base_states = [], [], []
        for ri, rid in enumerate(row_ids):
            r = frame[rid]
            gold = r["answer"]
            text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
            cands_pos = {c: mention_positions(text_s, off_s, c) for c in r["candidates"]}
            cands_pos = {c: p for c, p in cands_pos.items() if p}
            assert gold in cands_pos
            g_state, _ = forward(ids_s)
            base_states.append(np.stack([last_extra["states"][L] for L in cap_layers]))
            base_scores = dict(last_extra["scores"])
            for cand in sorted(cands_pos):
                pos_s = cands_pos[cand]
                mat = torch.from_numpy(np.tile(frozen["unit"].astype(np.float32) * frozen["norm_target"] * loop_rung, (len(pos_s), 1)))
                w_g = PerPositionWriter(mat, pos_s)
                forward(ids_s, writer=w_g)
                if w_g.prefill_calls < 1 or w_g.positions_written < 1:
                    raise RuntimeError(f"{rid}/{cand}: write hook did not fire in the capture forward")
                branch_states.append(np.stack([last_extra["states"][L] for L in cap_layers]))
                manifest.append({"id": rid, "row_ordinal": ri, "fired_candidate": cand, "fired_is_gold": cand == gold, "rung": loop_rung,
                                 "n_fired_positions": len(pos_s), "gauge_scores": dict(last_extra["scores"]), "base_gauge_scores": base_scores,
                                 "hook_prefill_calls": w_g.prefill_calls, "hook_positions_written": w_g.positions_written})
            print(f"capture row {ri+1}/{len(row_ids)} {rid}: {len(cands_pos)} branches ({time.time()-t0:.0f}s)", flush=True)
        np.savez(os.path.join(outdir, "wikihop_branch_states.npz"), branch_states=np.stack(branch_states),
                 base_states=np.stack(base_states), cap_layers=np.array(cap_layers))
        with open(os.path.join(outdir, "wikihop_branch_manifest.json"), "w") as f:
            json.dump({"rows": row_ids, "branches": manifest, "cap_layers": cap_layers, "loop_rung": loop_rung,
                       "frozen": {k: v for k, v in frozen.items() if k != "unit"}, "wx_job": frozen_job, "seed": seed0}, f)
        summary = {"mode": "capture_only", "n_rows": len(row_ids), "n_branches": len(manifest), "cap_layers": cap_layers,
                   "wx_job": frozen_job, "frozen": {k: v for k, v in frozen.items() if k != "unit"}, "seconds": round(time.time() - t0)}
        rp = os.environ.get("GMN_RESULT_PATH")
        if rp:
            with open(rp, "w") as f:
                json.dump(summary, f)
        print(json.dumps(summary), flush=True)
        return

    fout = open(os.path.join(outdir, "wikihop_wh.jsonl"), "w")
    row_summaries, n_fired, skipped = [], 0, []
    for ri, rid in enumerate(row_ids):
        r = frame[rid]
        gold = r["answer"]
        text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
        row_seed = seed0 + (ri * shard_count + shard_index) * 10007
        g_state, h_std_all = forward(ids_s)
        b_score = gauge_score(g_state)
        b_extra = dict(last_extra["scores"])
        cands_pos = {c: mention_positions(text_s, off_s, c) for c in r["candidates"]}
        cands_pos = {c: p for c, p in cands_pos.items() if p}
        assert gold in cands_pos, f"{rid}: gold has no mentions"
        rng = np.random.default_rng(row_seed)
        nongold_all = sorted(c for c in cands_pos if c != gold)
        nongold = [nongold_all[i] for i in rng.choice(len(nongold_all), size=min(n_nongold, len(nongold_all)), replace=False)]

        def emit(condition, rung, fired, outputs, gscore, writer=None, gauge_writer=None, delta_norm=None, npos=None, sample_offset=0, extra_scores=None):
            nonlocal n_fired
            for s, o in enumerate(outputs):
                rec = {"id": rid, "row_ordinal": ri, "condition": condition, "rung": rung, "fired_candidate": fired,
                       "fired_is_gold": None if fired is None else fired == gold, "sample_index": sample_offset + s,
                       "model_output": o, "normalized_output": normalize_answer(o),
                       "correct": normalize_answer(o) == normalize_answer(gold),
                       "answers_fired": None if fired is None else normalize_answer(o) == normalize_answer(fired),
                       "gauge_score": gscore, "base_gauge_score": b_score, "n_fired_positions": npos,
                       "gauge_scores": extra_scores, "base_gauge_scores": b_extra,
                       "delta_mean_position_norm": delta_norm, "write_kind": "frozen" if frozen is not None else "per_candidate",
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
            if frozen is not None and not is_gold:
                delta = np.tile(frozen["unit"].astype(np.float32) * frozen["norm_target"], (len(pos_s), 1))
                dnorm = frozen["norm_target"]
                row_deltas[cand] = {"n_positions": len(pos_s), "mean_position_norm": dnorm, "write_kind": "frozen"}
            else:
                _, h_hint = forward(ids_h, keep_positions=pos_h)
                own = (h_hint - h_std_all[pos_s]).numpy().astype(np.float32)
                if frozen is not None:
                    own_mean = own.astype(np.float64).mean(axis=0)
                    cos = float(own_mean @ frozen["unit"] / max(np.linalg.norm(own_mean), 1e-8))
                    delta = np.tile(frozen["unit"].astype(np.float32) * frozen["norm_target"], (len(pos_s), 1))
                    dnorm = frozen["norm_target"]
                    row_deltas[cand] = {"n_positions": len(pos_s), "mean_position_norm": dnorm, "write_kind": "frozen",
                                        "own_delta_mean_position_norm": float(np.linalg.norm(own, axis=1).mean()),
                                        "cos_frozen_vs_own_gold_delta": cos}
                else:
                    delta = own
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
                s_extra = dict(last_extra["scores"])
                w = PerPositionWriter(mat, pos_s)
                outs = generate(ids_s, k_write, row_seed + 1000 + ci * 100 + gi * 10, writer=w)
                emit("delta_write", rung, cand, outs, gauge_score(s_state), writer=w, gauge_writer=w_g,
                     delta_norm=dnorm * rung, npos=len(pos_s), extra_scores=s_extra)
        fout.flush()
        row_summaries.append({"id": rid, "n_tokens": len(ids_s), "n_candidates": len(cands_pos), "gold_positions": len(cands_pos[gold]),
                              "nongold_fired": nongold, "base_gauge_score": b_score, "deltas": row_deltas})
        print(f"row {ri+1}/{len(row_ids)} {rid}: {len(cands_pos)} cands, {len(ids_s)} tokens, gold pos {len(cands_pos[gold])}, "
              f"|δ_gold| {row_deltas.get(gold, {}).get('mean_position_norm', float('nan')):.0f} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    summary = {"n_rows": len(row_ids), "rows_key": rows_key, "shard_index": shard_index, "shard_count": shard_count,
               "loop": loop, "loop_rung": loop_rung, "n_fired_branches": n_fired, "write_layer": write_layer, "gauge_layer": gauge_layer,
               "rungs": rungs, "seed": seed0, "frozen": ({k: v for k, v in frozen.items() if k != "unit"} if frozen else None),
               "wx_job": frozen_job, "w0_base_norm": base_norm, "skipped": skipped, "rows": row_summaries,
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
