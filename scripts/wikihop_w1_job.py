"""Item W1 (registered) — WikiHop calibration, givemeanode H100 entrypoint.

On the 12 seeded doc-dependent failing rows (seed 20260822, pinned by W0):
  * k=8 in-job baselines (no hook) + gauge read at the pinned layer;
  * gold-only class-mean fires across the amplitude ladder {0.25,0.5,1.0} x base
    at k=4 (write at L30 on every gold-candidate mention position), with the
    gauge read under the write;
  * one full per-candidate pass (every candidate, k=4) at the middle rung;
  * extra telemetry beyond the registration: a few seeded non-gold fires at the
    non-middle rungs so the delivery fingerprint is readable at every rung.
Delivery is verified per branch: the L30 hook counts its prefill applications
INSIDE generate (the L-series bug class) and a branch with zero applications
aborts the job. W2 reuses this entrypoint: W1_ROWS_KEY=w2_pool, a single rung
(the W1-pinned one) as both W1_RUNGS and W1_PERCAND_RUNG, W1_NONGOLD_PER_RUNG=0,
and W1_SHARD_INDEX/W1_SHARD_COUNT over the pool. Registered:
docs/causal_handle_directions.md item W.
"""
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import torch

APP = os.environ.get("W1_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, normalize_answer, std_closed_prompts  # noqa: E402

MODEL = "google/gemma-3-27b-it"


def env_float_list(name, default):
    return [float(x) for x in os.environ.get(name, default).split(",") if x.strip()]


def layers_of(model):
    return model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def with_hidden(out, h):
    return (h, *out[1:]) if isinstance(out, tuple) else h


def candidate_positions(tok, text, offsets, candidates):
    found = {}
    for cand in candidates:
        spans = [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE)]
        pos = sorted({i for (a, b) in spans for i, (s0, s1) in enumerate(offsets)
                      if s0 < b and s1 > a and s1 > s0})
        if pos:
            found[cand] = pos
    return found


class Writer:
    """Additive write hook at one layer: vec added at `positions` on every
    prefill call (seq len > 1, all batch rows); counts applications."""

    def __init__(self, vec, positions):
        self.vec = vec
        self.positions = positions
        self.prefill_calls = 0
        self.positions_written = 0

    def __call__(self, _m, _i, out):
        h = hidden_of(out)
        if h.shape[1] <= 1:
            return out
        v = self.vec.to(h.device, h.dtype)
        n = 0
        for pp in self.positions:
            if pp < h.shape[1]:
                h[:, pp, :] += v
                n += 1
        self.prefill_calls += 1
        self.positions_written += n
        return with_hidden(out, h)


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    seed0 = int(os.environ.get("W1_SEED", "20260822"))
    write_layers = [int(x) for x in os.environ.get("W1_WRITE_LAYERS", os.environ.get("W1_WRITE_LAYER", "30")).split(",") if x.strip()]
    run_percand = os.environ.get("W1_PERCAND", "1") == "1"
    rungs = env_float_list("W1_RUNGS", "0.25,0.5,1.0")
    percand_rung = float(os.environ.get("W1_PERCAND_RUNG", "0.5"))
    n_nongold_extra = int(os.environ.get("W1_NONGOLD_PER_RUNG", "3"))
    k_base = int(os.environ.get("W1_K_BASE", "8"))
    k_fire = int(os.environ.get("W1_K_FIRE", "4"))
    max_new = int(os.environ.get("W1_MAX_NEW_TOKENS", "32"))
    rows_key = os.environ.get("W1_ROWS_KEY", "w1_rows")
    fake = os.environ.get("W1_FAKE_MODEL") == "1"

    pinned = json.load(open(os.path.join(APP, "wikihop_w0_pinned.json")))
    z = np.load(os.path.join(APP, "wikihop_w0_pinned.npz"))
    gauge_layer = int(os.environ.get("W1_GAUGE_LAYER", int(z["gauge_layer"][0])))
    if f"gauge_w_L{gauge_layer}" in z:
        gw, gb, gmean = z[f"gauge_w_L{gauge_layer}"].astype(np.float64), float(z[f"gauge_b_L{gauge_layer}"][0]), z[f"gauge_mean_L{gauge_layer}"].astype(np.float64)
    else:
        assert gauge_layer == int(z["gauge_layer"][0]), "no gauge stored for the requested layer"
        gw, gb, gmean = z["gauge_w"].astype(np.float64), float(z["gauge_b"][0]), z["gauge_mean"].astype(np.float64)

    def write_pins(L):
        cv = z[f"class_vector_L{L}"] if f"class_vector_L{L}" in z else z["class_vector"]
        bn = z[f"base_norm_L{L}"] if f"base_norm_L{L}" in z else z["base_norm"]
        cv = cv.astype(np.float64)
        return cv, cv / max(np.linalg.norm(cv), 1e-8), float(bn[0])
    write_layer = write_layers[0]
    class_vec, unit, base_norm = write_pins(write_layer)
    gauge_downstream = all(gauge_layer > L for L in write_layers)
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    shard_index, shard_count = int(os.environ.get("W1_SHARD_INDEX", "0")), int(os.environ.get("W1_SHARD_COUNT", "1"))
    row_ids = pinned["pools"][rows_key][: int(os.environ.get("W1_MAX_ROWS", "10000"))][shard_index::shard_count]
    print(f"rows={len(row_ids)} ({rows_key} shard {shard_index}/{shard_count}) write_layers={write_layers} gauge=L{gauge_layer} "
          f"(downstream of all writes: {gauge_downstream}) rungs={rungs} percand_rung={percand_rung} percand={run_percand}", flush=True)
    for L in write_layers:
        cv, _, bn = write_pins(L)
        print(f"  L{L}: base_norm={bn:.3f} |class_vec|={np.linalg.norm(cv):.3f}", flush=True)

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
    print(f"model loaded ({time.time()-t0:.0f}s) n_layers={len(layers)} eos_ids={eos_ids}", flush=True)

    def forward_capture(ids, writer=None):
        """One forward; returns (final-token gauge-layer state, per-position L30 norms)."""
        store = {}
        def cap_gauge(_m, _i, out):
            store["g"] = hidden_of(out)[0, -1].detach().float().cpu().numpy().astype(np.float64)
            return out
        def cap_norm(_m, _i, out):
            store["n"] = hidden_of(out)[0].detach().float().norm(dim=-1).cpu().numpy()
            return out
        handles = []
        if writer is not None:
            handles.append(layers[write_layer].register_forward_hook(writer))
        handles.append(layers[write_layer].register_forward_hook(cap_norm))
        handles.append(layers[gauge_layer].register_forward_hook(cap_gauge))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return store["g"], store["n"]

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
        new = out[:, len(ids):].detach().cpu().tolist()
        return [tok.decode(seq, skip_special_tokens=True).strip() for seq in new]

    fout = open(os.path.join(outdir, "wikihop_w1.jsonl"), "w")
    row_summaries = []
    n_fired_branches = 0
    for write_layer in write_layers:
        class_vec, unit, base_norm = write_pins(write_layer)
        for ri, rid in enumerate(row_ids):
            r = frame[rid]
            gold = r["answer"]
            text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + std_closed_prompts(r)["std"]}],
                                           tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]
            cpos = candidate_positions(tok, text, offsets, r["candidates"])
            assert gold in cpos, f"{rid}: gold candidate has no mention positions"
            cands = sorted(cpos)
            row_seed = seed0 + (ri * shard_count + shard_index) * 10007
            b_state, b_norms = forward_capture(ids)
            b_score = gauge_score(b_state)
            all_pos = sorted({p for ps in cpos.values() for p in ps})
            measured = {"mean_norm_all_positions": float(b_norms[1:].mean()),
                        "mean_norm_candidate_positions": float(b_norms[all_pos].mean()),
                        "mean_norm_gold_positions": float(b_norms[cpos[gold]].mean())}

            def emit(condition, rung, fired, outputs, gscore, writer, sample_offset=0, gauge_writer=None):
                nonlocal n_fired_branches
                recs = []
                for s, o in enumerate(outputs):
                    rec = {"id": rid, "row_ordinal": ri, "condition": condition, "rung": rung,
                           "amplitude": None if rung is None else rung * base_norm,
                           "fired_candidate": fired, "fired_is_gold": None if fired is None else fired == gold,
                           "sample_index": sample_offset + s, "model_output": o,
                           "normalized_output": normalize_answer(o),
                           "correct": normalize_answer(o) == normalize_answer(gold),
                           "answers_fired": None if fired is None else normalize_answer(o) == normalize_answer(fired),
                           "gauge_score": gscore, "base_gauge_score": b_score,
                           "n_fired_positions": None if fired is None else len(cpos[fired]),
                           "hook_prefill_calls": None if writer is None else writer.prefill_calls,
                           "hook_positions_written": None if writer is None else writer.positions_written,
                           "gauge_forward_hook_calls": None if gauge_writer is None else gauge_writer.prefill_calls,
                           "write_layer": write_layer, "gauge_layer": gauge_layer}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    recs.append(rec)
                if writer is not None:
                    n_fired_branches += 1
                    if writer.prefill_calls < 1 or writer.positions_written < 1:
                        raise RuntimeError(f"{rid}/{condition}/{fired}: write hook did not fire inside generate "
                                           f"(prefill_calls={writer.prefill_calls}) — execution-invalid")
                return recs

            base_out = []
            for chunk in range(0, k_base, k_fire):
                base_out += generate(ids, min(k_fire, k_base - chunk), row_seed + chunk)
            emit("baseline", None, None, base_out, b_score, None)

            def fire(condition, rung, cand, seed):
                vec = torch.from_numpy((unit * rung * base_norm).astype(np.float32))
                w_g = Writer(vec, cpos[cand])
                s_state, _ = forward_capture(ids, writer=w_g)
                w = Writer(vec, cpos[cand])
                outs = generate(ids, k_fire, seed, writer=w)
                return emit(condition, rung, cand, outs, gauge_score(s_state), w, gauge_writer=w_g)

            rng = np.random.default_rng(row_seed)
            nongold = [c for c in cands if c != gold]
            extra = [nongold[i] for i in rng.choice(len(nongold), size=min(n_nongold_extra, len(nongold)), replace=False)] if nongold else []
            for gi, rung in enumerate(rungs):
                fire("gold_ladder", rung, gold, row_seed + 1000 + gi)
                if rung != percand_rung:
                    for ei, c in enumerate(extra):
                        fire("nongold_ladder", rung, c, row_seed + 2000 + gi * 50 + ei)
            if run_percand:
                for ci, c in enumerate(cands):
                    fire("percand", percand_rung, c, row_seed + 5000 + ci)
            fout.flush()
            row_summaries.append({"id": rid, "write_layer": write_layer, "n_tokens": len(ids), "n_candidates": len(cands),
                                  "n_candidates_in_list": len(r["candidates"]), "gold_positions": len(cpos[gold]),
                                  "base_gauge_score": b_score, **measured})
            print(f"L{write_layer} row {ri+1}/{len(row_ids)} {rid}: {len(cands)} cands, {len(ids)} tokens, gold pos {len(cpos[gold])} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fout.close()

    summary = {"n_rows": len(row_ids), "rows_key": rows_key, "shard_index": shard_index, "shard_count": shard_count,
               "n_fired_branches": n_fired_branches, "write_layers": write_layers, "gauge_downstream_of_all_writes": gauge_downstream,
               "gauge_layer": gauge_layer, "base_norm_by_layer": {str(L): write_pins(L)[2] for L in write_layers}, "rungs": rungs,
               "percand_rung": percand_rung, "seed": seed0, "rows": row_summaries,
               "seconds": round(time.time() - t0)}
    with open(os.path.join(outdir, "wikihop_w1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "rows"}, f)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
