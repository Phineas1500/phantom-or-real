#!/usr/bin/env python3
"""Item WM stage-2 job: the retriever-addressed write on a haystack of tables.
Per test row (free-form std prompt): baseline k=8; the frozen WikiHop
hint-delta direction (WX donors, computed in-job at the write layer) written
at 2x the donor norm at the tokens of (a) the top-3 BM25 rows of the top BM25
table, (b) every row of the top table, (c) the gold row (oracle ceiling),
(d) three random rows of a random non-gold table (control); text arms: the
hint naming the answer (k=8) and the retrieval-text baseline that quotes the
top-3 retrieved rows before the documents (k=8). Records follow the WH job's
schema (condition / arm / correct / positions written)."""
import gzip
import json
import os
import random
import sys
import time

import numpy as np
import torch

APP = os.environ.get("WM_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, hint_first_prompt, normalize_answer, std_closed_prompts  # noqa: E402

MODEL = os.environ.get("WM_MODEL", "google/gemma-3-27b-it")


def layers_of(model):
    try:
        return model.model.language_model.layers
    except AttributeError:
        return model.model.layers


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def with_hidden(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class PerPositionWriter:
    def __init__(self, mat, positions):
        self.mat, self.positions, self.prefill_calls, self.positions_written = mat, positions, 0, 0

    def __call__(self, _m, _i, out):
        h = hidden_of(out)
        if h.shape[1] <= 1:
            return out
        m = self.mat.to(h.device, h.dtype); n = 0
        for j, pp in enumerate(self.positions):
            if pp < h.shape[1]:
                h[:, pp, :] += m[j]; n += 1
        self.prefill_calls += 1; self.positions_written += n
        return with_hidden(out, h)


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    seed0 = int(os.environ.get("WM_SEED", "20260924")); write_layer = int(os.environ.get("WM_WRITE_LAYER", "30")); rung = float(os.environ.get("WM_RUNG", "2.0"))
    k_gen = int(os.environ.get("WM_K", "8")); k_chunk = int(os.environ.get("WM_K_CHUNK", "2")); max_new = int(os.environ.get("WM_MAX_NEW_TOKENS", "32")); fake = os.environ.get("WM_FAKE_MODEL") == "1"
    max_rows = int(os.environ.get("WM_MAX_ROWS", "100000")); job = os.environ.get("WM_JOB", "A")
    pins = json.load(open(os.path.join(APP, os.environ.get("WM_PINS_FILE", "wm_pinned.json"))))
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line); frame[r["id"]] = r
    row_ids = pins["jobs"][job]["test_rows"][:max_rows]; donor_ids = pins["jobs"][job]["donor_rows"][:max_rows]
    print(f"rows={len(row_ids)} donors={len(donor_ids)} write=L{write_layer} rung={rung} k={k_gen} chunk={k_chunk}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(62, 64, len(tok)).float(); dev = "cpu"
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto"); model.eval(); dev = next(model.parameters()).device
    layers = layers_of(model)
    eos_ids = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<end_of_turn>")] if tok.convert_tokens_to_ids("<end_of_turn>") is not None else [])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"model loaded ({time.time()-t0:.0f}s) n_layers={len(layers)}", flush=True)

    def render(user_text):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user_text}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return text, enc["input_ids"], enc["offset_mapping"]

    def forward_keep(ids, keep):
        store = {}
        def cap(_m, _i, out):
            store["w"] = hidden_of(out)[0].detach().float().cpu()[keep]; return out
        h = layers[write_layer].register_forward_hook(cap)
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            h.remove()
        return store["w"]

    def generate(ids, k, seed, writer=None):
        """Long prompts: k sequences in chunks of k_chunk so the replicated prompt cache fits; the write hook fires on every chunk's prefill."""
        outs = []
        for c0 in range(0, k, k_chunk):
            kc = min(k_chunk, k - c0)
            handles = [layers[write_layer].register_forward_hook(writer)] if writer is not None else []
            try:
                torch.manual_seed(seed + c0)
                with torch.inference_mode():
                    out = model.generate(input_ids=torch.tensor([ids], device=dev), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev),
                                         do_sample=True, temperature=0.7, max_new_tokens=max_new, num_return_sequences=kc, eos_token_id=eos_ids, pad_token_id=pad_id)
            finally:
                for h in handles:
                    h.remove()
            outs += [tok.decode(seq, skip_special_tokens=True).strip() for seq in out[:, len(ids):].detach().cpu().tolist()]
            del out
            if not fake:
                torch.cuda.empty_cache()
        return outs

    def mention_positions(text, offsets, cand):
        import re
        pos = []
        for m in re.finditer(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE):
            pos += [j for j, (a, b) in enumerate(offsets) if a >= m.start() and b <= m.end() and b > a]
        return sorted(set(pos))

    # frozen direction from the WikiHop donors (their own prompts)
    acc, n_pos, norms, n_don = None, 0, [], 0
    for did in donor_ids:
        r = frame[did]; gold = r["answer"]
        text_s, ids_s, off_s = render(std_closed_prompts(r)["std"]); pos_s = mention_positions(text_s, off_s, gold)
        text_h, ids_h, off_h = render(hint_first_prompt(r, gold)); pos_h_all = mention_positions(text_h, off_h, gold)
        pos_h = pos_h_all[len(pos_h_all) - len(pos_s):] if len(pos_h_all) > len(pos_s) else None
        if not pos_s or pos_h is None or len(pos_h) != len(pos_s) or any(ids_h[a] != ids_s[b] for a, b in zip(pos_h, pos_s)):
            print(f"  donor skip {did}", flush=True); continue
        d = (forward_keep(ids_h, pos_h) - forward_keep(ids_s, pos_s)).numpy().astype(np.float64)
        acc = d.sum(0) if acc is None else acc + d.sum(0); n_pos += len(pos_s); norms += np.linalg.norm(d, axis=1).tolist(); n_don += 1
    mean_delta = acc / n_pos; unit = mean_delta / max(np.linalg.norm(mean_delta), 1e-8); norm_target = float(np.mean(norms))
    print(f"frozen direction from {n_don} donors / {n_pos} positions: norm target {norm_target:.0f} ({time.time()-t0:.0f}s)", flush=True)

    fout = open(os.path.join(outdir, "wm.jsonl"), "w"); n_fired = 0
    def emit(rid, ri, condition, arm, outputs, gold, npos=None, writer=None, extra=None):
        nonlocal n_fired
        for s, o in enumerate(outputs):
            rec = {"id": rid, "row_ordinal": ri, "condition": condition, "arm": arm, "sample_index": s, "model_output": o, "normalized_output": normalize_answer(o),
                   "correct": normalize_answer(o) == normalize_answer(gold), "n_positions": npos, "hook_prefill_calls": None if writer is None else writer.prefill_calls,
                   "hook_positions_written": None if writer is None else writer.positions_written, "write_layer": write_layer, "rung": rung if writer is not None else None}
            if extra: rec.update(extra)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if writer is not None:
            n_fired += 1
            if writer.prefill_calls < 1 or writer.positions_written < 1:
                raise RuntimeError(f"{rid}/{arm}: write hook did not fire — execution-invalid")

    for ri, rid in enumerate(row_ids):
        r = frame[rid]; gold = r["answer"]; ps = std_closed_prompts(r)
        text_s, ids_s, off_s = render(ps["std"]); row_seed = seed0 + ri * 10007
        doc_start = text_s.find(r["docs"][:80])
        assert doc_start >= 0, f"{rid}: documents not found in the rendered prompt"
        def span_positions(idx_list):
            pos = []
            for i in idx_list:
                sp = r["spans"][i]; a, b = doc_start + sp["start"], doc_start + sp["end"]
                pos += [j for j, (x, y) in enumerate(off_s) if x >= a and y <= b and y > x]
            return sorted(set(pos))
        rng = random.Random(row_seed)
        top_t = r["retrieved_tables"][0]; rows_top = r["retrieved_rows_in_top_table"]; all_top = [i for i, sp in enumerate(r["spans"]) if sp["t"] == top_t]
        non_gold_tables = [t for t in range(r["n_tables"]) if t != r["gold_table"]]; ctrl_t = rng.choice(non_gold_tables)
        ctrl_rows = rng.sample([i for i, sp in enumerate(r["spans"]) if sp["t"] == ctrl_t], min(3, sum(1 for sp in r["spans"] if sp["t"] == ctrl_t)))
        arms = {"retrieved_rows": rows_top, "retrieved_table": all_top, "gold_row": [r["gold_span_index"]], "control_rows": ctrl_rows}
        emit(rid, ri, "baseline", "baseline", generate(ids_s, k_gen, row_seed), gold)
        for ai, (arm, idx) in enumerate(arms.items()):
            pos = span_positions(idx)
            if not pos:
                emit(rid, ri, "skipped", arm, [], gold); continue
            mat = torch.from_numpy(np.tile((unit * norm_target * rung).astype(np.float32), (len(pos), 1)))
            w = PerPositionWriter(mat, pos)
            emit(rid, ri, "delta_write", arm, generate(ids_s, k_gen, row_seed + 100 + ai * 10, writer=w), gold, npos=len(pos), writer=w,
                 extra={"address_hits_gold": r["gold_span_index"] in idx, "n_spans": len(idx)})
        _, ids_h, _ = render(hint_first_prompt(r, r["answer_cased"]))
        emit(rid, ri, "text_hint", "hint", generate(ids_h, k_gen, row_seed + 500), gold)
        quoted = "\n".join(r["docs"][r["spans"][i]["start"]:r["spans"][i]["end"]] for i in rows_top)
        _, ids_q, _ = render(f"Possibly relevant rows (from Table {top_t}):\n{quoted}\n\n" + ps["std"])
        emit(rid, ri, "text_retrieval", "retrieval_text", generate(ids_q, k_gen, row_seed + 600), gold, extra={"address_hits_gold": r["gold_span_index"] in rows_top})
        fout.flush()
        print(f"row {ri+1}/{len(row_ids)} {rid}: {len(ids_s)} tokens, gold table {r['gold_table']} (bm25 top {top_t}), gold row rank in top {r['gold_row_rank_in_top_table']} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    summary = {"n_rows": len(row_ids), "n_fired": n_fired, "write_layer": write_layer, "rung": rung, "k": k_gen, "seed": seed0, "frozen": {"n_donors": n_don, "n_positions": n_pos, "norm_target": norm_target}, "seconds": round(time.time() - t0)}
    with open(os.path.join(outdir, "wm_summary.json"), "w") as f:
        json.dump(summary, f)
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
