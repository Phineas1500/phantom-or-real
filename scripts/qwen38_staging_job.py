"""Qwen3.8-27B staging capture (givemeanode H100 batch job entrypoint) — the
J1-analog reconnaissance: where does outcome information stage in the hybrid?

One forward per row (batch 1; no generation), output_hidden_states, capturing
three position kinds per layer: mean over gold-concept mention tokens, the
last mention token, and the final prompt token (internal consistency check vs
the gauge-hunt capture). Concept mentions found by the house convention
(stage2_hint_delta.concept_positions): stem = concept.lower().rstrip('s'),
all offset-overlapping tokens. Same 2,000 rows and row order as the gauge
hunt, so the existing greedy labels align by index. Thinking disabled.
"""
import gzip
import json
import os
import time


def find_positions(offsets, text, concept):
    stem = concept.lower().rstrip("s")
    lowered = text.lower()
    spans = []
    start = 0
    while True:
        hit = lowered.find(stem, start)
        if hit < 0:
            break
        spans.append((hit, hit + len(stem)))
        start = hit + 1
    return [i for i, (a, b) in enumerate(offsets)
            if any(a < se and b > ss for ss, se in spans)]


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("STAGING_MODEL", "Qwen/Qwen3.8-27B")
    rows = [json.loads(l) for l in gzip.open("/app/staging_input.jsonl.gz", "rt")]
    print(f"loaded {len(rows)} rows; model={model_id}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    t_load = time.time() - t0
    print(f"model loaded in {t_load:.0f}s; class={model.__class__.__name__}", flush=True)

    def build(row):
        msgs = [{"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["prompt_text"]}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            return tok.apply_chat_template(msgs, **kwargs)

    n_layers_plus1 = model.config.num_hidden_layers + 1 if not hasattr(model.config, "text_config") \
        else model.config.text_config.num_hidden_layers + 1
    d_model = getattr(getattr(model.config, "text_config", model.config), "hidden_size")
    n = len(rows)
    mean_mention = np.full((n_layers_plus1, n, d_model), np.nan, dtype=np.float16)
    last_mention = np.full((n_layers_plus1, n, d_model), np.nan, dtype=np.float16)
    final_tok = np.zeros((n_layers_plus1, n, d_model), dtype=np.float16)
    n_mentions = np.zeros(n, dtype=np.int64)

    t1 = time.time()
    for k, r in enumerate(rows):
        text = build(r)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                  add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        pos = find_positions(offsets, text, r["target_concept"])
        n_mentions[k] = len(pos)
        enc = {kk: v.to("cuda:0") for kk, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        for li, h in enumerate(out.hidden_states):
            h0 = h[0]
            final_tok[li, k] = h0[-1].to(torch.float16).cpu().numpy()
            if pos:
                sel = h0[pos].to(torch.float32)
                mean_mention[li, k] = sel.mean(dim=0).to(torch.float16).cpu().numpy()
                last_mention[li, k] = h0[pos[-1]].to(torch.float16).cpu().numpy()
        del out
        if k % 100 == 0:
            print(f"{k + 1}/{n} rows, {time.time()-t1:.0f}s "
                  f"(mentions this row: {n_mentions[k]})", flush=True)

    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    np.save(os.path.join(outdir, "staging_mean_mention.npy"), mean_mention)
    np.save(os.path.join(outdir, "staging_last_mention.npy"), last_mention)
    np.save(os.path.join(outdir, "staging_final_token.npy"), final_tok)
    with open(os.path.join(outdir, "staging_meta.json"), "w") as f:
        json.dump({"order": [r["row_index"] for r in rows],
                   "n_mentions": n_mentions.tolist(),
                   "shape": list(final_tok.shape), "dtype": "float16",
                   "position_kinds": ["mean_mention", "last_mention", "final_token"]}, f)

    summary = {"model": model_id, "n_rows": n,
               "rows_with_mentions": int((n_mentions > 0).sum()),
               "mention_tokens_mean": float(n_mentions.mean()),
               "load_seconds": round(t_load), "total_seconds": round(time.time() - t0)}
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        with open(rp, "w") as f:
            json.dump(summary, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
