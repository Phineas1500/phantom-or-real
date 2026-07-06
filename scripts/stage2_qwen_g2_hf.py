#!/usr/bin/env python3
"""Item G2 (HF pathway): Qwen layer sweep — concept_replace vs random_replace
at relative depths {0.40,0.50,0.60,0.67,0.75}. Winner rule pre-registered in
docs/causal_handle_directions.md item G."""
from __future__ import annotations
import argparse, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage2_qwen_patch_hf import (  # noqa: E402
    load_hf_model, torch_dtype, hidden_from_output, replace_hidden_in_output, validate_hf_layers)
from scripts.stage2_qwen_subspace_erasure import generate_one  # noqa: E402
from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
from scripts.stage2_hint_delta import concept_positions  # noqa: E402
from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
from scripts.stage2_rank_k_guard_v2 import select_fresh_rows  # noqa: E402
from scripts.stage2_qwen_g0_hf import G0_ROWS  # noqa: E402
from scripts.stage2_subtype_discriminator import json_default  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402

DEPTHS = (0.40, 0.50, 0.60, 0.67, 0.75)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--per-height", type=int, default=8)
    p.add_argument("--selection-seed", type=int, default=20260706)
    p.add_argument("--samples-per-row", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-block-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/qwen_g2_layersweep.jsonl"))
    p.add_argument("--output", type=Path, default=Path("docs/qwen_g2_layersweep.json"))
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)

    rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS), heights=[3, 4],
                             per_height=args.per_height, seed=args.selection_seed)
    print(f"selected {len(rows)} rows", flush=True)
    ct = {"enable_thinking": False}
    model, tokenizer = load_hf_model(args.model, dtype=torch_dtype(args.dtype),
        device_map="auto", device=None, attn_implementation="sdpa", trust_remote_code=True)
    n_layers = len(model.model.layers)
    layers = sorted({max(1, round(d * n_layers)) for d in DEPTHS})
    validate_hf_layers(model, layers)
    print(f"n_layers={n_layers} sweep_layers={layers}", flush=True)

    prepared = []
    for row in rows:
        gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
        rt = render_chat_text(tokenizer, system=row["system_prompt"], user=row["prompt_text"],
                              model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
        rids = tokenizer(rt, add_special_tokens=False)["input_ids"]
        ht = render_chat_text(tokenizer, system=row["system_prompt"],
                              user=make_user_prompt(row, "hint_concept_first"),
                              model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
        hids = tokenizer(ht, add_special_tokens=False)["input_ids"]
        h0, r0, blen = longest_common_token_block(hids, rids)
        if blen < args.min_block_tokens:
            print(f"skip {row['row_index']}: block={blen}", flush=True); continue
        pos_r = concept_positions(tokenizer, rt, gold, r0, blen)
        if not pos_r:
            print(f"skip {row['row_index']}: no concept positions", flush=True); continue
        rel = [q - r0 for q in pos_r]
        rng = random.Random(args.selection_seed + row["row_index"])
        rrel = sorted(rng.sample(range(blen), len(rel)))
        # capture hinted states at both position sets, all layers, one forward
        cap: dict[tuple[int, int], torch.Tensor] = {}
        need = sorted({h0 + q for q in rel} | {h0 + q for q in rrel})
        def mk(layer):
            def fn(_m, _i, out):
                h = hidden_from_output(out)
                if h.shape[1] >= len(hids):
                    for pp in need:
                        cap[(layer, pp)] = h[0, pp, :].detach().cpu()
            return fn
        handles = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]
        try:
            dev = next(model.parameters()).device
            with torch.inference_mode():
                model(input_ids=torch.tensor([hids], device=dev), use_cache=False)
        finally:
            for h in handles: h.remove()
        prepared.append({"row": row, "ri": row["row_index"], "gold": gold, "rids": rids,
                         "r0": r0, "h0": h0, "rel": rel, "rrel": rrel, "cap": cap})
        print(f"prepared {row['row_index']} ({len(rel)} pos)", flush=True)
    print(f"prepared_rows={len(prepared)}", flush=True)

    def replace_hook(layer, pos_state):
        def fn(_m, _i, out):
            h = hidden_from_output(out)
            if h.shape[1] > 1:
                for pp, st in pos_state.items():
                    if pp < h.shape[1]:
                        h[0, pp, :] = st.to(h.device, h.dtype)
                return replace_hidden_in_output(out, h)
            return out
        return fn

    arms = [("unhinted_baseline", None, None)] + [("hinted_baseline", None, None)] + \
        [(f"L{L}_concept_replace", L, "rel") for L in layers] + \
        [(f"L{L}_random_replace", L, "rrel") for L in layers]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as fout:
        for ai, (label, L, pk) in enumerate(arms):
            for prep in prepared:
                ids = prep["rids"]
                handles = []
                if label == "hinted_baseline":
                    ids = None  # rebuilt below
                    ht = render_chat_text(tokenizer, system=prep["row"]["system_prompt"],
                        user=make_user_prompt(prep["row"], "hint_concept_first"),
                        model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
                    ids = tokenizer(ht, add_special_tokens=False)["input_ids"]
                elif L is not None:
                    ps = {prep["r0"] + q: prep["cap"][(L, prep["h0"] + q)] for q in prep[pk]}
                    handles = [model.model.layers[L].register_forward_hook(replace_hook(L, ps))]
                try:
                    for s in range(args.samples_per_row):
                        torch.manual_seed(args.selection_seed + prep["ri"] * 10007 + ai * 101 + s)
                        new_ids, reply = generate_one(model=model, token_ids=ids,
                            max_new_tokens=args.max_new_tokens, do_sample=True,
                            temperature=args.temperature, stop_at_eos=True)
                        sc = score_reply(prep["row"], reply)
                        o = {"schema_version": 1, "source_row_index": prep["ri"],
                             "height": prep["row"].get("height"), "model": args.model,
                             "condition": label, "sample_index": s, "method": "qwen_g2_layersweep",
                             "patch_layer": L, "n_positions": len(prep[pk]) if pk else None,
                             "generated_token_count": len(new_ids), "model_output": reply, **sc}
                        fout.write(json.dumps(o, ensure_ascii=False, default=json_default) + "\n")
                finally:
                    for h in handles: h.remove()
                fout.flush()
            print(f"ARM DONE {label} ({time.time()-started:.0f}s)", flush=True)
    args.output.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started, "layers": layers, "n_layers": n_layers,
        "prepared_rows": len(prepared)}, indent=2) + "\n")
    print("done", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
