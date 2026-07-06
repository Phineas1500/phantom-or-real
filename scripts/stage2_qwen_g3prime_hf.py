#!/usr/bin/env python3
"""Item G3' (HF pathway): Qwen rank-and-scale ladder at L43.

Rank branch: winner = smallest k in {8,16,32,64} whose dP CI excludes zero
AND whose paired (rank_k - rand_subspace64 family) CI excludes zero.
Scale branch: rank8_fixednorm vs rand_normfull family. Replication gates:
unhinted_baseline (seed-ai 0) and rank8_loo (seed-ai 2) must reproduce job
458465 exactly. Pre-registered: docs/causal_handle_directions.md item G3'."""
from __future__ import annotations
import argparse, json, sys, time
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
from scripts.stage2_rank_k_guard_v2 import Arm, select_fresh_rows, control_add_matrix  # noqa: E402
from scripts.stage2_qwen_g0_hf import G0_ROWS  # noqa: E402
from scripts.stage2_subtype_discriminator import fit_pca_basis, rank_k_reconstruction, json_default  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402

RAND_NORMFULL_DRAW_OFFSET = 50


def build_g3prime_arms(layer: int) -> list[tuple[Arm, int]]:
    """(arm, seed_ai) pairs; seed_ai 0/2 pin the G3 replication arms."""
    return [
        (Arm("unhinted_baseline", "none", "none"), 0),
        (Arm(f"rank8_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), 8, "leave_one_row_out"), 2),
        (Arm(f"rank16_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), 16, "leave_one_row_out"), 10),
        (Arm(f"rank32_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), 32, "leave_one_row_out"), 11),
        (Arm(f"rank64_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), 64, "leave_one_row_out"), 12),
        (Arm(f"rank8_fixednorm_add_L{layer}", "rank_fixednorm_add", "unhinted_baseline", (layer,), 8, "leave_one_row_out"), 13),
        (Arm(f"rand_subspace64_add_L{layer}_d1", "rand_subspace_add", "unhinted_baseline", (layer,), 64, "random_orthonormal"), 14),
        (Arm(f"rand_subspace64_add_L{layer}_d2", "rand_subspace_add", "unhinted_baseline", (layer,), 64, "random_orthonormal"), 15),
        (Arm(f"rand_normfull_add_L{layer}_d1", "rand_norm_full_add", "unhinted_baseline", (layer,), None, "full_delta_norm_gaussian"), 16),
        (Arm(f"rand_normfull_add_L{layer}_d2", "rand_norm_full_add", "unhinted_baseline", (layer,), None, "full_delta_norm_gaussian"), 17),
    ]


def rescale_rows_to(target: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    want = np.linalg.norm(target, axis=1, keepdims=True)
    have = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    return (matrix * (want / have)).astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--layer", type=int, default=43)
    p.add_argument("--per-height", type=int, default=8)
    p.add_argument("--selection-seed", type=int, default=20260706)
    p.add_argument("--control-seed", type=int, default=20260706)
    p.add_argument("--samples-per-row", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-block-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/qwen_g3prime_ladder.jsonl"))
    p.add_argument("--output", type=Path, default=Path("docs/qwen_g3prime_ladder.json"))
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)

    rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS), heights=[3, 4],
                             per_height=args.per_height, seed=args.selection_seed)
    ct = {"enable_thinking": False}
    model, tokenizer = load_hf_model(args.model, dtype=torch_dtype(args.dtype),
        device_map="auto", device=None, attn_implementation="sdpa", trust_remote_code=True)
    validate_hf_layers(model, [args.layer])
    L = args.layer

    prepared, delta_by_row = [], {}
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
        if blen < args.min_block_tokens: continue
        pos_r = concept_positions(tokenizer, rt, gold, r0, blen)
        if not pos_r: continue
        rel = [q - r0 for q in pos_r]
        def cap_states(ids, positions):
            store = {}
            def fn(_m, _i, out):
                h = hidden_from_output(out)
                if h.shape[1] >= len(ids):
                    for pp in positions: store[pp] = h[0, pp, :].detach().cpu().float()
            hd = model.model.layers[L].register_forward_hook(fn)
            try:
                dev = next(model.parameters()).device
                with torch.inference_mode():
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
            finally: hd.remove()
            return store
        hs = cap_states(hids, [h0 + q for q in rel])
        us = cap_states(rids, [r0 + q for q in rel])
        delta = np.stack([(hs[h0 + q] - us[r0 + q]).numpy() for q in rel]).astype(np.float32)
        delta_by_row[row["row_index"]] = delta
        prepared.append({"row": row, "ri": row["row_index"], "rids": rids, "r0": r0,
                         "rel": rel, "delta": delta})
        print(f"prepared {row['row_index']} ({len(rel)} pos)", flush=True)
    print(f"prepared_rows={len(prepared)}", flush=True)

    basis_cache: dict[tuple[int, int], dict] = {}
    def loo_basis(k: int, ri: int) -> dict:
        key = (k, ri)
        if key not in basis_cache:
            basis_cache[key] = fit_pca_basis(delta_by_row, k, exclude_rows={ri})
        return basis_cache[key]

    def arm_vector(arm: Arm, prep: dict, q_cache: dict) -> np.ndarray:
        delta, ri = prep["delta"], prep["ri"]
        if arm.kind in ("rank_k_add", "rank_fixednorm_add"):
            basis = loo_basis(arm.rank_k, ri)
            recon = rank_k_reconstruction(torch.from_numpy(delta), basis).numpy().astype(np.float32)
            if arm.kind == "rank_k_add":
                return recon
            return rescale_rows_to(delta, recon)
        if arm.kind == "rand_subspace_add":
            basis = loo_basis(arm.rank_k, ri)
            recon = rank_k_reconstruction(torch.from_numpy(delta), basis).numpy().astype(np.float32)
            return control_add_matrix(arm, delta, basis, recon,
                control_seed=args.control_seed, shard_index=0,
                source_row_index=ri, q_cache=q_cache)
        if arm.kind == "rand_norm_full_add":
            draw = int(arm.label.rsplit("_d", 1)[1]) + RAND_NORMFULL_DRAW_OFFSET
            rng = np.random.default_rng(args.control_seed + 7919 * draw + ri)
            noise = rng.standard_normal(delta.shape).astype(np.float32)
            return rescale_rows_to(delta, noise)
        raise ValueError(f"unhandled arm kind: {arm.kind}")

    def add_hook(vec_np, positions):
        vec = torch.from_numpy(vec_np)
        def fn(_m, _i, out):
            h = hidden_from_output(out)
            if h.shape[1] > 1:
                for j, pp in enumerate(positions):
                    if pp < h.shape[1]:
                        h[0, pp, :] += vec[j].to(h.device, h.dtype)
                return replace_hidden_in_output(out, h)
            return out
        return fn

    arms = build_g3prime_arms(L)
    q_cache: dict[int, np.ndarray] = {}
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as fout:
        for arm, seed_ai in arms:
            for prep in prepared:
                ids = prep["rids"]; handles = []
                if arm.kind != "none":
                    vec = arm_vector(arm, prep, q_cache)
                    positions = [prep["r0"] + q for q in prep["rel"]]
                    handles = [model.model.layers[L].register_forward_hook(add_hook(vec, positions))]
                try:
                    for s in range(args.samples_per_row):
                        torch.manual_seed(args.selection_seed + prep["ri"] * 10007 + seed_ai * 101 + s)
                        new_ids, reply = generate_one(model=model, token_ids=ids,
                            max_new_tokens=args.max_new_tokens, do_sample=True,
                            temperature=args.temperature, stop_at_eos=True)
                        sc = score_reply(prep["row"], reply)
                        o = {"schema_version": 1, "source_row_index": prep["ri"],
                             "height": prep["row"].get("height"), "model": args.model,
                             "condition": arm.label, "sample_index": s,
                             "method": "qwen_g3prime_rank_scale_ladder", "patch_layer": L,
                             "seed_arm_index": seed_ai,
                             "generated_token_count": len(new_ids), "model_output": reply, **sc}
                        fout.write(json.dumps(o, ensure_ascii=False, default=json_default) + "\n")
                finally:
                    for h in handles: h.remove()
                fout.flush()
            print(f"ARM DONE {arm.label} ({time.time()-started:.0f}s)", flush=True)

    effective_ranks = {str(k): min(b["effective_rank_k"] for (kk, _), b in basis_cache.items() if kk == k)
                       for k in {kk for (kk, _) in basis_cache}}
    args.output.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started, "layer": L,
        "prepared_rows": len(prepared), "effective_ranks_min": effective_ranks},
        indent=2, default=json_default) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
