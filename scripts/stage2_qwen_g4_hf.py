#!/usr/bin/env python3
"""Item G4' (HF pathway): Qwen answer-free class-mean repair at L43/rank-16.

PRIMARY = class_mean_proj16 dP CI excludes zero AND paired (proj16 -
shuffled-label family) CI excludes zero. Replication gates: unhinted_baseline
(seed-ai 0) and rank16_loo (seed-ai 10) must reproduce job 458468 verbatim.
Pre-registered: docs/causal_handle_directions.md item G4'."""
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
from scripts.stage2_rank_k_guard_v2 import (  # noqa: E402
    Arm, select_fresh_rows, select_correct_rows, class_vector_from_labels, shuffled_label_vectors)
from scripts.stage2_qwen_g0_hf import G0_ROWS  # noqa: E402
from scripts.stage2_subtype_discriminator import fit_pca_basis, rank_k_reconstruction, json_default  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402

SCREEN_SEED_OFFSET = 777
RAND_DIR_DRAW_OFFSET = 80


def build_g4_arms(layer: int) -> list[tuple[Arm, int]]:
    """(arm, seed_ai) pairs; seed_ai 0/10 pin the G3' replication arms."""
    return [
        (Arm("unhinted_baseline", "none", "none"), 0),
        (Arm(f"rank16_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), 16, "leave_one_row_out"), 10),
        (Arm(f"class_mean_raw_add_L{layer}", "class_mean_raw_add", "unhinted_baseline", (layer,), 16, "natural_class_mean_pooled_norm"), 20),
        (Arm(f"class_mean_proj16_add_L{layer}", "class_mean_proj_add", "unhinted_baseline", (layer,), 16, "natural_class_mean_rank_projected_pooled_norm"), 21),
        (Arm(f"shuffled_label_proj16_L{layer}_d1", "shuflabel_proj_add", "unhinted_baseline", (layer,), 16, "shuffled_label_class_mean_projected"), 22),
        (Arm(f"shuffled_label_proj16_L{layer}_d2", "shuflabel_proj_add", "unhinted_baseline", (layer,), 16, "shuffled_label_class_mean_projected"), 23),
        (Arm(f"signflip_proj16_add_L{layer}", "signflip_proj_add", "unhinted_baseline", (layer,), 16, "negated_class_mean_projected"), 24),
        (Arm(f"rand_norm16_add_L{layer}_d1", "rand_dir_add", "unhinted_baseline", (layer,), 16, "random_direction_pooled_norm"), 25),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--layer", type=int, default=43)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--per-height", type=int, default=8)
    p.add_argument("--source-per-height", type=int, default=16)
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--selection-seed", type=int, default=20260706)
    p.add_argument("--source-seed", type=int, default=20260707)
    p.add_argument("--control-seed", type=int, default=20260707)
    p.add_argument("--screen-samples", type=int, default=4)
    p.add_argument("--samples-per-row", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-block-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/qwen_g4_classmean.jsonl"))
    p.add_argument("--output", type=Path, default=Path("docs/qwen_g4_classmean.json"))
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)

    test_rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS), heights=[3, 4],
                                  per_height=args.per_height, seed=args.selection_seed)
    test_indices = {r["row_index"] for r in test_rows}
    source_exclude = set(G0_ROWS) | test_indices
    cand_correct = select_correct_rows(args.jsonl, exclude=source_exclude, heights=[3, 4],
                                       per_height=args.source_per_height, seed=args.source_seed)
    cand_incorrect = select_fresh_rows(args.jsonl, exclude=source_exclude, heights=[3, 4],
                                       per_height=args.source_per_height, seed=args.source_seed)
    print(f"candidates: {len(cand_correct)} correct / {len(cand_incorrect)} incorrect", flush=True)

    ct = {"enable_thinking": False}
    model, tokenizer = load_hf_model(args.model, dtype=torch_dtype(args.dtype),
        device_map="auto", device=None, attn_implementation="sdpa", trust_remote_code=True)
    validate_hf_layers(model, [args.layer])
    L = args.layer
    dev = next(model.parameters()).device

    def unhinted_ids(row):
        rt = render_chat_text(tokenizer, system=row["system_prompt"], user=row["prompt_text"],
                              model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
        return rt, tokenizer(rt, add_special_tokens=False)["input_ids"]

    def capture_states(ids, positions):
        store = {}
        def fn(_m, _i, out):
            h = hidden_from_output(out)
            if h.shape[1] >= len(ids):
                for pp in positions: store[pp] = h[0, pp, :].detach().cpu().float()
        hd = model.model.layers[L].register_forward_hook(fn)
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally: hd.remove()
        return store

    def screen_and_capture(cands, want_correct):
        confirmed, row_means, labels = [], {}, {}
        for row in cands:
            if len(confirmed) >= args.per_class:
                break
            ri = row["row_index"]
            rt, rids = unhinted_ids(row)
            gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
            pos = concept_positions(tokenizer, rt, gold, 0, len(rids))
            if not pos:
                continue
            n_strong = 0
            for s in range(args.screen_samples):
                torch.manual_seed(args.source_seed + ri * 10007 + SCREEN_SEED_OFFSET + s)
                _, reply = generate_one(model=model, token_ids=rids,
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, stop_at_eos=True)
                n_strong += bool(score_reply(row, reply)["is_correct_strong"])
            majority = None
            if n_strong >= 3: majority = True
            elif n_strong <= 1: majority = False
            if majority is None or majority != want_correct:
                continue
            states = capture_states(rids, pos)
            row_means[ri] = np.stack([states[q].numpy() for q in pos]).astype(np.float64).mean(axis=0)
            labels[ri] = want_correct
            confirmed.append(ri)
            print(f"confirmed {'correct' if want_correct else 'incorrect'} {ri} ({n_strong}/4, {len(pos)} pos)", flush=True)
        return confirmed, row_means, labels

    conf_c, means_c, labels_c = screen_and_capture(cand_correct, True)
    conf_i, means_i, labels_i = screen_and_capture(cand_incorrect, False)
    row_means = {**means_c, **means_i}
    labels = {**labels_c, **labels_i}
    print(f"confirmed_source_rows: {len(conf_c)} correct / {len(conf_i)} incorrect", flush=True)
    class_vector = class_vector_from_labels(row_means, labels)
    shuf_vectors = shuffled_label_vectors(row_means, labels, draws=2, seed=args.control_seed)

    prepared, delta_by_row = [], {}
    for row in test_rows:
        gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
        rt, rids = unhinted_ids(row)
        ht = render_chat_text(tokenizer, system=row["system_prompt"],
                              user=make_user_prompt(row, "hint_concept_first"),
                              model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
        hids = tokenizer(ht, add_special_tokens=False)["input_ids"]
        h0, r0, blen = longest_common_token_block(hids, rids)
        if blen < args.min_block_tokens: continue
        pos_r = concept_positions(tokenizer, rt, gold, r0, blen)
        if not pos_r: continue
        rel = [q - r0 for q in pos_r]
        hs = capture_states(hids, [h0 + q for q in rel])
        us = capture_states(rids, [r0 + q for q in rel])
        delta = np.stack([(hs[h0 + q] - us[r0 + q]).numpy() for q in rel]).astype(np.float32)
        delta_by_row[row["row_index"]] = delta
        prepared.append({"row": row, "ri": row["row_index"], "rids": rids, "r0": r0,
                         "rel": rel, "delta": delta})
        print(f"prepared test {row['row_index']} ({len(rel)} pos)", flush=True)
    print(f"prepared_rows={len(prepared)}", flush=True)

    basis_cache: dict[int, dict] = {}
    def loo_basis(ri: int) -> dict:
        if ri not in basis_cache:
            basis_cache[ri] = fit_pca_basis(delta_by_row, args.rank, exclude_rows={ri})
        return basis_cache[ri]

    recon_norms = []
    for prep in prepared:
        recon = rank_k_reconstruction(torch.from_numpy(prep["delta"]), loo_basis(prep["ri"])).numpy()
        recon_norms.append(float(np.linalg.norm(recon, axis=1).mean()))
    pooled_norm = float(np.mean(recon_norms))
    print(f"pooled_norm={pooled_norm:.3f} class_vector_norm={np.linalg.norm(class_vector):.3f}", flush=True)

    def arm_vector(arm: Arm, prep: dict) -> np.ndarray:
        n_pos = len(prep["rel"])
        if arm.kind == "rank_k_add":
            basis = loo_basis(prep["ri"])
            return rank_k_reconstruction(torch.from_numpy(prep["delta"]), basis).numpy().astype(np.float32)
        if arm.kind == "rand_dir_add":
            draw = int(arm.label.rsplit("_d", 1)[1]) + RAND_DIR_DRAW_OFFSET
            rng = np.random.default_rng(args.control_seed + 7919 * draw + prep["ri"])
            direction = rng.standard_normal(class_vector.shape)
        elif arm.kind == "class_mean_raw_add":
            direction = class_vector
        else:
            if arm.kind == "shuflabel_proj_add":
                source_vec = shuf_vectors[int(arm.label.rsplit("_d", 1)[1]) - 1]
            elif arm.kind == "signflip_proj_add":
                source_vec = -class_vector
            else:
                source_vec = class_vector
            components = loo_basis(prep["ri"])["components"]
            direction = (source_vec @ components.T) @ components
        tiled = np.tile(direction.astype(np.float64), (n_pos, 1))
        current = np.maximum(np.linalg.norm(tiled, axis=1, keepdims=True), 1e-8)
        return (tiled * (pooled_norm / current)).astype(np.float32)

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

    arms = build_g4_arms(L)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as fout:
        for arm, seed_ai in arms:
            for prep in prepared:
                handles = []
                if arm.kind != "none":
                    vec = arm_vector(arm, prep)
                    positions = [prep["r0"] + q for q in prep["rel"]]
                    handles = [model.model.layers[L].register_forward_hook(add_hook(vec, positions))]
                try:
                    for s in range(args.samples_per_row):
                        torch.manual_seed(args.selection_seed + prep["ri"] * 10007 + seed_ai * 101 + s)
                        new_ids, reply = generate_one(model=model, token_ids=prep["rids"],
                            max_new_tokens=args.max_new_tokens, do_sample=True,
                            temperature=args.temperature, stop_at_eos=True)
                        sc = score_reply(prep["row"], reply)
                        o = {"schema_version": 1, "source_row_index": prep["ri"],
                             "height": prep["row"].get("height"), "model": args.model,
                             "condition": arm.label, "sample_index": s,
                             "method": "qwen_g4_answer_free_classmean", "patch_layer": L,
                             "seed_arm_index": seed_ai,
                             "generated_token_count": len(new_ids), "model_output": reply, **sc}
                        fout.write(json.dumps(o, ensure_ascii=False, default=json_default) + "\n")
                finally:
                    for h in handles: h.remove()
                fout.flush()
            print(f"ARM DONE {arm.label} ({time.time()-started:.0f}s)", flush=True)

    args.output.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started, "layer": L, "rank": args.rank,
        "prepared_rows": len(prepared), "confirmed_correct": conf_c, "confirmed_incorrect": conf_i,
        "pooled_norm": pooled_norm, "class_vector_norm": float(np.linalg.norm(class_vector))},
        indent=2, default=json_default) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
