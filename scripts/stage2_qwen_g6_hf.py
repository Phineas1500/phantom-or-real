#!/usr/bin/env python3
"""Item G6 (HF pathway): Qwen content-transfer ladder, protocol-matched to Gemma F(ii).

96-row class-mean (stage-1 labels, no screen), per-position/per-row norm
targets, rank-16 and rank-64 projections, x2 dose, ~24 test rows. PRIMARY =
protomatched_proj16 CI excludes zero AND paired (proj16 - shuffled96) CI
excludes zero. Gates: unhinted + rank16 verbatim vs G3' on the shared 15
rows. Pre-registered: docs/causal_handle_directions.md item G6."""
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
from scripts.stage2_rank_k_guard_v2 import Arm, select_fresh_rows, class_vector_from_labels  # noqa: E402
from scripts.stage2_natural_state_capture import select_balanced_rows  # noqa: E402
from scripts.stage2_qwen_g0_hf import G0_ROWS  # noqa: E402
from scripts.stage2_subtype_discriminator import fit_pca_basis, rank_k_reconstruction, json_default  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402

RAND_DIR_DRAW_OFFSET = 90


def build_g6_arms(layer: int) -> list[tuple[Arm, int]]:
    """(arm, seed_ai) pairs; seed_ai 0/10 pin the G3' replication gates."""
    return [
        (Arm("unhinted_baseline", "none", "none"), 0),
        (Arm(f"rank16_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), 16, "leave_one_row_out"), 10),
        (Arm(f"protomatched_proj16_add_L{layer}", "proto_proj_add", "unhinted_baseline", (layer,), 16, "classmean96_rank_projected_perpos_norm"), 30),
        (Arm(f"protomatched_proj64_add_L{layer}", "proto_proj_add", "unhinted_baseline", (layer,), 64, "classmean96_rank_projected_perpos_norm"), 31),
        (Arm(f"protomatched_proj16_x2_L{layer}", "proto_proj_x2_add", "unhinted_baseline", (layer,), 16, "classmean96_rank_projected_perpos_norm_x2"), 32),
        (Arm(f"class_mean_raw96_add_L{layer}", "proto_raw_add", "unhinted_baseline", (layer,), 16, "classmean96_raw_perpos_norm"), 33),
        (Arm(f"shuffled96_proj16_L{layer}_d1", "proto_shuffled_add", "unhinted_baseline", (layer,), 16, "shuffled_label_classmean96_projected"), 34),
        (Arm(f"signflip96_proj16_L{layer}", "proto_signflip_add", "unhinted_baseline", (layer,), 16, "negated_classmean96_projected"), 35),
        (Arm(f"rand_norm_perpos_add_L{layer}_d1", "rand_dir_add", "unhinted_baseline", (layer,), 16, "random_direction_perpos_norm"), 36),
    ]


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> float:
    """5-fold logistic CV AUC (liblinear); numpy-rank AUC on pooled held-out scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores, labels = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(X[tr], y[tr])
        scores.append(clf.decision_function(X[te])); labels.append(y[te])
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(scores)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--layer", type=int, default=43)
    p.add_argument("--per-height", type=int, default=8)
    p.add_argument("--extra-per-height", type=int, default=5)
    p.add_argument("--source-per-cell", type=int, default=24)
    p.add_argument("--selection-seed", type=int, default=20260706)
    p.add_argument("--extra-seed", type=int, default=20260708)
    p.add_argument("--source-seed", type=int, default=20260708)
    p.add_argument("--control-seed", type=int, default=20260708)
    p.add_argument("--samples-per-row", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-block-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/qwen_g6_ladder.jsonl"))
    p.add_argument("--output", type=Path, default=Path("docs/qwen_g6_ladder.json"))
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)

    core_rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS), heights=[3, 4],
                                  per_height=args.per_height, seed=args.selection_seed)
    core_indices = {r["row_index"] for r in core_rows}
    extra_rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS) | core_indices, heights=[3, 4],
                                   per_height=args.extra_per_height, seed=args.extra_seed)
    test_rows = core_rows + extra_rows
    test_indices = {r["row_index"] for r in test_rows}
    source_rows = select_balanced_rows(args.jsonl, exclude=set(G0_ROWS) | test_indices,
                                       heights=[3, 4], per_cell=args.source_per_cell,
                                       seed=args.source_seed)
    print(f"test candidates: {len(core_rows)} core + {len(extra_rows)} extra; sources: {len(source_rows)}", flush=True)

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

    row_means, labels = {}, {}
    for row in source_rows:
        ri = row["row_index"]
        rt, rids = unhinted_ids(row)
        gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
        pos = concept_positions(tokenizer, rt, gold, 0, len(rids))
        if not pos:
            continue
        states = capture_states(rids, pos)
        row_means[ri] = np.stack([states[q].numpy() for q in pos]).astype(np.float64).mean(axis=0)
        labels[ri] = bool(row["is_correct_strong"])
    n_pos_lab = sum(labels.values()); n_neg_lab = len(labels) - n_pos_lab
    print(f"captured sources: {len(row_means)} ({n_pos_lab} correct / {n_neg_lab} incorrect)", flush=True)
    class_vector = class_vector_from_labels(row_means, labels)
    rng = np.random.default_rng(args.control_seed + 1)
    keys = sorted(row_means)
    vals = np.array([labels[k] for k in keys])
    shuffled_labels = dict(zip(keys, vals[rng.permutation(len(keys))]))
    shuffled_vector = class_vector_from_labels(row_means, shuffled_labels)

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
        is_core = row["row_index"] in core_indices
        if is_core:
            delta_by_row[row["row_index"]] = delta
        prepared.append({"row": row, "ri": row["row_index"], "rids": rids, "r0": r0,
                         "rel": rel, "delta": delta, "core": is_core})
        print(f"prepared test {row['row_index']} ({len(rel)} pos, {'core' if is_core else 'extra'})", flush=True)
    print(f"prepared_rows={len(prepared)} (core basis pool={len(delta_by_row)})", flush=True)

    basis_cache: dict[tuple[int, int], dict] = {}
    def basis_for(k: int, ri: int) -> dict:
        key = (k, ri)
        if key not in basis_cache:
            basis_cache[key] = fit_pca_basis(delta_by_row, k, exclude_rows={ri})
        return basis_cache[key]

    fi_analog = {}
    try:
        X = np.stack([row_means[k] for k in keys]); y = vals.astype(int)
        Xc = X - X.mean(axis=0, keepdims=True)
        fi_analog["full_dim"] = cv_auc(Xc, y)
        rng_fi = np.random.default_rng(args.control_seed + 5)
        for k in (16, 64):
            comps = fit_pca_basis(delta_by_row, k, exclude_rows=set())["components"]
            fi_analog[f"rank{k}_slice"] = cv_auc(Xc @ comps.T, y)
            null = [cv_auc(Xc @ np.linalg.qr(rng_fi.standard_normal((X.shape[1], k)))[0], y)
                    for _ in range(200)]
            fi_analog[f"rank{k}_null_median"] = float(np.median(null))
            fi_analog[f"rank{k}_null_p95"] = float(np.percentile(null, 95))
        print(f"F(i)-analog: {fi_analog}", flush=True)
    except Exception as exc:  # noqa: BLE001
        fi_analog = {"error": repr(exc)}
        print(f"F(i)-analog failed (descriptive rider only): {exc!r}", flush=True)

    def perpos_targets(prep: dict, k: int) -> np.ndarray:
        recon = rank_k_reconstruction(torch.from_numpy(prep["delta"]), basis_for(k, prep["ri"])).numpy()
        return np.linalg.norm(recon.astype(np.float64), axis=1, keepdims=True)

    def norm_matched(direction: np.ndarray, prep: dict, k: int, scale: float = 1.0) -> np.ndarray:
        tiled = np.tile(direction.astype(np.float64), (len(prep["rel"]), 1))
        current = np.maximum(np.linalg.norm(tiled, axis=1, keepdims=True), 1e-8)
        return (tiled * (perpos_targets(prep, k) * scale / current)).astype(np.float32)

    def arm_vector(arm: Arm, prep: dict) -> np.ndarray:
        if arm.kind == "rank_k_add":
            basis = basis_for(arm.rank_k, prep["ri"])
            return rank_k_reconstruction(torch.from_numpy(prep["delta"]), basis).numpy().astype(np.float32)
        if arm.kind == "rand_dir_add":
            draw = int(arm.label.rsplit("_d", 1)[1]) + RAND_DIR_DRAW_OFFSET
            rng_a = np.random.default_rng(args.control_seed + 7919 * draw + prep["ri"])
            return norm_matched(rng_a.standard_normal(class_vector.shape), prep, arm.rank_k)
        if arm.kind == "proto_raw_add":
            return norm_matched(class_vector, prep, arm.rank_k)
        if arm.kind == "proto_shuffled_add":
            source_vec = shuffled_vector
        elif arm.kind == "proto_signflip_add":
            source_vec = -class_vector
        else:
            source_vec = class_vector
        components = basis_for(arm.rank_k, prep["ri"])["components"]
        direction = (source_vec @ components.T) @ components
        scale = 2.0 if arm.kind == "proto_proj_x2_add" else 1.0
        return norm_matched(direction, prep, arm.rank_k, scale)

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

    arms = build_g6_arms(L)
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
                             "method": "qwen_g6_content_transfer", "patch_layer": L,
                             "seed_arm_index": seed_ai, "core_row": prep["core"],
                             "generated_token_count": len(new_ids), "model_output": reply, **sc}
                        fout.write(json.dumps(o, ensure_ascii=False, default=json_default) + "\n")
                finally:
                    for h in handles: h.remove()
                fout.flush()
            print(f"ARM DONE {arm.label} ({time.time()-started:.0f}s)", flush=True)

    args.output.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started, "layer": L,
        "prepared_rows": len(prepared), "core_rows": sorted(delta_by_row),
        "n_sources": len(row_means), "source_balance": [n_pos_lab, n_neg_lab],
        "class_vector_norm": float(np.linalg.norm(class_vector)),
        "fi_analog": fi_analog}, indent=2, default=json_default) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
