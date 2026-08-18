#!/usr/bin/env python3
"""Item N (HF pathway): the closed answer-free loop ported to Qwen3.5-27B.

--calibration (N0): fit + freeze the final-token L43 gauge (64 balanced rows,
seed 20260819), pin the answer-free scalar write amplitude (mean rank-16 recon
norm over the 12 gate rows), and run the registered gates (natural AUC >= 0.65,
selection signal, delivery) on the 12 seeded failing gate rows.
--shard-index/--shard-count (N1): the composition on 96 fresh failing rows
(seed 20260820) — k=8 baselines + per-candidate raw-class-mean fires at L43
with the frozen gauge scoring each steered branch's final-token state.
Pre-registered: docs/causal_handle_directions.md item N + amendment."""
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
    select_fresh_rows, class_vector_from_labels, all_concept_names)
from scripts.stage2_natural_state_capture import select_balanced_rows  # noqa: E402
from scripts.stage2_qwen_g0_hf import G0_ROWS  # noqa: E402
from scripts.stage2_subtype_discriminator import fit_pca_basis, rank_k_reconstruction, json_default  # noqa: E402
from scripts.stage2_interchange_concept_analysis import canon, subjects_of  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402

GAUGE_NPZ = Path("results/stage2/erasure/qwen_loop_gauge_L43final.npz")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--layer", type=int, default=43)
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=6)
    p.add_argument("--cal-per-cell", type=int, default=16)
    p.add_argument("--cal-seed", type=int, default=20260819)
    p.add_argument("--test-seed", type=int, default=20260820)
    p.add_argument("--test-per-height", type=int, default=48)
    p.add_argument("--source-per-cell", type=int, default=24)
    p.add_argument("--source-seed", type=int, default=20260708)
    p.add_argument("--samples-per-row", type=int, default=8)
    p.add_argument("--percand-samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-block-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)
    L = args.layer
    stem = ("qwen_loop_calibration" if args.calibration
            else f"qwen_loop_27b_property_shard{args.shard_index}of{args.shard_count}")
    out_jsonl = Path(f"results/stage2/erasure/{stem}.jsonl")
    out_json = Path(f"docs/{stem}.json")

    ct = {"enable_thinking": False}
    model, tokenizer = load_hf_model(args.model, dtype=torch_dtype(args.dtype),
        device_map="auto", device=None, attn_implementation="sdpa", trust_remote_code=True)
    validate_hf_layers(model, [L])
    dev = next(model.parameters()).device

    def unhinted_ids(row):
        rt = render_chat_text(tokenizer, system=row["system_prompt"], user=row["prompt_text"],
                              model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
        return rt, tokenizer(rt, add_special_tokens=False)["input_ids"]

    def capture_states(ids, positions, inject=None):
        """inject: (positions, matrix) added at L during the same prefill."""
        store = {}
        def cap_fn(_m, _i, out):
            h = hidden_from_output(out)
            if h.shape[1] >= len(ids):
                if inject is not None:
                    ipos, mat = inject
                    for j, pp in enumerate(ipos):
                        if pp < h.shape[1]:
                            h[0, pp, :] += torch.from_numpy(mat[j]).to(h.device, h.dtype)
                for pp in positions:
                    store[pp] = h[0, pp, :].detach().cpu().float()
                if inject is not None:
                    return replace_hidden_in_output(out, h)
            return out
        hd = model.model.layers[L].register_forward_hook(cap_fn)
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            hd.remove()
        return store

    source_rows = select_balanced_rows(args.jsonl, exclude=set(G0_ROWS), heights=[3, 4],
                                       per_cell=args.source_per_cell, seed=args.source_seed)
    source_idx = {r["row_index"] for r in source_rows}
    row_means, labels = {}, {}
    for row in source_rows:
        rt, rids = unhinted_ids(row)
        gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
        pos = concept_positions(tokenizer, rt, gold, 0, len(rids))
        if not pos:
            continue
        st = capture_states(rids, pos)
        row_means[row["row_index"]] = np.stack([st[q].numpy() for q in pos]).astype(np.float64).mean(axis=0)
        labels[row["row_index"]] = bool(row["is_correct_strong"])
    class_vector = class_vector_from_labels(row_means, labels)
    print(f"sources captured: {len(row_means)} |class_vector|={np.linalg.norm(class_vector):.1f}", flush=True)

    def gate_rows_select(exclude):
        return select_fresh_rows(args.jsonl, exclude=exclude, heights=[3, 4],
                                 per_height=6, seed=args.cal_seed)

    def prep_row(row, need_delta):
        gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
        rt, rids = unhinted_ids(row)
        cands = {}
        for name in all_concept_names(row):
            cp = concept_positions(tokenizer, rt, name, 0, len(rids))
            if cp:
                cands[name] = cp
        if gold not in cands and not any(canon(n) == canon(gold) for n in cands):
            return None
        prep = {"row": row, "ri": row["row_index"], "rt": rt, "rids": rids,
                "gold": gold, "cands": dict(sorted(cands.items()))}
        if need_delta:
            ht = render_chat_text(tokenizer, system=row["system_prompt"],
                                  user=make_user_prompt(row, "hint_concept_first"),
                                  model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
            hids = tokenizer(ht, add_special_tokens=False)["input_ids"]
            h0, r0, blen = longest_common_token_block(hids, rids)
            if blen < args.min_block_tokens:
                return None
            pos_r = concept_positions(tokenizer, rt, gold, r0, blen)
            if not pos_r:
                return None
            rel = [q - r0 for q in pos_r]
            hs = capture_states(hids, [h0 + q for q in rel])
            us = capture_states(rids, [r0 + q for q in rel])
            prep["delta"] = np.stack([(hs[h0 + q] - us[r0 + q]).numpy() for q in rel]).astype(np.float32)
        return prep

    def write_matrix(positions, norm_target):
        tiled = np.tile(class_vector.astype(np.float64), (len(positions), 1))
        cur = np.maximum(np.linalg.norm(tiled, axis=1, keepdims=True), 1e-8)
        return (tiled * (norm_target / cur)).astype(np.float32)

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

    def emit(fout, prep, condition, sample_index, reply, new_ids, extra):
        sc = score_reply(prep["row"], reply)
        subs = subjects_of(reply)
        o = {"schema_version": 1, "source_row_index": prep["ri"],
             "height": prep["row"].get("height"), "model": args.model,
             "condition": condition, "sample_index": sample_index,
             "method": "qwen_closed_loop", "patch_layer": L,
             "targets_gold_concept": canon(prep["gold"]) in subs,
             "generated_token_count": len(new_ids), "model_output": reply, **extra, **sc}
        if "fired_concept" in extra:
            o["targets_fired_concept"] = canon(extra["fired_concept"]) in subs
        fout.write(json.dumps(o, ensure_ascii=False, default=json_default) + "\n")
        return o

    def run_branches(fout, prep, gauge, norm_target, base_seed_row):
        b_state = capture_states(prep["rids"], [len(prep["rids"]) - 1])
        b_score = float(gauge["w"] @ (b_state[len(prep["rids"]) - 1].numpy() - gauge["mean"]) + gauge["b"])
        for s in range(args.samples_per_row):
            torch.manual_seed(base_seed_row + 0 * 101 + s)
            new_ids, reply = generate_one(model=model, token_ids=prep["rids"],
                max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, stop_at_eos=True)
            emit(fout, prep, "unhinted_baseline", s, reply, new_ids, {"base_gauge_score": b_score})
        for ci, (cand, cpos) in enumerate(prep["cands"].items()):
            mat = write_matrix(cpos, norm_target)
            s_state = capture_states(prep["rids"], [len(prep["rids"]) - 1], inject=(cpos, mat))
            s_score = float(gauge["w"] @ (s_state[len(prep["rids"]) - 1].numpy() - gauge["mean"]) + gauge["b"])
            fired_is_gold = canon(cand) == canon(prep["gold"])
            handles = [model.model.layers[L].register_forward_hook(add_hook(mat, cpos))]
            try:
                for s in range(args.percand_samples):
                    torch.manual_seed(base_seed_row + 95 * 101 + ci * 13 + s)
                    new_ids, reply = generate_one(model=model, token_ids=prep["rids"],
                        max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=args.temperature, stop_at_eos=True)
                    emit(fout, prep, f"percand_raw_fire_L{L}", s, reply, new_ids,
                         {"fired_concept": cand, "fired_is_gold": fired_is_gold,
                          "fired_candidate_index": ci, "n_fired_positions": len(cpos),
                          "gauge_score": s_score, "base_gauge_score": b_score})
            finally:
                for h in handles:
                    h.remove()
        fout.flush()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "layer": L,
              "model": args.model, "n_sources": len(row_means),
              "class_vector_norm": float(np.linalg.norm(class_vector)),
              "pre_registered_decision_rule": "docs/causal_handle_directions.md item N + amendment"}

    if args.calibration:
        cal_rows = select_balanced_rows(args.jsonl, exclude=set(G0_ROWS) | source_idx,
                                        heights=[3, 4], per_cell=args.cal_per_cell, seed=args.cal_seed)
        X, y = [], []
        for row in cal_rows:
            _, rids = unhinted_ids(row)
            st = capture_states(rids, [len(rids) - 1])
            X.append(st[len(rids) - 1].numpy())
            y.append(int(bool(row["is_correct_strong"])))
        X = np.stack(X).astype(np.float64); y = np.array(y)
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        mean = X.mean(axis=0)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260819)
        scores, labs = [], []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
            clf.fit(X[tr] - mean, y[tr])
            scores.append(clf.decision_function(X[te] - mean)); labs.append(y[te])
        cv = float(roc_auc_score(np.concatenate(labs), np.concatenate(scores)))
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear").fit(X - mean, y)
        gauge = {"w": clf.coef_[0].astype(np.float64), "b": float(clf.intercept_[0]), "mean": mean}
        report["natural_gate"] = {"cv_auc": cv, "n_cal_rows": len(y), "pass": bool(cv >= 0.65)}
        print(f"NATURAL GATE: cv_auc={cv:.4f} ({'PASS' if cv >= 0.65 else 'FAIL'})", flush=True)

        gate_excl = set(G0_ROWS) | source_idx | {r["row_index"] for r in cal_rows}
        gate_rows = gate_rows_select(gate_excl)
        preps, deltas = [], {}
        for row in gate_rows:
            pr = prep_row(row, need_delta=True)
            if pr is not None:
                preps.append(pr); deltas[pr["ri"]] = pr["delta"]
        basis = fit_pca_basis(deltas, 16, exclude_rows=set())
        norms = [float(np.linalg.norm(
            rank_k_reconstruction(torch.from_numpy(pr["delta"]), basis).numpy().astype(np.float64),
            axis=1).mean()) for pr in preps]
        norm_target = float(np.mean(norms))
        report["pinned_norm_target"] = norm_target
        report["gate_rows"] = [pr["ri"] for pr in preps]
        print(f"pinned answer-free norm target: {norm_target:.4f} over {len(preps)} gate rows", flush=True)

        with out_jsonl.open("w") as fout:
            for pr in preps:
                run_branches(fout, pr, gauge, norm_target,
                             args.cal_seed + pr["ri"] * 10007)
                print(f"gate row {pr['ri']} done ({time.time()-started:.0f}s)", flush=True)
        np.savez(GAUGE_NPZ, w=gauge["w"], b=np.array([gauge["b"]]), mean=gauge["mean"],
                 cv_auc=np.array([cv]), norm_target=np.array([norm_target]))
        report["gauge_npz"] = str(GAUGE_NPZ)
    else:
        g = np.load(GAUGE_NPZ)
        gauge = {"w": g["w"], "b": float(g["b"][0]), "mean": g["mean"]}
        norm_target = float(g["norm_target"][0])
        cal_rows = select_balanced_rows(args.jsonl, exclude=set(G0_ROWS) | source_idx,
                                        heights=[3, 4], per_cell=args.cal_per_cell, seed=args.cal_seed)
        gate_excl = set(G0_ROWS) | source_idx | {r["row_index"] for r in cal_rows}
        gate_idx = {r["row_index"] for r in gate_rows_select(gate_excl)}
        test_all = select_fresh_rows(args.jsonl, exclude=gate_excl | gate_idx, heights=[3, 4],
                                     per_height=args.test_per_height, seed=args.test_seed)
        shard = [r for i, r in enumerate(test_all) if i % args.shard_count == args.shard_index]
        print(f"N1 shard {args.shard_index}/{args.shard_count}: {[r['row_index'] for r in shard]}", flush=True)
        report["shard_rows"] = [r["row_index"] for r in shard]
        report["pinned_norm_target"] = norm_target
        with out_jsonl.open("w") as fout:
            for row in shard:
                pr = prep_row(row, need_delta=False)
                if pr is None:
                    print(f"skip {row['row_index']}: no usable candidates", flush=True)
                    continue
                run_branches(fout, pr, gauge, norm_target,
                             args.test_seed + pr["ri"] * 10007)
                print(f"row {pr['ri']} done: {len(pr['cands'])} candidates ({time.time()-started:.0f}s)", flush=True)

    report["elapsed_seconds"] = time.time() - started
    out_json.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
