#!/usr/bin/env python3
"""Target-concept hint-state interchange on the recognition-gap rowset.

Donors: hint-first prompts (concept hint before the ontology context), with
behaviorally validated effects (right-concept hint repairs at 1.000,
wrong-concept hint misdirects to 0.000). Receiver: the unhinted free-form
prompt. Patches the donor's hint-conditioned context-block residuals into the
receiver at matched positions and asks whether bidirectional control of
`target_concept` transfers through activations. See
docs/causal_handle_directions.md experiment 7 (re-aimed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import (  # noqa: E402
    generate_one,
    json_default,
    torch_dtype,
)
from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
from scripts.stage2_proposal_hints import make_user_prompt, wrong_concept  # noqa: E402
from scripts.stage2_recognition_state_patch import (  # noqa: E402
    longest_common_token_block,
    make_patch_hook,
    run_donor,
    summarize_patch_rows,
)
from src.activations import (  # noqa: E402
    input_device_for_model,
    load_tl_model,
    render_chat_text,
    validate_hooks,
)
from src.bd_path import ensure_on_path  # noqa: E402
from src.gemma3_parse import parse_hypotheses  # noqa: E402
from src.stage2_steering import parse_int_list, score_reply  # noqa: E402


@dataclass(frozen=True)
class InterchangeCondition:
    label: str
    donor: str | None
    patch_kind: str | None


CONDITION_PLAN = (
    InterchangeCondition("baseline", None, None),
    InterchangeCondition("patch_hint_gold", "gold", "patch_recognition"),
    InterchangeCondition("patch_hint_wrong", "wrong", "patch_recognition"),
    InterchangeCondition("patch_shuffled", "gold", "patch_shuffled"),
    InterchangeCondition("noise_matched", "gold", "noise_matched"),
)


def generate_sample_batch(
    *,
    model,
    token_ids: list[int],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_at_eos: bool,
    cache_dtype: torch.dtype,
) -> list[tuple[list[int], str]]:
    """Sample n completions of one prompt in a single batched generate call.

    The batch rows share an identical prompt, so no padding or attention-mask
    handling is needed; patch hooks broadcast over the batch dimension.
    """
    tokenizer = model.tokenizer
    device = input_device_for_model(model)
    tokens = torch.tensor([token_ids] * n_samples, dtype=torch.long, device=device)
    previous_default_dtype = torch.get_default_dtype()
    with torch.inference_mode():
        try:
            torch.set_default_dtype(cache_dtype)
            output_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                stop_at_eos=stop_at_eos,
                do_sample=True,
                temperature=temperature,
                prepend_bos=False,
                return_type="tokens",
                verbose=False,
                use_past_kv_cache=True,
            )
        finally:
            torch.set_default_dtype(previous_default_dtype)
    results = []
    for sample_row in output_tokens:
        new_ids = sample_row.detach().cpu().tolist()[len(token_ids):]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append((new_ids, reply))
    return results


def concept_mention_fields(reply: str, gold_concept: str, wrong: str) -> dict[str, Any]:
    hyps = " ".join(parse_hypotheses(reply or "")).lower()
    return {
        "output_mentions_gold_concept": gold_concept.lower().rstrip("s") in hyps,
        "output_mentions_wrong_concept": wrong.lower().rstrip("s") in hyps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--prefix-trajectory-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--prefix-checkpoint", default="0")
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="30,40,45")
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260611)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/hint_state_interchange_27b_property_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/hint_state_interchange_27b_property_manifest.json"))
    args = parser.parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    if args.samples_per_row > 1 and not args.do_sample:
        raise ValueError("--samples-per-row > 1 requires --do-sample")
    layers = parse_int_list(args.layers)
    dtype = torch_dtype(args.dtype)

    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=0.0,
    )
    total = len(selected_rows) * len(CONDITION_PLAN) * args.samples_per_row
    print(f"selected_rows={len(selected_rows)} total_generations={total}", flush=True)
    if args.dry_run:
        print(json.dumps({"selection": selection_summary, "layers": layers,
                          "conditions": [c.label for c in CONDITION_PLAN],
                          "total_generations": total}, indent=2, default=json_default), flush=True)
        return 0

    ensure_on_path()
    model = load_tl_model(args.model, n_devices=args.n_devices, n_ctx=args.n_ctx,
                          dtype=dtype, load_mode=args.load_mode)
    hook_names = validate_hooks(model, layers)
    hook_name_by_layer = dict(zip(layers, hook_names))
    tokenizer = model.tokenizer
    print(f"using_hooks={hook_name_by_layer}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            source_row_index = int(stage1_row["row_index"])
            gold_concept = stage1_row["ontology_fol_structured"]["hypothesis"]["subject"]
            wrong = wrong_concept(stage1_row)
            receiver_text = render_chat_text(
                tokenizer, system=stage1_row["system_prompt"], user=stage1_row["prompt_text"],
                model_name=args.model, add_generation_prompt=True)
            receiver_ids = tokenizer(receiver_text, add_special_tokens=False)["input_ids"]

            donor_blocks: dict[str, dict[int, torch.Tensor]] = {}
            receiver_starts: dict[str, int] = {}
            ok = True
            for donor_key, condition_name in (("gold", "hint_concept_first"), ("wrong", "hint_wrong_concept_first")):
                donor_user = make_user_prompt(stage1_row, condition_name)
                donor_text = render_chat_text(
                    tokenizer, system=stage1_row["system_prompt"], user=donor_user,
                    model_name=args.model, add_generation_prompt=True)
                donor_ids = tokenizer(donor_text, add_special_tokens=False)["input_ids"]
                d_start, r_start, block_len = longest_common_token_block(donor_ids, receiver_ids)
                if block_len < args.min_block_tokens:
                    skipped.append({"source_row_index": source_row_index,
                                    "reason": f"{donor_key} block too short: {block_len}"})
                    ok = False
                    break
                cache, _ = run_donor(model=model, token_ids=donor_ids, hook_names=hook_names,
                                     max_choice_tokens=1, cache_dtype=dtype)
                donor_blocks[donor_key] = {
                    layer: cache[hook_name_by_layer[layer]][d_start : d_start + block_len]
                    for layer in layers
                }
                receiver_starts[donor_key] = r_start
                donor_blocks[donor_key + "_len"] = block_len
            if not ok:
                print(f"row {row_idx}: skipped", flush=True)
                continue
            print(f"row {row_idx}/{len(selected_rows)} source_row={source_row_index} "
                  f"gold_concept={gold_concept} wrong_concept={wrong} "
                  f"block_gold={donor_blocks['gold_len']} block_wrong={donor_blocks['wrong_len']}",
                  flush=True)

            for condition_index, condition in enumerate(CONDITION_PLAN):
                if args.do_sample:
                    torch.manual_seed(args.sample_seed + source_row_index * 10007
                                      + condition_index * 101)
                fwd_hooks = []
                if condition.patch_kind is not None:
                    blocks = donor_blocks[condition.donor]
                    r_start = receiver_starts[condition.donor]
                    for layer in layers:
                        hook_fn, _ = make_patch_hook(
                            donor_block=blocks[layer],
                            receiver_start=r_start,
                            patch_kind=condition.patch_kind,
                            seed=args.sample_seed + source_row_index * 10007
                            + condition_index * 101 + layer)
                        fwd_hooks.append((hook_name_by_layer[layer], hook_fn))
                with model.hooks(fwd_hooks=fwd_hooks):
                    if args.do_sample and args.samples_per_row > 1:
                        sample_batch = generate_sample_batch(
                            model=model, token_ids=receiver_ids,
                            n_samples=args.samples_per_row,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos, cache_dtype=dtype)
                    else:
                        sample_batch = [generate_one(
                            model=model, token_ids=receiver_ids,
                            max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                            temperature=args.temperature, stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype)]
                for sample_index, (new_ids, reply) in enumerate(sample_batch):
                    score = score_reply(stage1_row, reply)
                    output_row = {
                        "schema_version": 1,
                        "source_file": str(args.jsonl),
                        "source_row_index": source_row_index,
                        "example_id": stage1_row.get("example_id"),
                        "height": stage1_row.get("height"),
                        "model": args.model,
                        "original_is_correct_strong": bool(stage1_row.get("is_correct_strong")),
                        "condition": condition.label,
                        "sample_index": sample_index,
                        "method": "hint_state_interchange",
                        "target_variable": "target_concept",
                        "representation_type": "patched_residual_state",
                        "patch_layers": layers,
                        "donor": condition.donor,
                        "gold_concept": gold_concept,
                        "wrong_concept": wrong,
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        **concept_mention_fields(reply, gold_concept, wrong),
                        **score,
                    }
                    rows.append(output_row)
                    fout.write(json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                    print(f"  {condition.label}[{sample_index}]: strong={output_row['is_correct_strong']} "
                          f"gold_concept={output_row['output_mentions_gold_concept']} "
                          f"wrong_concept={output_row['output_mentions_wrong_concept']}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_patch_rows(rows)
    from collections import defaultdict
    concept_rates: dict[str, Any] = defaultdict(dict)
    for label in {r["condition"] for r in rows}:
        subset = [r for r in rows if r["condition"] == label]
        concept_rates[label] = {
            "mentions_gold_concept": sum(r["output_mentions_gold_concept"] for r in subset) / len(subset),
            "mentions_wrong_concept": sum(r["output_mentions_wrong_concept"] for r in subset) / len(subset),
        }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_hint_state_interchange.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "hint_state_interchange",
        "representation_type": "patched_residual_state",
        "layers": layers,
        "selection": selection_summary,
        "skipped_rows": skipped,
        "generation": {
            "conditions": [c.__dict__ for c in CONDITION_PLAN],
            "samples_per_row": args.samples_per_row,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": summary,
        "concept_mention_rates": concept_rates,
        "n": len(rows),
        "controls": ["regenerated_baseline", "position_shuffled_donor",
                     "magnitude_matched_gaussian_noise", "bidirectional_wrong_concept_donor"],
        "causal_abstraction_claim": (
            "Interchange test on target_concept: behaviorally validated hint-first donors "
            "(gold repairs at 1.000, wrong-concept misdirects at 0.000) are patched into the "
            "unhinted run at the shared context block. A causal-handle claim requires gold-donor "
            "repair and wrong-donor misdirection beyond shuffled and matched-noise controls."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
