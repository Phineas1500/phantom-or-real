# Phantom or Real? — behavioral/data pipeline

Stage 1 of the project: generate InAbHyD reasoning examples, run Gemma 3 4B-IT
and Gemma 3 27B-IT inference, score correctness, annotate structural features,
and ship a labeled JSONL dataset for Teammate B (probes) and Ram (causal
validation). Full spec in `BEHAVIORAL_DATA_PLAN.md`.

## Layout

```
src/                     — importable library code (all modules are importable as `src.foo`)
  config.py              — HEIGHT_SAMPLE_SIZES, SYSTEM_PROMPT, make_user_prompt, get_seed
  example.py             — ExampleView dataclass (pickle-safe Ontology stand-in)
  bd_path.py             — locate beyond-deduction on sys.path
  generate_examples.py   — Phase 3.3 generator (python -m src.generate_examples ...)
  annotations.py         — Phase 2.1 structural annotations (has_direct_member etc.)
  export.py              — Phase 2.2 JSONL row schema + kb_to_dict structured FOL
  inference.py           — Phase 2.3 async concurrent runner (single task/height)
  analysis.py            — Phase 4 slicing + summary JSON builders
  error_classification.py — Phase 5 GPT-as-judge error-type classifier

scripts/
  run_inference.py       — orchestrator: sweep heights for one (model, task)
  make_plots.py          — Phase 4 plots + Phase 5.5 error-type stacked bars
  sanity_check.py        — pre-handoff integrity + warnings per plan §Sanity Checks
  validate_annotations.py — smoke-test annotations on 100 examples per height
  serve_gemma3_4b.sh     — launch local vLLM OpenAI-compatible server

data/
  pilot/                 — 50 per height × 4 heights × 2 tasks (for Phase 1 pilot)
  full/                  — 1000/2000/3000/5000 per height × 2 tasks (for Phase 3)

results/                 — per-(model, task) JSONL + run metadata + summary JSON
configs/                 — reserved for any future parameter sweeps
third_party_beyond_deduction — symlink to ~/beyond-deduction (NOT checked in)

environment.yml          — portable conda env spec (loose pins)
environment.lock.yml     — exact pinned lockfile (current machine)
requirements.txt         — pip freeze
```

## Reproducibility notes

- Conda env name: `phantom`. Python 3.12 (upstream beyond-deduction uses 3.12+
  nested f-strings, so 3.11 fails to import ontology.py).
- `beyond-deduction` is NOT vendored. Clone it separately (or symlink). The
  project locates it via:
    1. `BD_PATH` env var
    2. `./third_party_beyond_deduction`
    3. `~/beyond-deduction`
  See `src/bd_path.py`.
- The Ontology object tree is NOT pickled — it has a hash-by-name /
  dict-of-node internal structure that crashes on `pickle.load`. We serialize
  a `src.example.ExampleView` dataclass instead. Downstream code accesses
  `theories / observations / hypotheses / fol_* / config.hops` via duck
  typing — no code path requires a real Ontology after generation.
- **Seeds**: `src.config.get_seed(task, height)` returns a deterministic
  integer — NOT `hash()` (which would be randomized per-process). The shipped
  dataset's exact seeds are pinned in `SHIPPED_SEEDS`; unseen combinations
  fall back to an md5-derived integer.
  - Note: even with a fixed seed, upstream `Ontology.__init__` has some
    residual set-iteration non-determinism (e.g., a generated hypothesis may
    come out as `"Every X is Y"` vs `"Each X is Y"` on different runs). The
    authoritative source of the shipped examples is the pickles in `data/full/`,
    not a regeneration.
- System prompt intentionally preserves the "assitant" typo from the
  upstream v2 paper's `generate.py` (plan §Implementation Notes) to keep
  comparability.
- **Gemma 3 has no `system` role.** `src.inference.build_messages` concatenates
  system+user into one user message whenever the model name contains `"gemma"`.
  Probe-position reconstruction must do the same — see `results/README.md`.
- **Structured FOL schema.** `ontology_fol_structured` is a single flat KB
  with keys `{membership, inheritance, properties, negated_properties, hypothesis}`
  (matching BEHAVIORAL_DATA_PLAN.md §2.2), built by combining theories +
  observations into one KnowledgeBase.

## Common commands

All commands assume `conda activate phantom` and `cd ~/phantom-or-real`.

### Generate examples (Phase 3.3)

```bash
# Pilot — 50 per height × 4 heights × 2 tasks
python -m src.generate_examples --counts pilot

# Full — 1000/2000/3000/5000 × 4 heights × 2 tasks (44k total per model)
python -m src.generate_examples --counts full
```

### Run inference

Both examples assume `.env` is populated; `src/env_loader.py` loads it on import
so the `--base-url` / `--api-key` flags can be omitted.

```bash
# Local Gemma 3 4B via vLLM (requires HF_TOKEN in .env)
./scripts/serve_gemma3_4b.sh                             # separate terminal

python scripts/run_inference.py \
  --examples-dir data/pilot \
  --tasks property ontology \
  --heights 1 2 3 4 \
  --model google/gemma-3-4b-it \
  --base-url http://localhost:8000/v1 \
  --concurrency 16 \
  --model-slug gemma3_4b \
  --output-dir results/pilot

# Gemma 3 27B via Modal (preferred)
python scripts/run_inference.py \
  --examples-dir data/full \
  --tasks property ontology \
  --model gemma3-27b \
  --concurrency 16 \
  --model-slug gemma3_27b \
  --output-dir results/full
```

### Error-type classification (Phase 5)

The default judge model is `gpt-5.4-mini`, not `gpt-5.4-nano`. The plan §5.3
agreement check (`scripts/error_classify_agreement.py`) came in at 57.5% on
this dataset, below the 85% threshold the plan uses for sticking with nano,
so the shipped dataset was classified with mini. Override with `--model
gpt-5.4-nano` only if you're re-running for the cheaper tier.

```bash
OPENAI_API_KEY_GPT=... python -m src.error_classification \
  --input  results/full/gemma3_27b_infer_property.jsonl \
  --output results/full/gemma3_27b_infer_property.with_errortype.jsonl \
  --concurrency 16
```

### Analysis & sanity checks

```bash
python scripts/sanity_check.py --jsonl results/full/*.jsonl
python scripts/make_plots.py --results-dir results/full --output-dir docs/figures \
  --jsonl results/full/*.jsonl
```

## External credentials required

Drop into `.env` in the project root (see `.env.example`). `src/env_loader.py`
auto-loads it at module import; no code reads disk after that.

| Purpose                     | Env var              | Provider options |
|-----------------------------|----------------------|------------------|
| Gemma 3 4B (gated on HF)    | `HF_TOKEN`           | HuggingFace — accept license at https://huggingface.co/google/gemma-3-4b-it |
| Gemma 3 27B (inference)     | `OPENAI_BASE_URL` + `OPENAI_API_KEY` | Modal (preferred) / OpenRouter / Together / Fireworks |
| Error classification        | `OPENAI_API_KEY_GPT` | OpenAI proper (`gpt-5.4-nano`, `gpt-5.4-mini`) |

Two separate OpenAI-compatible keys are intentional — the inference key (Modal
URL / OpenRouter) is distinct from the OpenAI judge key. The error classifier
reads `OPENAI_API_KEY_GPT` first, falling back to `OPENAI_API_KEY` only if
unset.

### Modal endpoint specifics

The project's Gemma 3 27B Modal deployment lives at
`beyond-deduction/deployment/gemma3_27b_modal.py` (app name `gemma3-27b-inference`).
After `modal deploy`, note the printed URL; the OpenAI-compatible base URL is
that URL with `/v1` appended.

```bash
# Deploy / confirm URL
modal deploy third_party_beyond_deduction/deployment/gemma3_27b_modal.py

# In .env:
OPENAI_BASE_URL=modal-url
OPENAI_API_KEY=not-needed

# Sanity check: one prompt, then exit
python scripts/check_modal_endpoint.py --model gemma3-27b
```

**Gotcha — served model name:** the Modal deployment registers the model as
`gemma3-27b` (via `--served-model-name`), *not* `google/gemma-3-27b-it`. Pass
`--model gemma3-27b` to `scripts/run_inference.py`.

**Gotcha — no system role:** Gemma chat templates don't accept a `system` role.
`src.inference.build_messages` detects `"gemma"` in the model name and
concatenates system+user into a single user message, matching the upstream
`gemma3_27b_modal.py` test. This is automatic; no flag needed.

## Handoff notes for Teammate B

- **Entity-name mangling in any pre-patch data.** The `bd_path` monkey-patch
  (`src/bd_path.py::_apply_normalize_singular_patch`) fixes an upstream
  `normalize_to_singular` bug that stemmed four proper nouns:
  `Thomas → thoma`, `Charles → charle`, `James → jame`, `Nicholas → nichola`.
  The shipped JSONLs under `results/full/with_errortype/` are post-patch and
  clean — a full scan of 44,000 rows finds zero mangled forms in
  `ontology_fol_structured.{inheritance,membership}` or
  `structural.target_concept`. But if you run any pre-commit `30d910e` scratch
  files, notebooks, or partial reruns that predate the patch, filter or
  search with both the proper and the mangled forms and expect mangled entries
  to be wrong.
- **Within-height structural slicing is vacuous.** `has_direct_member = 100%`
  on every (task, height) cell of the shipped dataset; `num_direct_paths`,
  max non-direct proof depth, and target-concept branching factor are
  deterministic per height as well. The only `structural` feature with any
  within-height variance is `parent_salience`, and even that is 4984/5000 at
  the modal value for h=4. Probe analyses cannot use these slices to
  decorrelate shortcut-availability from depth on this dataset.
- **Scoring is unaffected by the normalize-proper-nouns fix.** The same
  mangling was applied to both predictions and ground truth pre-patch, so
  `is_correct_strong` / `is_correct_weak` are symmetric across the bug. After
  the fix landed, `scripts/rescore_jsonl.py` was run end-to-end over the
  shipped files and reported `changed_correct = 20, 19, 5, 1` across
  `(27b property, 27b subtype, 4b property, 4b subtype)`. Those flips are
  attributable to the separate Gemma-3-specific parser extensions in
  `src/gemma3_parse.py` (canonicalizing `"Being X is a property of Y"`,
  `"X is (not) a property of Y"`, etc.), not to the normalize fix. The
  scoring-axis of the shipped dataset matches the patched code.

## Coding conventions followed here

- Write no comments that merely describe what the code does. Keep docstrings
  on modules and non-trivial functions only.
- Prefer editing existing modules over adding parallel ones.
- Tests only exercise scoring/annotation plumbing — no network tests.
- Don't write results summaries that restate the diff; the JSONL is the source
  of truth.
