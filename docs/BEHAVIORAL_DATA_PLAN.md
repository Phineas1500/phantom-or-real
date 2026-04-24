# Behavioral + Data Pipeline Implementation Plan

## Context

**Project:** "Phantom or Real?" investigates whether sparse autoencoder (SAE) features from Gemma Scope 2 can predict LLM reasoning failures on depth-controlled ontology tasks, and whether those predictive features are causally relevant or merely correlational "phantoms" — a concern raised by Ma et al. (2026), who found that 59-94% of apparent "reasoning features" identified via standard contrastive methods actually activate on surface-level lexical cues rather than driving reasoning. The project adapts Cox et al. (2026)'s methodology — difference-of-means probes trained on pre-generation activations, validated causally via feature steering against an orthogonal-direction baseline — to InAbHyD's depth-controlled abductive reasoning setting.

The project has three components:
1. **Behavioral evaluation** (this plan): generate InAbHyD examples and run Gemma 3 inference to produce labeled success/failure data across four ontology tree depths.
2. **Probe training**: train classifiers on multiple feature sources (raw residual stream activations, SAE features at two widths, skip-transcoder features) to predict success/failure from internal representations.
3. **Causal validation**: steer the top features identified by probes and measure whether this causally flips model outcomes, compared to an orthogonal-direction baseline that controls for generic perturbation effects.

This plan covers Component 1 only. The output is a labeled dataset that Teammate B will use for probe training and that Ram will use for causal validation experiments.

**Reference codebase:** `beyond-deduction/` repository (fork provided separately). Key files:
- `benchmark/run_experiments.py` — main driver for dataset generation + inference
- `benchmark/ontology.py` — `Ontology`, `OntologyConfig`, `Difficulty` classes
- `benchmark/morphology.py` — name pools
- `benchmark/evaluate.py` — accuracy and quality scorers; also contains the `KnowledgeBase` class we need for structural annotations
- `benchmark/fol.py`, `benchmark/symbolic_fol.py` — FOL utilities

**Reference paper:** Sun & Saparov v2 (InAbHyD, 2026) — "Do Language Models Follow Occam's Razor?"

**Downstream interface:** Teammate B needs a JSONL file per (model, task) combination with one row per example, plus an aggregated summary for the final report's Behavioral Results section.

## Objectives

1. Generate InAbHyD single-hypothesis examples at heights 1-4 for two tasks: infer_property and infer_subtype_relation.
2. Run inference on Gemma 3 4B-IT and Gemma 3 27B-IT.
3. Score each example for correctness (strong accuracy, which equals weak accuracy for single-hypothesis).
4. Compute structural annotations per example for shortcut-availability analysis.
5. Classify error types on incorrect examples using GPT-4o-as-judge following the v2 paper's methodology.
6. Export a clean JSONL per-example dataset (with structured FOL representations) that Teammate B can consume directly.
7. Produce accuracy-vs-depth curves comparable to paper Figure 3.

**Not in scope:**
- Infer_membership_relation task (has built-in shortcut per paper footnote 5).
- Multi-hypothesis examples (severe class imbalance at deep depths on Gemma 3 models).
- Matched pairs (tautology issue identified in prior work).
- Naturalistic paraphrasing step from v2 (breaks minimal-pair controls for MI).

---

## Dataset Specification

### Task A: `infer_property` (primary)
- Code parameter: `task_type='property'` → `OntologyConfig(recover_property=True)`
- Ground truth format: "Every [concept] is [property]" or "Every [concept] is not [property]"
- The model must identify which concept in the ontology tree to generalize to, and which property to attribute to it.

### Task B: `infer_subtype_relation` (secondary, for transfer test)
- Code parameter: `task_type='ontology'` → `OntologyConfig(recover_ontology=True)`
- Ground truth format: "Every [concept_A] is [concept_B]" (subtype relation)
- The model must identify a subtype relation from observations of members belonging to both concepts.

### Per-height example counts

Run the same counts for both tasks, both models:

| Height | Examples per (task, model) | Rationale |
|--------|---------------------------|-----------|
| 1      | 1000                      | Near-ceiling accuracy; baseline class for probes |
| 2      | 2000                      | Expected ~50/50 split on 27B; primary probe training |
| 3      | 3000                      | Lower accuracy (~25% on 27B); need more positives |
| 4      | 5000                      | Near-floor (~5% on 27B); need many for positive class |

**Total per (task, model) combination: 11,000 examples.**
**Total across 2 tasks × 2 models: 44,000 inferences.**

Rationale for scaling: the goal is to have at least 100 positive examples at the deepest depth (h=4) on Gemma 3 27B for infer_property. At 5% accuracy, 5000 examples gives ~250 positives. Gemma 3 4B at h=4 will likely be worse (~2% accuracy or lower), so the ~100 positive threshold may not be met there even at 5000 examples; this is a known limitation and the probe results at (4B, h=4) should be reported with appropriate caveats rather than dropped.

### Generation settings (do NOT change)

- `Difficulty.SINGLE` — single-hypothesis examples only
- `mix_hops=False` — fixed tree height per example
- No GPT-4o paraphrasing — keep templated surface form
- Uniform 2/3 branching (v1-style, as in beyond-deduction code)
- Seed reproducibility: use a documented seed for each (task, height) combination so regeneration is deterministic

---

## Phase 1: Setup and Pilot (Day 1-2)

### Step 1.1: Fork and verify the codebase

Clone `beyond-deduction/` into the working environment. Verify the existing code runs by generating 10 examples at height 2 with `task_type='property'` and inspecting the output. The `Ontology` objects should have attributes `theories` (str), `observations` (str), `hypotheses` (str), `fol_theories`, `fol_observations`, `fol_hypotheses`, `nodes` (list of lists of `OntologyNode`), and `root`.

### Step 1.2: Set up inference endpoints

Two models need to be accessible:
- **Gemma 3 4B-IT**: Can run locally (requires ~12GB VRAM) or via API (Together.ai, Fireworks). Recommend local via vLLM for speed and control.
- **Gemma 3 27B-IT**: Requires ~60GB VRAM local (A100 80GB or similar), or use API. Recommend API for simplicity unless Ram has a local cluster.

The existing code uses OpenAI-compatible endpoints via `OPENAI_BASE_URL` environment variable. This works with vLLM's OpenAI-compatible server and with Together.ai/Fireworks. No code changes needed for endpoint support.

Verify both models respond correctly by running 5 test prompts at h=1.

### Step 1.3: Pilot run (50 examples per config)

Before committing to the full 44,000 inferences, run a pilot:
- 50 examples × 4 heights × 2 tasks × 2 models = 800 inferences total.
- Takes roughly 10-30 minutes.

**Purpose of pilot:** Verify three things:
1. **Output parsing works on Gemma 3.** The existing `parse_hypotheses_from_response()` in `evaluate.py` was built for Gemma 2 outputs. Gemma 3 may format responses slightly differently (newline conventions, lead-in phrases). Check failures for parsing errors vs genuine incorrect answers.
2. **Accuracy numbers are sane.** Compare pilot accuracy on Gemma 3 27B against paper Figure 3 Gemma 3-27B data points (read from the figure, not exact values):
   - **infer_property** strong accuracy: h=1 ~90%, h=2 ~50%, h=3 ~22%, h=4 ~5%
   - **infer_subtype_relation** strong accuracy: h=1 ~85%, h=2 ~47%, h=3 ~37%, h=4 ~7%
   
   If pilot numbers are wildly off (>15pp deviation), there's likely a prompt, tokenization, or parsing issue. Note: Gemma 3 4B does not have published benchmark numbers — the 4B pilot is a pipeline correctness check, not an accuracy calibration.
3. **Class balance is workable.** Count positive examples at each (model, task, height). Decision tree for Gemma 3 4B × h=4 specifically:
   - **Pilot yields ≥5 positives in 50 examples (≥10% accuracy):** proceed with full 5000 at h=4, expect ~500 positives, probe training will work.
   - **Pilot yields 1-4 positives (2-8%):** proceed with full 5000 at h=4, expect ~100-400 positives, acknowledge wider confidence intervals in final report.
   - **Pilot yields 0 positives (0%):** run 1000 extra examples at h=4 as a fast confirmation. If still 0 positives, drop (4B, h=4) from probe analysis and report as "4B could not solve any h=4 examples in 6000 trials." Keep the accuracy curve for behavioral comparison but don't train probes on this cell.

**Exit criteria for Phase 1:** Pilot accuracy on Gemma 3 27B infer_property at h=1 and h=2 within ±10 percentage points of paper Figure 3. Parse failure rate below 5%.

---

## Phase 2: Code Modifications

### 2.1: Add structural annotations

Add a new module `benchmark/annotations.py` that computes structural properties per example. For each generated `Ontology`, compute:

```python
def compute_structural_annotations(ontology: Ontology, task_type: str) -> dict:
    """
    Compute structural features that may enable shortcut-based reasoning.
    
    Returns a dict with:
      - target_concept: str — the concept named in the ground truth hypothesis
      - has_direct_member: bool — whether any entity in observations is directly 
        a member of target_concept in the visible theories (proof depth == 1)
      - num_direct_paths: int — count of observation entities that are direct members
      - parent_salience: int — how many times target_concept name appears in theories text
      - num_theories_axioms: int — total count of axioms in theories
      - num_observations: int — count of observation statements
      - tree_height: int — should match config.hops
    """
```

**IMPORTANT: Use the existing `KnowledgeBase` class from `evaluate.py`.** This class already parses theories into internal `membership`, `inheritance`, `properties`, and `negated_properties` dicts. The key method is `get_all_concepts_for_entity(entity)` which returns `(concept, proof_depth)` tuples. **An entity is a direct member iff the target concept appears at proof_depth=1.**

**Implementation pattern.** *Post-implementation update (2026-04):* on all 44,000 shipped rows, `has_direct_member` is a deterministic **100%**, not the ~92% this doc originally cited. The 92% figure reflected an upstream bug in `benchmark/evaluate.py::normalize_to_singular`: the function stemmed the proper nouns **Thomas / Charles / James / Nicholas** (four of the ~100 entity names in `morphology.py`), so their membership facts were mis-routed to `KnowledgeBase.inheritance` and silently missed. The bug is fixed by a monkey-patch in `src/bd_path.py::_apply_normalize_singular_patch` — we patch rather than edit upstream because `third_party_beyond_deduction` is a symlink that may back other work. **Analysis consequence:** the within-height split on `has_direct_member` in Phase 4.2 is **vacuous on this dataset** — every row has it set to `True`. `num_direct_paths`, the max non-direct proof depth, and the target-concept branching factor are also deterministic per height (see `results/README.md`). The within-height "structural slicing" angle is therefore a dead end on this dataset; probe analyses cannot rely on it to decorrelate shortcut-availability from depth. Pre-implementation pattern kept below for reference:

```python
import re
from evaluate import KnowledgeBase, parse_ground_truth, parse_hypothesis_structure

def compute_structural_annotations(ontology, task_type):
    # Parse the ground truth to extract target concept
    gt_hyps = parse_ground_truth(ontology.hypotheses)
    subject, predicate = parse_hypothesis_structure(gt_hyps[0])
    target_concept = subject  # e.g., "lerpant" from "Every lerpant is rainy"
    
    # Build KB from visible theories
    kb = KnowledgeBase()
    kb.add_from_text(ontology.theories)
    
    # Parse entities from observations (capitalized words)
    observation_entities = set(w.lower() for w in re.findall(r'\b[A-Z][a-z]+\b', 
                                                               ontology.observations))
    
    # Count entities that are direct members of target_concept
    num_direct_paths = 0
    for entity in observation_entities:
        for concept, depth in kb.get_all_concepts_for_entity(entity):
            if concept == target_concept and depth == 1:
                num_direct_paths += 1
                break
    
    has_direct_member = num_direct_paths > 0
    parent_salience = ontology.theories.lower().count(target_concept.lower())
    
    return {
        'target_concept': target_concept,
        'has_direct_member': has_direct_member,
        'num_direct_paths': num_direct_paths,
        'parent_salience': parent_salience,
        'num_theories_axioms': ontology.theories.count('.'),
        'num_observations': ontology.observations.count('.'),
        'tree_height': ontology.config.hops,
    }
```

**Special handling for `task_type='ontology'` (subtype relation):** The ground truth is of the form "Every [concept_A] is [concept_B]". The "target" is the subtype relation itself. Both concepts should be tracked:
```python
if task_type == 'ontology':
    # For "Every A is B", subject='a', predicate is a concept name
    target_subtype = subject  # the narrower concept
    target_supertype = predicate  # the broader concept
    # has_direct_member can be redefined as: any observation entity directly 
    # a member of target_subtype (the narrower concept)
```

Store both `target_subtype` and `target_supertype` in the annotations dict for subtype tasks.

**Validation:** Run on 100 examples at h=2, h=3, h=4 for `infer_property` and verify `has_direct_member` rates match expectations (should be very high, 80-95%, consistent with prior analysis). If the rates are wildly off, there's a parsing bug.

### 2.2: Extend output format to JSONL with structured FOL

The current code saves results as pickles containing `Ontology` objects. Add a post-processing step that flattens to JSONL with this schema:

```json
{
  "example_id": "property_h2_0042",
  "task": "infer_property",
  "height": 2,
  "model": "gemma-3-27b-it",
  "prompt_text": "Q: [full user prompt]...",
  "system_prompt": "You are a helpful assitant...",
  "ground_truth": "Every lerpant is not salty",
  "model_output": "Every lerpant is not salty.",
  "is_correct_strong": true,
  "is_correct_weak": true,
  "quality_score": 1.0,
  "parse_failed": false,
  "failure_mode": null,
  "error_type": null,
  
  "structural": {
    "target_concept": "lerpant",
    "has_direct_member": false,
    "num_direct_paths": 0,
    "parent_salience": 3,
    "num_theories_axioms": 6,
    "num_observations": 3,
    "tree_height": 2
  },
  
  "ontology_raw": {
    "theories": "Barbara is a lerpant. ...",
    "observations": "Pamela is not salty. ...",
    "hypotheses": "Every lerpant is not salty"
  },
  
  "ontology_fol_string": {
    "theories": "∀x(rifpist(x) → lerpant(x)). timple(Carol). ...",
    "observations": "¬salty(Pamela). ¬salty(Barbara). ...",
    "hypotheses": "∀x(lerpant(x) → ¬salty(x))"
  },
  
  "ontology_fol_structured": {
    "membership": {"Barbara": ["lerpant"], "Pamela": ["yumpus"], "Carol": ["timple"]},
    "inheritance": {"yumpus": ["lerpant"], "rifpist": ["lerpant"], "timple": ["lerpant"]},
    "properties": {},
    "negated_properties": {"Pamela": ["salty"], "Barbara": ["salty"], "Carol": ["salty"]},
    "hypothesis": {"type": "rule", "subject": "lerpant", "predicate": "salty", "negated": true}
  }
}
```

**Structured FOL implementation (~1-2 hours of work):** Rather than re-parsing FOL strings, reuse the `KnowledgeBase` class that already builds internal structured representations. After `kb = KnowledgeBase(); kb.add_from_text(theories)`, serialize the KB's internal state:

```python
def kb_to_dict(kb: KnowledgeBase) -> dict:
    """Serialize KnowledgeBase internal state to JSON-compatible dict."""
    return {
        'membership': {k: sorted(list(v)) for k, v in kb.membership.items()},
        'inheritance': {k: sorted(list(v)) for k, v in kb.inheritance.items()},
        'properties': {k: sorted(list(v)) for k, v in kb.properties.items()},
        'negated_properties': {k: sorted(list(v)) for k, v in kb.negated_properties.items()},
    }
```

For the hypothesis specifically, parse it into a structured form using `parse_hypothesis_structure()` from `evaluate.py`. This gives you subject and predicate; additional fields like `negated`, `type`, and relation type can be derived.

**Why include both FOL string and structured form:** String form is immediately inspectable (trivial to `jq`), structured form is usable for downstream symbolic analysis without re-parsing. Marginal storage cost, significant flexibility gain.

**Why JSONL and not pickle:**
- Language-agnostic; Teammate B can use whatever framework.
- Streamable for large files.
- Inspectable with standard tools (`head`, `jq`, etc.)
- No dependency on the `Ontology` class definition at read time.

Save one JSONL per (model, task) combination: `gemma3_27b_infer_property.jsonl`, `gemma3_27b_infer_subtype.jsonl`, `gemma3_4b_infer_property.jsonl`, `gemma3_4b_infer_subtype.jsonl`. Each file contains all heights concatenated, with `height` as a field.

### 2.3: Add concurrent inference support

The current `run_zero_shot_single_hypothesis` function calls the API sequentially. For 44,000 inferences, this is a bottleneck. Rewrite to use `asyncio` with `openai.AsyncOpenAI` and a semaphore limiting concurrency.

Recommended concurrency levels:
- Gemma 3 4B (local vLLM): 8-16 concurrent requests, depending on GPU throughput.
- Gemma 3 27B (API): 4-8 concurrent requests (most API providers allow this).

This should give ~5-10x speedup over sequential inference.

**Error handling:** Retry with exponential backoff on transient API errors. Log but don't fail on persistent errors — mark the example as `parse_failed=true` and continue.

**Distinguish failure modes via a `failure_mode` field** (string, nullable):
- `"api_error"` — API returned an error (rate limit, timeout, server error) after retries
- `"empty_response"` — API returned but `model_output` is empty or whitespace only
- `"parse_error"` — model returned a non-empty response but `parse_hypotheses_from_response` couldn't extract any hypothesis
- `"refusal"` — model explicitly refused or responded with meta-commentary (detect via simple keyword heuristics: output contains "cannot", "unable to", "not able to" with no hypothesis structure)
- `null` — normal parse, example scored successfully

Set `parse_failed=true` iff `failure_mode` is non-null. The breakdown by failure mode helps diagnose systematic issues during the pilot.

### 2.4: Configurable per-height sample size

The current code takes a single `num_examples` argument. Modify to accept a per-height dict:

```python
HEIGHT_SAMPLE_SIZES = {1: 1000, 2: 2000, 3: 3000, 4: 5000}
```

Pass this through `run_zero_shot_single_hypothesis` and iterate heights.

### 2.5: Seed management

Currently uses a single `SEED = 62471893`. For reproducibility across tasks and heights, use a deterministic seed derivation:

```python
def get_seed(task_type: str, height: int) -> int:
    base = hash(f"{task_type}_h{height}") % (2**31)
    return base
```

This means regenerating a specific (task, height) combination produces the same examples without needing to regenerate everything.

### 2.6: Verify Gemma 3 parse compatibility

Run 20 example outputs through `parse_hypotheses_from_response()` manually. Check for:
- Lead-in phrases Gemma 3 uses that Gemma 2 didn't (e.g., "Based on the observations, ...")
- Formatting quirks (numbered lists, bullet points, markdown)
- Whether Gemma 3 outputs multiple hypotheses even in single-hypothesis mode (if so, use `first_only=True` which the evaluator already supports)

Update the parser if needed. Keep the original Gemma 2 parsing paths intact for backward compat.

---

## Phase 3: Full Inference Runs

### 3.1: Execution order

Run in this order to minimize risk and get early feedback:

1. **Gemma 3 4B, infer_property, all heights** — fastest model, gives quick feedback on pipeline.
2. **Gemma 3 27B, infer_property, all heights** — primary dataset for probe training.
3. **Gemma 3 4B, infer_subtype, all heights** — secondary task, 4B first.
4. **Gemma 3 27B, infer_subtype, all heights** — completes the full grid.

Between each major run, do a quick check that outputs look sensible (accuracy curves shape, JSONL is well-formed, structural annotations are populated).

### 3.2: Estimated wall clock time

Assuming ~2-5 inferences/sec for 27B and ~5-15 inferences/sec for 4B with concurrent requests:

- 4B, infer_property (11,000 inferences): ~15-40 minutes
- 27B, infer_property (11,000 inferences): ~40-90 minutes
- 4B, infer_subtype (11,000 inferences): ~15-40 minutes
- 27B, infer_subtype (11,000 inferences): ~40-90 minutes

Total: ~2-4 hours of compute. Budget a full day for Phase 3 including sanity checks and any re-runs.

### 3.3: Data generation

Generation is fast (~1 second per 5000 examples). Generate all examples first, save to disk, then run inference on the saved examples. This decouples generation from inference and means inference failures don't lose the generation state.

```
data/
  examples_property_h1.pkl      (1000 Ontology objects)
  examples_property_h2.pkl      (2000)
  examples_property_h3.pkl      (3000)
  examples_property_h4.pkl      (5000)
  examples_subtype_h1.pkl       (1000)
  ... etc
```

Total disk usage: trivial, each pickle is a few MB.

### 3.4: Inference output layout

```
results/
  gemma3_4b_infer_property.jsonl       (11,000 rows)
  gemma3_27b_infer_property.jsonl      (11,000 rows)
  gemma3_4b_infer_subtype.jsonl        (11,000 rows)
  gemma3_27b_infer_subtype.jsonl       (11,000 rows)
  
  summary_accuracy.json                (per-config accuracy with CIs)
  summary_by_structure.json            (accuracy sliced by has_direct_member)
```

---

## Phase 4: Analysis and Deliverables

### 4.1: Core accuracy plots

Reproduce the format of paper Figure 3 for the two tasks we ran:
- X-axis: height (1, 2, 3, 4)
- Y-axis: accuracy
- Two lines per plot: Gemma 3 4B, Gemma 3 27B
- Error bars: Wilson 95% CI
- One plot per task

Save as PNG/PDF. These go in the final report.

### 4.2: Structural slicing

For each (model, task, height), compute accuracy separately for:
- `has_direct_member=True` subset
- `has_direct_member=False` subset

If there's a large gap (e.g., 80% accuracy when direct member exists, 10% when not), this confirms the shortcut concern at the dataset level. This becomes an important piece of context for Teammate B's probe analysis — probes trained on the full dataset may be detecting `has_direct_member` rather than depth-related reasoning.

Also compute accuracy bins by `parent_salience` (1-2 mentions vs 3-5 vs 6+). If salience correlates with accuracy, this is another structural confound to flag.

### 4.3: Output strategy analysis

For Gemma 3 12B specifically, prior work identified an "entity-level enumeration" strategy where the model lists entity-level hypotheses before committing to a concept-level generalization. Check if this happens on Gemma 3 4B or 27B:

- Parse the model output and check whether the first hypothesis is entity-level (e.g., "Jerry is not muffled") vs concept-level (e.g., "Dalpists are not muffled").
- Compute the fraction of examples using each strategy at each depth.
- Report as a small analysis in the Behavioral Results section.

### 4.4: Summary statistics for the final report

Produce a summary JSON containing:
- Per-config: accuracy mean, Wilson 95% CI, n, positive count, parse failure rate
- Per-config: accuracy by structural subsets
- Aggregate: comparison to paper Figure 3 data points (sanity check)

---

## Phase 5: Error Type Classification (required)

Apply LLM-as-judge classification to incorrect examples, following the v2 paper's methodology (Appendix H, Figure 10). This gives Teammate B a richer multi-class probe training target beyond binary correct/incorrect.

**Note on model choice:** The InAbHyD v2 paper used GPT-4o because it was the standard LLM-as-judge model at publication. As of April 2026, GPT-4o is legacy and the current OpenAI lineup has substantially cheaper nano/mini variants that handle 4-category classification tasks well. **Use `gpt-5.4-nano` as the primary classifier**, with `gpt-5.4-mini` as a fallback if nano's accuracy is insufficient. Cost estimate for the full ~31,600 classifications: ~$15 with nano vs ~$170 with GPT-4o-equivalent pricing.

### 5.1: Classification pipeline

For each example where `is_correct_strong=False`, call the OpenAI API with the exact classification prompt from paper Figure 10. The prompt asks the model to categorize the error into one of four types:

- **Error Type 1: Wrong ontology direction** — hypothesis reverses the subtype relation (e.g., produces "All rats are mammals" when GT is "All mammals are rats")
- **Error Type 2: Ignore ontology / unnecessary hypotheses** — model produces correct hypothesis plus extras that don't follow Occam's Razor. **Most common error in the paper.**
- **Error Type 3: Fall back to trivial hypotheses** — model reuses observations as hypotheses (e.g., "Jerry is not muffled" as hypothesis when the GT requires generalization)
- **Error Type 4: Hallucinated entities** — model uses entities not present in the input

### 5.2: Implementation

```python
ERROR_TYPE_PROMPT = """You are an expert evaluator of reasoning quality. Your task is to analyze a model's reasoning process for a question and determine the type of error it contains.

## Task
Given:
* A question
* The model's reasoning process
* The correct answer

Identify which error type best describes the mistake in the reasoning. Choose exactly one error type from the following categories.

## Error Types

**Error Type 1: Wrong ontology direction**
The hypothesis contains the wrong ontology direction.

**Error Type 2: Fall back to trivial hypotheses**
The hypothesis is the observation itself.

**Error Type 3: Ignore the ontology and produce unnecessary hypotheses**
The hypothesis is technically correct but unnecessary given the ontology.

**Error Type 4: Hallucinated entities**
The reasoning relies on non-existent entities.

[Include 1 example per type from paper H.1]

## Now evaluate the following case

Question: {question}
Model Reasoning: {model_output}
Correct Answer: {correct_answer}

## Output Format

Error Type: <one of: Wrong ontology direction | Fall back to trivial hypotheses | Ignore the ontology and produce unnecessary hypotheses | Hallucinated entities>
"""
```

Note: the paper's error type list in the body (Section 4.5) and the prompt (Figure 10) differ slightly in ordering/numbering. Use the ordering from the prompt in Figure 10 for consistency.

### 5.3: Scope of classification and validation

Don't classify every example — only those where `is_correct_strong=False`.

**Two-stage validation before full run:**

**Stage 1 — Model agreement check.** Before running nano on the full ~31,600 failures, sample 200 failures (covering all heights and tasks roughly proportionally) and classify each with both `gpt-5.4-nano` and `gpt-5.4-mini`. Compute inter-model agreement rate.
- **Agreement ≥90%:** proceed with nano for the full run. Expected cost ~$15.
- **Agreement 85-90%:** proceed with nano but flag in the final report that ~10-15% of classifications may have low confidence.
- **Agreement <85%:** step up to `gpt-5.4-mini` for the full run. Expected cost ~$50.

**Stage 2 — Budget sampling (optional).** If minimizing cost further is desired, sample `min(500, N_failures)` failures per (task, height, model) bucket instead of classifying all failures. This reduces the total to ~8,000 classifications (~$4 with nano, ~$12 with mini). Only worth doing if budget is a serious constraint — the full classification is already cheap at nano tier.

**Expected failure volumes** (for context):
- Per-model, failures at roughly these rates: h=1 ~10%, h=2 ~40%, h=3 ~75%, h=4 ~95%
- Total failures per (task, model): ~100 + 800 + 2250 + 4750 = ~7900
- Across 4 (model × task) combinations: ~31,600 failures total

### 5.4: Storage

Add `error_type` as a top-level field in the JSONL rows. Values:
- `null` for correct examples
- One of `{"wrong_direction", "trivial", "unnecessary", "hallucinated", "unclassified"}` for incorrect examples
- `"unclassified"` when GPT-4o returns an unparseable response or the example wasn't classified (e.g., due to budget sampling)

### 5.5: Analysis

Reproduce paper Figure 9 layout: stacked bar chart showing distribution of error types at each ontology tree height, for each model. This is a standalone deliverable for the Behavioral Results section.

Also compute: **does error type distribution correlate with `has_direct_member`?** If Type 2 errors (unnecessary hypotheses) happen mostly when `has_direct_member=False`, that's evidence the shortcut confound affects the error mode, not just accuracy.

---

## Interface Contract with Teammate B

Teammate B consumes the JSONL files produced in Phase 2.2 and 5.4. Their probe training pipeline needs:

1. **Prompt text** (`prompt_text` field) — to feed to the model and extract activations at specified token positions.
2. **Binary label** (`is_correct_strong` field) — primary probe training target.
3. **Error type** (`error_type` field) — optional multi-class probe training target.
4. **Height** (`height` field) — for stratified analysis by depth.
5. **Task** (`task` field) — for cross-task transfer tests.
6. **Structural fields** — for within-dataset slicing during analysis.
7. **Structured FOL** (`ontology_fol_structured` field) — available for symbolic analysis if needed, ignore otherwise.

**Pre-CoT token position:** Following Cox et al., probes are trained on activations at the last token before chain-of-thought begins. For Gemma 3, the exact token position depends on the chat template. Teammate B's workflow:

1. Load the Gemma 3 tokenizer with chat template support.
2. Construct messages from the JSONL row: `[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]`
3. Apply the chat template with `tokenize=False, add_generation_prompt=True`. This produces the exact string that was fed to the model during inference (since Teammate A used the same chat template via the OpenAI-compatible chat completions API).
4. Tokenize. The last token in this sequence is the "last pre-CoT token" — typically the end-of-turn marker or the start-of-assistant token.
5. Run a forward pass (no generation needed) and extract residual stream activations at that position.

Teammate A should verify during Phase 1 that `prompt_text` and `system_prompt` in the JSONL can be fed back through the Gemma 3 chat template to reproduce the exact input the model originally received. One way to check: at temperature=0, re-running inference with the reconstructed input should produce the same `model_output` byte-for-byte.

**Handoff deliverable:** A README.md in the `results/` directory explaining:
- How many examples per config
- What fields are in each JSONL row (including nullable `error_type`)
- Known limitations (e.g., low positive count at h=4 for 4B)
- Seed values used (for reproducibility)
- Any examples where error_type classification was skipped

---

## Sanity Checks Before Handoff

Before passing the dataset to Teammate B, verify all of the following:

**Integrity checks:**
- [ ] Each JSONL file has the expected number of rows (sum of per-height counts).
- [ ] All rows have all required fields.
- [ ] `is_correct_strong` is boolean; `parse_failed` is boolean; `error_type` and `failure_mode` are strings or null.
- [ ] Parse failure rate is below 5% at every (model, task, height) combination.
- [ ] If parse failure rate is higher than expected, `failure_mode` breakdown identifies the dominant cause.
- [ ] Structured FOL dicts are well-formed JSON (all keys are strings, all values are list/dict/string/null).

**Sanity checks:**
- [ ] Accuracy at h=1 is high (>70%) for both models on both tasks.
- [ ] Accuracy decreases monotonically with height for infer_property (may be non-monotonic for infer_subtype on weak accuracy metric, but should decrease for strong accuracy).
- [ ] Gemma 3 27B beats Gemma 3 4B at most (height, task) combinations.
- [ ] `has_direct_member` rate is 100% at every cell (post-patch; pre-patch drafts of this plan expected ~70-95%, which reflected an upstream `normalize_to_singular` bug on the entity names Thomas/Charles/James/Nicholas — see Phase 2.1 note).
- [ ] Error Type 2 is the most common error type across all heights (matches paper's headline finding).

**Class balance check:**
- [ ] At least 100 positive examples at each (model, task, height) except possibly (4B, h=4) which may have fewer.
- [ ] At least 100 negative examples at each (model, task, height) except possibly (27B, h=1) where near-ceiling accuracy is expected.

If any check fails, document why before handing off.

---

## Implementation Notes

### Task type naming

The beyond-deduction code uses these `task_type` strings (see `run_experiments.py` line ~96):
- `'property'` → infer_property (v2 paper's Task 1)
- `'membership'` → infer_membership_relation (v2 paper's Task 2, SKIP this)
- `'ontology'` → infer_subtype_relation (v2 paper's Task 3)

Map these consistently in the output JSONL using the v2 paper names for clarity:
- Code `'property'` → JSONL `"task": "infer_property"`
- Code `'ontology'` → JSONL `"task": "infer_subtype"`

### Where the shortcut-detection infrastructure lives

**NOT in `mi/` — that directory has shortcut-creation code for matched pairs (Set 4 generators), not shortcut-detection for arbitrary examples.**

**The reusable primitive IS in `benchmark/evaluate.py`:** the `KnowledgeBase` class and its `get_all_concepts_for_entity(entity)` method, which returns `(concept, proof_depth)` tuples. Proof depth of 1 means direct membership, which is exactly what `has_direct_member` should detect.

Verified post-implementation: computing `has_direct_member` using this method **through the `src/bd_path.py` patched normalizer** on all 44,000 shipped rows yields 100% True in every cell (not 92%). The 92% figure cited earlier — including the prior Gemma 2 9B "92/100" — reflected the upstream `normalize_to_singular` bug on four entity names (Thomas/Charles/James/Nicholas), not a property of the generator. See Phase 2.1 and `results/README.md` for the full write-up.

### Pseudo-root handling

When `recover_ontology=True` (subtype relation task), the code adds a `pseudo_root` node above the real root. This is structural and affects how `theories` is formatted. Make sure structural annotations account for this — specifically, the "target concept" for subtype tasks is a subtype relation between two concepts, not just a single concept.

### Morphology / name pool

The code has 90 concept names and ~100 entity names. At maximum tree height (h=4) with multiple children per node, an example uses ~30-50 concepts. There's plenty of combinatorial room for unique examples up to 5000+ per height (verified: 5000 examples at h=4 gives 100% uniqueness in generation testing).

### Evaluation quirks

For single-hypothesis examples, `compute_strong_accuracy` returns 1 or 0, and `compute_weak_accuracy` is normally equal to it (both equal 1 iff the first hypothesis is correct). The existing code (line ~170) sets `weak_acc = 1` when `strong_acc == 1`, which is correct behavior. The `quality_score` will usually be 1.0 for correct single-hypothesis examples.

### Test-time prompt format

Use the exact system prompt from `run_experiments.py` (including the "assitant" typo — this matches the paper's `generate.py` and ensures comparability). Use `temperature=0` for deterministic outputs across runs.

---

## Deliverables Checklist

At the end of Phase 3:
- [ ] `gemma3_4b_infer_property.jsonl` (~11,000 rows)
- [ ] `gemma3_27b_infer_property.jsonl` (~11,000 rows)
- [ ] `gemma3_4b_infer_subtype.jsonl` (~11,000 rows)
- [ ] `gemma3_27b_infer_subtype.jsonl` (~11,000 rows)
- [ ] `summary_accuracy.json`
- [ ] `summary_by_structure.json`
- [ ] `README.md` in results directory

At the end of Phase 4:
- [ ] Figure reproducing paper Figure 3 layout (accuracy-vs-depth, both tasks, both models)
- [ ] Structural slicing analysis (accuracy by `has_direct_member`)
- [ ] Output strategy analysis (first-hypothesis entity-level vs concept-level)
- [ ] Draft text for the final report's Behavioral Results section

At the end of Phase 5:
- [ ] `error_type` field populated in all JSONL files
- [ ] Figure 9-style stacked bar chart of error type distribution by height and model
- [ ] Analysis of error type × structural feature correlations
