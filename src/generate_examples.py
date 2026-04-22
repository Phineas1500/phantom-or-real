"""Phase 3.3: generate all examples up-front and save to pickles.

Decouples generation from inference so a failed inference run does not lose
the generation state. Deterministic per (task, height) via get_seed().

We do NOT pickle the full Ontology object tree. Its nodes use hash-by-name
and dict-of-node internals which break on pickle reconstruction (an upstream
quirk — `pickle.load` crashes in __hash__ before __init__ restores `name`).
Instead we serialize a lightweight dict capturing only the fields every
downstream consumer needs: theories/observations/hypotheses strings, their
FOL-string counterparts, and the tree height. All scoring, annotation, and
JSONL export code accesses these through an `ExampleView` shim that quacks
like the original Ontology.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from random import seed as random_seed

import numpy as np

from .bd_path import ensure_on_path

ensure_on_path()

from ontology import Ontology, OntologyConfig, Difficulty  # noqa: E402

from .config import HEIGHTS, HEIGHT_SAMPLE_SIZES, TASK_CODES, get_seed
from .example import ExampleView


def _make_config(task_type: str, height: int) -> OntologyConfig:
    return OntologyConfig(
        hops=height,
        recover_membership=(task_type == "membership"),
        recover_ontology=(task_type == "ontology"),
        recover_property=(task_type == "property"),
        difficulty=Difficulty.SINGLE,
        mix_hops=False,
    )


def generate_one(cfg: OntologyConfig) -> Ontology:
    while True:
        try:
            return Ontology(cfg)
        except Exception:
            # same retry pattern as run_experiments.generate_single_example
            continue


def generate_and_save(
    task_type: str,
    height: int,
    n: int,
    out_dir: Path,
) -> Path:
    s = get_seed(task_type, height)
    random_seed(s)
    np.random.seed(s)

    cfg = _make_config(task_type, height)
    examples = [ExampleView.from_ontology(generate_one(cfg)) for _ in range(n)]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"examples_{task_type}_h{height}.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"seed": s, "task_type": task_type, "height": height, "examples": examples}, f)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate InAbHyD examples")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--counts",
        choices=["full", "pilot"],
        default="full",
        help="full = per BEHAVIORAL_DATA_PLAN.md 2.4; pilot = 50 per height",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_CODES),
        choices=list(TASK_CODES),
    )
    parser.add_argument(
        "--heights",
        nargs="+",
        type=int,
        default=list(HEIGHTS),
    )
    args = parser.parse_args()

    counts = HEIGHT_SAMPLE_SIZES if args.counts == "full" else {h: 50 for h in HEIGHTS}

    for task in args.tasks:
        for h in args.heights:
            n = counts[h]
            print(f"Generating {task} h={h} n={n} ...", flush=True)
            out = generate_and_save(task, h, n, args.out_dir / args.counts)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
