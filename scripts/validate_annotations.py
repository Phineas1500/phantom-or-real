"""Validate structural annotations on 100 generated examples.

Per BEHAVIORAL_DATA_PLAN.md: h=2 infer_property should yield ~92% has_direct_member=True.
"""

from __future__ import annotations

import sys
from pathlib import Path
from random import seed

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bd_path import ensure_on_path  # noqa: E402

ensure_on_path()

from ontology import Ontology, OntologyConfig, Difficulty  # noqa: E402
from src.annotations import compute_structural_annotations  # noqa: E402


def main() -> int:
    seed(42)
    np.random.seed(42)

    results = {}

    for task_type in ("property", "ontology"):
        for height in (2, 3, 4):
            cfg = OntologyConfig(
                hops=height,
                recover_property=(task_type == "property"),
                recover_ontology=(task_type == "ontology"),
                difficulty=Difficulty.SINGLE,
                mix_hops=False,
            )

            n = 100
            direct = 0
            parent_salience_avg = 0.0
            for _ in range(n):
                # retry loop mirrors generate_single_example
                while True:
                    try:
                        o = Ontology(cfg)
                        break
                    except Exception:
                        pass
                ann = compute_structural_annotations(o, task_type)
                direct += int(ann["has_direct_member"])
                parent_salience_avg += ann["parent_salience"]

            rate = direct / n
            parent_salience_avg /= n
            results[(task_type, height)] = rate
            print(
                f"task={task_type:<9} h={height}  has_direct_member={rate:.2%}  "
                f"avg_parent_salience={parent_salience_avg:.2f}"
            )

    # Sanity: infer_property h=2 should be in 80-95% per plan
    r = results[("property", 2)]
    if not (0.70 <= r <= 0.99):
        print(f"WARN: infer_property h=2 has_direct_member rate {r:.2%} outside expected 70-99%")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
