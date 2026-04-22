"""Pickle-stable container for a single generated InAbHyD example.

Kept in its own module so the fully qualified class name stays
`src.example.ExampleView` regardless of how generation was invoked
(`python -m src.generate_examples` vs programmatic import).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class ExampleView:
    """Lightweight, pickle-safe replacement for the Ontology object.

    Downstream code (annotations, export, inference scoring) reads the same
    attribute names via duck typing: theories, observations, hypotheses,
    fol_theories, fol_observations, fol_hypotheses, config.hops.
    """

    theories: str
    observations: str
    hypotheses: str
    fol_theories: str
    fol_observations: str
    fol_hypotheses: str
    hops: int

    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace(hops=self.hops)

    @classmethod
    def from_ontology(cls, o) -> "ExampleView":
        return cls(
            theories=o.theories,
            observations=o.observations,
            hypotheses=o.hypotheses,
            fol_theories=getattr(o, "fol_theories", ""),
            fol_observations=getattr(o, "fol_observations", ""),
            fol_hypotheses=getattr(o, "fol_hypotheses", ""),
            hops=o.config.hops,
        )
