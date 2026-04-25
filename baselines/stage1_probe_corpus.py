"""Stage 1 probe corpus container: typed bundle of ProbeRecords with kind-dispatch validation.

``Stage1ProbeCorpus`` wraps a list of ``ProbeRecord`` objects and a ``kind`` discriminator
(``"france"`` or ``"arab_spring"``), providing a single ``validate()`` entry point that
dispatches to the appropriate gate annotation validator and coverage checker.

``validate(manifest=None)``:
  - ``france``: runs ``validate_france_plumbing_gate_annotations`` +
    ``validate_gate_coverage(..., FRANCE_PLUMBING_TRAINING_CONTEXT_ID)``.
    ``manifest`` is accepted but silently ignored (France probes do not require hint
    resolution against a warehouse manifest for gate plumbing tests).
  - ``arab_spring``: runs ``validate_arab_spring_probe_gate_annotations`` +
    ``validate_gate_coverage(..., ARAB_SPRING_TRAINING_CONTEXT_ID)``.
    When ``manifest`` is not ``None``, also runs ``validate_probe_hints_against_manifest``
    to catch entity hints that do not resolve in the loaded warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from baselines.arab_spring_probes import (
    ARAB_SPRING_TRAINING_CONTEXT_ID,
    build_arab_spring_probe_corpus,
    validate_arab_spring_probe_gate_annotations,
    validate_probe_hints_against_manifest,
)
from baselines.france_plumbing_probes import (
    FRANCE_PLUMBING_TRAINING_CONTEXT_ID,
    build_france_plumbing_probe_corpus,
    validate_france_plumbing_gate_annotations,
    validate_gate_coverage,
)
from schemas.graph_builder_probe import ProbeRecord
from schemas.graph_builder_warehouse import NodeWarehouseManifest


@dataclass
class Stage1ProbeCorpus:
    """Typed container for a Stage 1 probe bundle.

    Attributes:
        kind: Corpus kind — ``"france"`` or ``"arab_spring"``.
        probes: Ordered list of ``ProbeRecord`` objects in this corpus.
    """

    kind: Literal["france", "arab_spring"]
    probes: list[ProbeRecord] = field(default_factory=list)

    def validate(self, manifest: NodeWarehouseManifest | None = None) -> None:
        """Validate gate annotations and coverage for this corpus.

        Dispatches based on ``kind``:
        - ``france``: gate annotation check + gate coverage check.
          ``manifest`` is accepted but unused.
        - ``arab_spring``: gate annotation check + gate coverage check.
          When ``manifest`` is provided, additionally validates all entity_hints
          resolve in the manifest hint index.

        Raises:
            ValueError: on any validation failure (single error per check, listing all failures).
        """
        if self.kind == "france":
            self._validate_france()
        elif self.kind == "arab_spring":
            self._validate_arab_spring(manifest)
        else:
            raise ValueError(f"Unknown corpus kind: {self.kind!r}")

    def _validate_france(self) -> None:
        validate_france_plumbing_gate_annotations(self.probes)
        validate_gate_coverage(self.probes, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)

    def _validate_arab_spring(self, manifest: NodeWarehouseManifest | None) -> None:
        validate_arab_spring_probe_gate_annotations(self.probes)
        validate_gate_coverage(self.probes, training_context_id=ARAB_SPRING_TRAINING_CONTEXT_ID)
        if manifest is not None:
            validate_probe_hints_against_manifest(self.probes, manifest)

    @classmethod
    def france_default(cls) -> Stage1ProbeCorpus:
        """Build the default France plumbing corpus (~216 rows)."""
        return cls(kind="france", probes=build_france_plumbing_probe_corpus())

    @classmethod
    def arab_spring_default(cls) -> Stage1ProbeCorpus:
        """Build the default Arab Spring probe corpus (>= 150 rows)."""
        return cls(kind="arab_spring", probes=build_arab_spring_probe_corpus())


__all__ = ["Stage1ProbeCorpus"]
