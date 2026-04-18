from __future__ import annotations

import pytest

from baselines.france_plumbing_probes import (
    FRANCE_BASE_PROBE_DEFS,
    FRANCE_PLUMBING_TRAINING_CONTEXT_ID,
    build_france_plumbing_probe_corpus,
    validate_gate_coverage,
)
from schemas import AssumptionEmphasis, ProbeRecord


def test_base_probe_defs_count_and_model() -> None:
    assert len(FRANCE_BASE_PROBE_DEFS) == 20
    for row in FRANCE_BASE_PROBE_DEFS:
        assert isinstance(row, ProbeRecord)


def test_corpus_size_and_gate_coverage() -> None:
    corpus = build_france_plumbing_probe_corpus()
    assert len(corpus) >= 180
    validate_gate_coverage(corpus, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)


def test_validate_gate_coverage_raises_when_gate_missing() -> None:
    corpus = build_france_plumbing_probe_corpus()
    only_persistence = [p for p in corpus if p.assumption_emphasis == AssumptionEmphasis.PERSISTENCE]
    assert only_persistence
    with pytest.raises(ValueError, match=r"missing gate\(s\)"):
        validate_gate_coverage(only_persistence, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)


def test_validate_gate_coverage_error_names_missing_gates() -> None:
    corpus = build_france_plumbing_probe_corpus()
    subset = [p for p in corpus if p.assumption_emphasis != AssumptionEmphasis.COORDINATION]
    with pytest.raises(ValueError, match="Coordination"):
        validate_gate_coverage(subset, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)
