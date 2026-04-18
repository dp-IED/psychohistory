from __future__ import annotations

import warnings

import pytest

from baselines.france_plumbing_probes import (
    FRANCE_BASE_PROBE_DEFS,
    FRANCE_PLUMBING_TRAINING_CONTEXT_ID,
    build_france_plumbing_probe_corpus,
    validate_france_plumbing_gate_annotations,
    validate_gate_coverage,
)
from schemas import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)


def test_base_probe_defs_count_and_model() -> None:
    assert len(FRANCE_BASE_PROBE_DEFS) == 20
    for row in FRANCE_BASE_PROBE_DEFS:
        assert isinstance(row, ProbeRecord)
        assert row.generation_meta.assumption_gate_coverage == row.assumption_emphasis


def test_corpus_size_and_gate_coverage() -> None:
    corpus = build_france_plumbing_probe_corpus()
    assert len(corpus) == 216
    validate_france_plumbing_gate_annotations(corpus)
    validate_gate_coverage(corpus, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)


def test_validate_gate_coverage_raises_when_gate_missing() -> None:
    corpus = build_france_plumbing_probe_corpus()
    only_persistence = [p for p in corpus if p.assumption_emphasis == AssumptionEmphasis.PERSISTENCE]
    assert only_persistence
    with pytest.raises(ValueError, match=r"missing gate\(s\)"):
        validate_gate_coverage(only_persistence, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)


def test_validate_france_plumbing_gate_annotations_requires_meta() -> None:
    d = FRANCE_BASE_PROBE_DEFS[0].origin
    bad = ProbeRecord(
        probe_id="no-meta-gate",
        origin=d,
        nl_text="y",
        q_struct=QStructV0(
            actor_state=ActorStateQuery(geography=["France"], actor_type=["government"], as_of=d),
        ),
        lens_params=LensParamsV0(),
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        generation_meta=GenerationMeta(template_id="t", generator_version="1"),
    )
    assert bad.generation_meta.assumption_gate_coverage is None
    with pytest.raises(ValueError, match="no-meta-gate"):
        validate_france_plumbing_gate_annotations([bad])


def test_validate_france_plumbing_gate_annotations_lists_all_offenders() -> None:
    d = FRANCE_BASE_PROBE_DEFS[0].origin
    base_kw = dict(
        origin=d,
        nl_text="y",
        q_struct=QStructV0(
            actor_state=ActorStateQuery(geography=["France"], actor_type=["government"], as_of=d),
        ),
        lens_params=LensParamsV0(),
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        generation_meta=GenerationMeta(template_id="t", generator_version="1"),
    )
    a = ProbeRecord(probe_id="missing-a", **base_kw)
    b = ProbeRecord(probe_id="missing-b", **base_kw)
    with pytest.raises(ValueError, match="missing-a, missing-b"):
        validate_france_plumbing_gate_annotations([a, b])


def test_validate_gate_coverage_emits_single_deprecation_when_meta_missing() -> None:
    corpus = build_france_plumbing_probe_corpus()
    p0 = corpus[0]
    stripped = p0.model_copy(
        update={
            "generation_meta": p0.generation_meta.model_copy(
                update={"assumption_gate_coverage": None},
            ),
        },
    )
    corpus2 = [stripped] + corpus[1:]
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        validate_gate_coverage(corpus2, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)
    dep = [w for w in wrec if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1
    assert "assumption_gate_coverage" in str(dep[0].message)


def test_validate_gate_coverage_error_names_missing_gates() -> None:
    corpus = build_france_plumbing_probe_corpus()
    subset = [p for p in corpus if p.assumption_emphasis != AssumptionEmphasis.COORDINATION]
    with pytest.raises(ValueError, match="Coordination"):
        validate_gate_coverage(subset, training_context_id=FRANCE_PLUMBING_TRAINING_CONTEXT_ID)
