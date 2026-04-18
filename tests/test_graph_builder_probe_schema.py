from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from schemas.graph_builder_probe import (
    ActorStateQuery,
    AssumptionEmphasis,
    CompileTrace,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)


def test_probe_record_rejects_gate_coverage_mismatch() -> None:
    origin = date(2024, 5, 1)
    with pytest.raises(ValidationError, match=r"assumption_gate_coverage must match"):
        ProbeRecord(
            probe_id="p-mismatch",
            origin=origin,
            nl_text="x",
            q_struct=QStructV0(
                actor_state=ActorStateQuery(
                    geography=["France"],
                    actor_type=["government"],
                    as_of=origin,
                ),
            ),
            lens_params=LensParamsV0(),
            assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
            generation_meta=GenerationMeta(
                template_id="t",
                generator_version="1",
                assumption_gate_coverage=AssumptionEmphasis.COORDINATION,
            ),
        )


def test_probe_record_valid_when_as_of_on_or_before_origin() -> None:
    origin = date(2024, 6, 15)
    as_of = date(2024, 6, 10)
    probe = ProbeRecord(
        probe_id="p-1",
        origin=origin,
        nl_text="Regional tension rising.",
        q_struct=QStructV0(
            actor_state=ActorStateQuery(
                geography=["Eastern Europe"],
                actor_type=["institution"],
                as_of=as_of,
            ),
        ),
        lens_params=LensParamsV0(horizon_days=7, context_snippet="weekly slice"),
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        generation_meta=GenerationMeta(
            template_id="t-a",
            generator_version="0.0.1",
            seed=42,
            assumption_gate_coverage=AssumptionEmphasis.PERSISTENCE,
        ),
    )
    assert probe.q_struct.actor_state.as_of == as_of
    assert probe.assumption_emphasis == AssumptionEmphasis.PERSISTENCE


def test_probe_record_same_day_as_of_allowed() -> None:
    d = date(2024, 1, 1)
    ProbeRecord(
        probe_id="p-same",
        origin=d,
        nl_text="ok",
        q_struct=QStructV0(actor_state=ActorStateQuery(geography=["X"], actor_type=["group"], as_of=d)),
        lens_params=LensParamsV0(),
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        generation_meta=GenerationMeta(
            template_id="t",
            generator_version="1",
            assumption_gate_coverage=AssumptionEmphasis.COORDINATION,
        ),
    )


def test_compile_trace_round_trip() -> None:
    trace = CompileTrace(input_hash="abc", validation_errors=["warn"], normalized_embedding_text=None)
    back = CompileTrace.model_validate(trace.model_dump())
    assert back.input_hash == "abc"
    assert back.validation_errors == ["warn"]


def test_probe_record_rejects_as_of_after_origin() -> None:
    origin = date(2024, 3, 1)
    as_of = date(2024, 3, 5)
    with pytest.raises(ValidationError, match=r"q_struct\.actor_state\.as_of"):
        ProbeRecord(
            probe_id="p-bad",
            origin=origin,
            nl_text="bad dates",
            q_struct=QStructV0(
                actor_state=ActorStateQuery(
                    geography=["Y"],
                    actor_type=["person"],
                    as_of=as_of,
                ),
            ),
            lens_params=LensParamsV0(),
            assumption_emphasis=AssumptionEmphasis.PRECURSOR,
            generation_meta=GenerationMeta(
                template_id="t",
                generator_version="1",
                assumption_gate_coverage=AssumptionEmphasis.PRECURSOR,
            ),
        )
