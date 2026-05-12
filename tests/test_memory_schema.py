from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from harness.memory_schema import ConceptualPattern, EpisodicRecord, StructuralFact, ToolCallRecord


def _tool_call() -> ToolCallRecord:
    return ToolCallRecord(
        tool_name="web_search",
        query="ceasefire mediation timeline",
        as_of_time="2026-05-12T00:00:00Z",
        evidence_count=3,
        notes="seed retrieval",
    )


def _episode() -> EpisodicRecord:
    return EpisodicRecord(
        job_id="job-123",
        market_id="market-abc",
        market_family="event_negotiation",
        question="Will a ceasefire be announced before June 1?",
        resolution_date=date(2026, 6, 1),
        cutoff_date=date(2026, 5, 20),
        blind_spot_checks_fired=["counterfactual_spoiler"],
        blind_spot_checks_skipped=["macro_liquidity_backdrop"],
        tool_calls=[_tool_call()],
        subgraph_node_count=42,
        gnn_score_trajectory=[0.41, 0.47, 0.52],
        final_p_yes=0.55,
        confidence_interval=(0.48, 0.62),
        brier_score=None,
        misses=[],
        notes="episode completed",
    )


def _pattern() -> ConceptualPattern:
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    return ConceptualPattern(
        pattern_id="p-001",
        name="Negotiation window under mediator pressure",
        description="Patterns where mediator pressure narrows spoiler pathways.",
        applicable_market_families=["event_negotiation"],
        evidence_job_ids=["job-123"],
        confidence=0.72,
        blind_spot_check_mapping="counterfactual_spoiler",
        created_at=now,
        last_reinforced_at=now,
        source="policy_patch",
    )


def _fact() -> StructuralFact:
    return StructuralFact(
        fact_id="f-001",
        subject="Mediator X",
        predicate="engaged_with",
        object="Faction Y",
        confidence=0.8,
        source_url="https://example.org/report",
        valid_from=date(2026, 1, 1),
        valid_until=date(2026, 12, 31),
        last_verified=date(2026, 5, 10),
    )


def test_construction_happy_path_all_types() -> None:
    episode = _episode()
    pattern = _pattern()
    fact = _fact()

    assert episode.brier_score is None
    assert pattern.source == "policy_patch"
    assert fact.predicate == "engaged_with"


def test_episodic_rejects_invalid_field_values() -> None:
    with pytest.raises(ValueError, match="cutoff_date"):
        EpisodicRecord(
            job_id="job",
            market_id="m",
            market_family="event_negotiation",
            question="q",
            resolution_date=date(2026, 1, 1),
            cutoff_date=date(2026, 1, 2),
            blind_spot_checks_fired=[],
            blind_spot_checks_skipped=[],
            tool_calls=[_tool_call()],
            subgraph_node_count=1,
            gnn_score_trajectory=[0.4],
            final_p_yes=0.5,
            confidence_interval=(0.4, 0.6),
            brier_score=None,
            misses=[],
            notes="",
        )

    with pytest.raises(ValueError, match="must be disjoint"):
        EpisodicRecord(
            job_id="job",
            market_id="m",
            market_family="event_negotiation",
            question="q",
            resolution_date=date(2026, 1, 2),
            cutoff_date=date(2026, 1, 1),
            blind_spot_checks_fired=["x"],
            blind_spot_checks_skipped=["x"],
            tool_calls=[_tool_call()],
            subgraph_node_count=1,
            gnn_score_trajectory=[0.4],
            final_p_yes=0.5,
            confidence_interval=(0.4, 0.6),
            brier_score=None,
            misses=[],
            notes="",
        )

    with pytest.raises(ValueError, match="final_p_yes"):
        EpisodicRecord(
            job_id="job",
            market_id="m",
            market_family="event_negotiation",
            question="q",
            resolution_date=date(2026, 1, 2),
            cutoff_date=date(2026, 1, 1),
            blind_spot_checks_fired=[],
            blind_spot_checks_skipped=[],
            tool_calls=[_tool_call()],
            subgraph_node_count=1,
            gnn_score_trajectory=[0.4],
            final_p_yes=1.5,
            confidence_interval=(0.4, 0.6),
            brier_score=None,
            misses=[],
            notes="",
        )


def test_episodic_allows_empty_gnn_trajectory_and_validates_confidence_interval() -> None:
    episode = EpisodicRecord(
        job_id="job-empty-traj",
        market_id="m-empty",
        market_family="event_negotiation",
        question="q",
        resolution_date=date(2026, 2, 1),
        cutoff_date=date(2026, 1, 1),
        blind_spot_checks_fired=[],
        blind_spot_checks_skipped=[],
        tool_calls=[_tool_call()],
        subgraph_node_count=0,
        gnn_score_trajectory=[],
        final_p_yes=0.5,
        confidence_interval=(0.3, 0.7),
        brier_score=None,
        misses=[],
        notes="",
    )
    assert episode.gnn_score_trajectory == []

    with pytest.raises(ValueError, match="confidence_interval"):
        EpisodicRecord(
            job_id="job-bad-ci",
            market_id="m",
            market_family="event_negotiation",
            question="q",
            resolution_date=date(2026, 2, 1),
            cutoff_date=date(2026, 1, 1),
            blind_spot_checks_fired=[],
            blind_spot_checks_skipped=[],
            tool_calls=[_tool_call()],
            subgraph_node_count=0,
            gnn_score_trajectory=[],
            final_p_yes=0.5,
            confidence_interval=(0.8, 0.2),
            brier_score=None,
            misses=[],
            notes="",
        )


def test_conceptual_pattern_source_gate_and_timestamp_validation() -> None:
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="source"):
        ConceptualPattern(
            pattern_id="p",
            name="n",
            description="d",
            applicable_market_families=["event_negotiation"],
            evidence_job_ids=[],
            confidence=0.5,
            blind_spot_check_mapping=None,
            created_at=now,
            last_reinforced_at=now,
            source="unknown",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="last_reinforced_at"):
        ConceptualPattern(
            pattern_id="p",
            name="n",
            description="d",
            applicable_market_families=["event_negotiation"],
            evidence_job_ids=[],
            confidence=0.5,
            blind_spot_check_mapping=None,
            created_at=now,
            last_reinforced_at=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
            source="hand_authored",
        )


def test_structural_fact_rejects_invalid_temporal_window() -> None:
    with pytest.raises(ValueError, match="valid_from"):
        StructuralFact(
            fact_id="f",
            subject="s",
            predicate="p",
            object="o",
            confidence=0.5,
            source_url=None,
            valid_from=date(2026, 2, 1),
            valid_until=date(2026, 1, 1),
            last_verified=date(2026, 2, 1),
        )


def test_round_trip_serialization_all_types() -> None:
    episode = _episode()
    pattern = _pattern()
    fact = _fact()
    tool = _tool_call()

    assert EpisodicRecord.from_dict(episode.to_dict()) == episode
    mixed_payload = episode.to_dict()
    mixed_payload["tool_calls"] = [_tool_call()]
    assert EpisodicRecord.from_dict(mixed_payload) == episode
    assert ConceptualPattern.from_dict(pattern.to_dict()) == pattern
    assert StructuralFact.from_dict(fact.to_dict()) == fact
    assert ToolCallRecord.from_dict(tool.to_dict()) == tool
