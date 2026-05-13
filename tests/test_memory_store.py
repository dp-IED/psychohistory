from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from harness.memory_schema import ConceptualPattern, EpisodicRecord, StructuralFact, ToolCallRecord
from harness.memory_store import JsonlMemoryStore, MemoryStore, NullMemoryStore


@pytest.fixture
def tmp_store_dir(tmp_path: Path) -> Path:
    return tmp_path / "memory-store"


def _tool_call() -> ToolCallRecord:
    return ToolCallRecord(
        tool_name="web_search",
        query="ceasefire mediation timeline",
        as_of_time="2026-05-12T00:00:00Z",
        evidence_count=3,
        notes="seed retrieval",
    )


def _episode(
    *,
    job_id: str,
    market_family: str = "event_negotiation",
    resolution_date: date = date(2026, 6, 1),
    cutoff_date: date = date(2026, 5, 20),
) -> EpisodicRecord:
    return EpisodicRecord(
        job_id=job_id,
        market_id=f"market-{job_id}",
        market_family=market_family,
        question="Will a ceasefire be announced before June 1?",
        resolution_date=resolution_date,
        cutoff_date=cutoff_date,
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


def test_null_store_protocol_and_noop_behavior() -> None:
    store: MemoryStore = NullMemoryStore()

    assert isinstance(store, MemoryStore)

    episode = _episode(job_id="job-null")
    pattern = _pattern()
    fact = _fact()

    store.write_episode(episode)
    store.update_episode_brier("job-null", brier_score=0.2, misses=["check-A"])
    store.write_pattern(pattern)
    store.write_fact(fact)

    assert store.read_recent_episodes("event_negotiation", 5) == []
    assert store.read_episode_by_id("job-null") is None
    assert store.read_patterns("event_negotiation") == []
    assert store.read_facts("Mediator X") == []


def test_jsonl_round_trip_all_layers(tmp_store_dir: Path) -> None:
    store = JsonlMemoryStore(tmp_store_dir)
    episode = _episode(job_id="job-001")
    pattern = _pattern()
    fact = _fact()

    store.write_episode(episode)
    store.write_pattern(pattern)
    store.write_fact(fact)

    assert store.read_recent_episodes("event_negotiation", 10) == [episode]
    assert store.read_patterns("event_negotiation") == [pattern]
    assert store.read_facts("Mediator X") == [fact]


def test_jsonl_update_episode_brier_persists(tmp_store_dir: Path) -> None:
    store = JsonlMemoryStore(tmp_store_dir)
    episode = _episode(job_id="job-002")
    store.write_episode(episode)

    misses = ["counterfactual_spoiler", "regional_spillover"]
    store.update_episode_brier("job-002", brier_score=0.31, misses=misses)

    [updated] = store.read_recent_episodes("event_negotiation", 10)
    assert updated.job_id == "job-002"
    assert updated.brier_score == 0.31
    assert updated.misses == misses


def test_jsonl_update_episode_brier_missing_job_raises(tmp_store_dir: Path) -> None:
    store = JsonlMemoryStore(tmp_store_dir)

    with pytest.raises(KeyError, match="job-missing"):
        store.update_episode_brier("job-missing", brier_score=0.5, misses=["x"])


def test_jsonl_read_episode_by_id_returns_matching_episode(tmp_store_dir: Path) -> None:
    store = JsonlMemoryStore(tmp_store_dir)
    episode = _episode(job_id="job-lookup")
    store.write_episode(episode)

    assert store.read_episode_by_id("job-lookup") == episode
    assert store.read_episode_by_id("job-absent") is None


def test_jsonl_read_recent_episodes_filters_family_and_orders_recency(tmp_store_dir: Path) -> None:
    store = JsonlMemoryStore(tmp_store_dir)

    older = _episode(
        job_id="job-older",
        market_family="event_negotiation",
        resolution_date=date(2026, 4, 1),
        cutoff_date=date(2026, 3, 1),
    )
    middle = _episode(
        job_id="job-middle",
        market_family="event_negotiation",
        resolution_date=date(2026, 5, 1),
        cutoff_date=date(2026, 4, 1),
    )
    newest = _episode(
        job_id="job-newest",
        market_family="event_negotiation",
        resolution_date=date(2026, 6, 1),
        cutoff_date=date(2026, 5, 1),
    )
    other_family = _episode(
        job_id="job-other",
        market_family="macro_rates",
        resolution_date=date(2026, 7, 1),
        cutoff_date=date(2026, 6, 1),
    )

    for episode in [older, middle, newest, other_family]:
        store.write_episode(episode)

    got = store.read_recent_episodes("event_negotiation", 2)

    assert [ep.job_id for ep in got] == ["job-newest", "job-middle"]


def test_jsonl_persists_across_instances(tmp_store_dir: Path) -> None:
    store1 = JsonlMemoryStore(tmp_store_dir)
    episode = _episode(job_id="job-persist")
    pattern = _pattern()
    fact = _fact()

    store1.write_episode(episode)
    store1.write_pattern(pattern)
    store1.write_fact(fact)

    store2 = JsonlMemoryStore(tmp_store_dir)

    assert store2.read_recent_episodes("event_negotiation", 10) == [episode]
    assert store2.read_patterns("event_negotiation") == [pattern]
    assert store2.read_facts("Mediator X") == [fact]


def test_jsonl_read_recent_episodes_raises_on_corrupt_jsonl_line(tmp_store_dir: Path) -> None:
    store = JsonlMemoryStore(tmp_store_dir)
    store.write_episode(_episode(job_id="job-good"))

    episodes_path = tmp_store_dir / "episodes.jsonl"
    with episodes_path.open("a", encoding="utf-8") as handle:
        handle.write('{"truncated": ')

    with pytest.raises(json.JSONDecodeError):
        store.read_recent_episodes("event_negotiation", 10)
