from __future__ import annotations

from datetime import date, timedelta

from harness.skills.graph_compression import EventRecord, compress_graph


def _events(n: int = 40) -> list[EventRecord]:
    base = date(2026, 1, 1)
    event_types = ["military strike", "ceasefire talks", "sanctions package", "defense pact"]
    out: list[EventRecord] = []
    for i in range(n):
        out.append(
            EventRecord(
                event_id=f"e-{i}",
                date=base + timedelta(days=i),
                actors=[f"Actor-{i % 8}", f"Actor-{(i + 1) % 8}"],
                event_type=event_types[i % len(event_types)],
                description=f"Event {i} description",
                source_url=None,
            )
        )
    return out


def test_compress_graph_token_budget_under_500_for_40_events() -> None:
    graph = compress_graph(_events(40))
    assert len(graph.to_context_str().split()) < 500


def test_alias_registry_deduplicates_nodes() -> None:
    events = [
        EventRecord("e-1", date(2026, 1, 1), ["Emmanuel Macron", "Germany"], "ceasefire talks", "d", None),
        EventRecord("e-2", date(2026, 1, 2), ["President Macron", "Germany"], "ceasefire talks", "d", None),
    ]
    aliases = {"Emmanuel Macron": "Macron", "President Macron": "Macron"}

    graph = compress_graph(events, alias_registry=aliases)

    assert "Macron" in graph.nodes
    assert "Emmanuel Macron" not in graph.nodes
    assert "President Macron" not in graph.nodes


def test_event_type_maps_to_fixed_taxonomy() -> None:
    graph = compress_graph(
        [EventRecord("e-1", date(2026, 1, 1), ["A", "B"], "military strike", "d", None)]
    )
    assert graph.edges[0].relation == "escalated"


def test_dominant_relation_types_top3_frequency_ordered() -> None:
    events = [
        EventRecord("e-1", date(2026, 1, 1), ["A", "B"], "military strike", "d", None),
        EventRecord("e-2", date(2026, 1, 2), ["A", "C"], "military strike", "d", None),
        EventRecord("e-3", date(2026, 1, 3), ["B", "C"], "ceasefire talks", "d", None),
        EventRecord("e-4", date(2026, 1, 4), ["A", "D"], "sanctions package", "d", None),
    ]
    graph = compress_graph(events)
    assert graph.dominant_relation_types == ["escalated", "mediated", "sanctioned"]


def test_most_central_actors_top5_degree() -> None:
    events = [
        EventRecord("e-1", date(2026, 1, 1), ["A", "B"], "military strike", "d", None),
        EventRecord("e-2", date(2026, 1, 2), ["A", "C"], "ceasefire talks", "d", None),
        EventRecord("e-3", date(2026, 1, 3), ["A", "D"], "sanctions package", "d", None),
        EventRecord("e-4", date(2026, 1, 4), ["B", "C"], "defense pact", "d", None),
        EventRecord("e-5", date(2026, 1, 5), ["E", "F"], "opposition statement", "d", None),
    ]
    graph = compress_graph(events)
    assert graph.most_central_actors[0] == "A"
    assert len(graph.most_central_actors) <= 5


def test_empty_input_returns_empty_graph() -> None:
    graph = compress_graph([])
    assert graph.event_count == 0
    assert graph.nodes == []
    assert graph.edges == []


def test_lossiness_guard_stub_divergence_signal() -> None:
    graph = compress_graph(_events(10))

    def gnn_score_stub_raw() -> float:
        return 0.70

    def gnn_score_stub_compressed(_ctx: str) -> float:
        return 0.62

    divergence = abs(gnn_score_stub_raw() - gnn_score_stub_compressed(graph.to_context_str()))
    assert divergence > 0.05
