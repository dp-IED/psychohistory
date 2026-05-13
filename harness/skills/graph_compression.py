from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Literal

Relation = Literal["escalated", "de-escalated", "mediated", "sanctioned", "allied", "opposed"]

EVENT_TYPE_TO_RELATION: dict[str, Relation] = {
    "military strike": "escalated",
    "troop mobilization": "escalated",
    "clash": "escalated",
    "ceasefire": "de-escalated",
    "de-escalation": "de-escalated",
    "ceasefire talks": "mediated",
    "mediation": "mediated",
    "sanctions package": "sanctioned",
    "sanctions": "sanctioned",
    "defense pact": "allied",
    "alliance": "allied",
    "opposition statement": "opposed",
    "condemnation": "opposed",
}


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    date: date
    actors: list[str]
    event_type: str
    description: str
    source_url: str | None


@dataclass(frozen=True)
class CompressedEdge:
    source: str
    target: str
    relation: Relation
    event_ids: list[str]


@dataclass(frozen=True)
class CompressedGraph:
    nodes: list[str]
    edges: list[CompressedEdge]
    event_count: int
    date_range: tuple[date, date] | None
    dominant_relation_types: list[str]
    most_central_actors: list[str]

    def to_context_str(self) -> str:
        if not self.nodes and not self.edges:
            return "graph: empty"
        dr = "-" if self.date_range is None else f"{self.date_range[0].isoformat()}..{self.date_range[1].isoformat()}"
        lines = [
            f"events={self.event_count} range={dr}",
            f"nodes({len(self.nodes)}): " + ", ".join(self.nodes[:25]),
            "dominant_relations: " + ", ".join(self.dominant_relation_types),
            "central_actors: " + ", ".join(self.most_central_actors),
            "edges:",
        ]
        for edge in self.edges[:40]:
            lines.append(f"- {edge.source}->{edge.target} {edge.relation} [{','.join(edge.event_ids[:4])}]")
        return "\n".join(lines)


def _relation_from_event_type(event_type: str) -> Relation:
    key = event_type.strip().lower()
    if key in EVENT_TYPE_TO_RELATION:
        return EVENT_TYPE_TO_RELATION[key]
    if "strike" in key or "attack" in key or "war" in key:
        return "escalated"
    if "ceasefire" in key or "truce" in key:
        return "de-escalated"
    if "talk" in key or "negot" in key or "mediat" in key:
        return "mediated"
    if "sanction" in key:
        return "sanctioned"
    if "alliance" in key or "pact" in key:
        return "allied"
    return "opposed"


def _canon(name: str, aliases: dict[str, str] | None) -> str:
    n = name.strip()
    if not n:
        return n
    if aliases is None:
        return n
    return aliases.get(n, n)


def compress_graph(events: list[EventRecord], alias_registry: dict[str, str] | None = None) -> CompressedGraph:
    if not events:
        return CompressedGraph(nodes=[], edges=[], event_count=0, date_range=None, dominant_relation_types=[], most_central_actors=[])

    edge_events: dict[tuple[str, str, Relation], list[str]] = defaultdict(list)
    relation_counter: Counter[str] = Counter()
    node_set: set[str] = set()
    dates: list[date] = []

    for event in events:
        dates.append(event.date)
        relation = _relation_from_event_type(event.event_type)
        relation_counter[relation] += 1

        actors = [_canon(a, alias_registry) for a in event.actors if _canon(a, alias_registry)]
        uniq = sorted(set(actors))
        for a in uniq:
            node_set.add(a)

        if len(uniq) < 2:
            continue
        # pair first two actors for compactness
        source, target = uniq[0], uniq[1]
        edge_events[(source, target, relation)].append(event.event_id)

    edges = [
        CompressedEdge(source=s, target=t, relation=r, event_ids=ids)
        for (s, t, r), ids in edge_events.items()
    ]
    edges.sort(key=lambda e: (-len(e.event_ids), e.source, e.target, e.relation))

    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge.source] += 1
        degree[edge.target] += 1

    dominant = [k for k, _ in relation_counter.most_common(3)]
    central = [k for k, _ in degree.most_common(5)]
    nodes = sorted(node_set)

    return CompressedGraph(
        nodes=nodes,
        edges=edges,
        event_count=len(events),
        date_range=(min(dates), max(dates)),
        dominant_relation_types=dominant,
        most_central_actors=central,
    )


__all__ = [
    "EVENT_TYPE_TO_RELATION",
    "CompressedEdge",
    "CompressedGraph",
    "EventRecord",
    "compress_graph",
]
