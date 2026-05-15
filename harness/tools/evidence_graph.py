"""Lightweight evidence graph built from web_search ToolCallRecords."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date

from harness.memory_schema import ToolCallRecord

_WS_SPLIT = re.compile(r"\s+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


@dataclass
class EvidenceNode:
    node_id: str
    label: str
    node_type: str
    relevance: float
    timestamp: str


@dataclass
class EvidenceGraph:
    nodes: list[EvidenceNode]
    edges: list[tuple[str, str, float]]
    summary: str
    articles: list[tuple[str, str, str]] = field(default_factory=list)
    """(url, title, snippet) for downstream LLM context."""


def _stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha256(f"{prefix}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"


def _tokens_for_overlap(text: str) -> set[str]:
    lower = text.lower()
    parts = {p for p in _WS_SPLIT.split(lower) if len(p) >= 4}
    return parts


def _extract_entities(text: str) -> set[str]:
    found = {m.group(0) for m in _ENTITY_RE.finditer(text)}
    return {e for e in found if len(e) >= 3}


def _parse_article_chunks(notes: str) -> list[tuple[str, str, str, str, str]]:
    """Return list of (iso_date, title, source, url, summary_snippet)."""
    out: list[tuple[str, str, str, str, str]] = []
    for chunk in notes.split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        if "|" in piece:
            parts = piece.split("|", 4)
            if len(parts) >= 4:
                ds = parts[0].strip()
                title = parts[1].strip() if len(parts) > 1 else ""
                source = parts[2].strip() if len(parts) > 2 else ""
                url = parts[3].strip() if len(parts) > 3 else ""
                summary = parts[4].strip() if len(parts) > 4 else ""
                out.append((ds, title, source, url, summary))
            continue
        tokens = piece.split()
        if len(tokens) >= 3:
            ds = tokens[0]
            url = next((t for t in tokens if t.startswith("http")), "")
            source = tokens[1] if len(tokens) > 1 else ""
            title = " ".join(t for t in tokens[1:] if t != url and t != source)
            out.append((ds, title, source, url, ""))
    return out


def build_evidence_graph(
    tool_calls: list[ToolCallRecord],
    question: str,
    cutoff_date: date,
) -> EvidenceGraph:
    """Extract entities, articles, and co-reference edges from search evidence."""

    web_calls = [t for t in tool_calls if t.tool_name == "web_search"]
    question_entities = _extract_entities(question)
    question_tokens = _tokens_for_overlap(question)

    nodes: list[EvidenceNode] = []
    edges: list[tuple[str, str, float]] = []
    articles: list[tuple[str, str, str]] = []
    article_ids: list[str] = []
    entity_to_articles: dict[str, set[str]] = {}

    for call in web_calls:
        chunks = _parse_article_chunks(call.notes)
        if not chunks and call.query:
            nid = _stable_id("evidence", call.query)
            nodes.append(
                EvidenceNode(
                    node_id=nid,
                    label=call.query[:280],
                    node_type="claim",
                    relevance=0.35,
                    timestamp=cutoff_date.isoformat(),
                )
            )
            article_ids.append(nid)
            articles.append(("", call.query[:200], ""))
            continue

        for ds, title, _source, url, summary in chunks:
            label = title or url or call.query[:120]
            nid = _stable_id("article", url or f"{ds}:{label}")
            rel = 0.4
            overlap = len(question_tokens & _tokens_for_overlap(f"{title} {summary}"))
            if question_tokens:
                rel = min(0.95, 0.35 + 0.08 * overlap)
            nodes.append(
                EvidenceNode(
                    node_id=nid,
                    label=label[:500],
                    node_type="article",
                    relevance=rel,
                    timestamp=ds[:10] if len(ds) >= 10 else cutoff_date.isoformat(),
                )
            )
            article_ids.append(nid)
            articles.append((url, title, summary))

            body_entities = _extract_entities(f"{title} {summary}")
            body_entities |= question_entities
            for ent in body_entities:
                eid = _stable_id("entity", ent)
                if eid not in {n.node_id for n in nodes}:
                    nodes.append(
                        EvidenceNode(
                            node_id=eid,
                            label=ent,
                            node_type="entity",
                            relevance=0.55,
                            timestamp=cutoff_date.isoformat(),
                        )
                    )
                edges.append((nid, eid, 0.9))
                entity_to_articles.setdefault(eid, set()).add(nid)

    for eid, aids in entity_to_articles.items():
        aid_list = sorted(aids)
        for i in range(len(aid_list)):
            for j in range(i + 1, len(aid_list)):
                edges.append((aid_list[i], aid_list[j], 0.7))

    if not nodes:
        qid = _stable_id("claim", question)
        nodes.append(
            EvidenceNode(
                node_id=qid,
                label=question[:280],
                node_type="claim",
                relevance=0.25,
                timestamp=cutoff_date.isoformat(),
            )
        )

    art_count = sum(1 for n in nodes if n.node_type == "article")
    ent_count = sum(1 for n in nodes if n.node_type == "entity")
    summary = (
        f"As of {cutoff_date.isoformat()}, gathered {art_count} article node(s), {ent_count} entity node(s), "
        f"and {len(edges)} edge(s) from {len(web_calls)} search call(s)."
    )

    return EvidenceGraph(nodes=nodes, edges=edges, summary=summary, articles=articles)
