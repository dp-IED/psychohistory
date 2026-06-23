"""Weighted mechanism graph for calibration pooling.

Instead of binary related/not-related via wikilinks, edges have
weights derived from:

1. Co-occurrence: how often the agent invokes both mechanisms together
2. Domain: same-domain edges get a boost
3. Tag overlap: how many Polymarket tags the mechanisms share
4. Directness: 1-hop gets full weight, 2-hop gets decayed

Weighted pooling means a mechanism with N=1 but strong edges to
a mechanism with N=10 gets substantial pooling. A mechanism with
N=1 and weak edges to everything stays near the prior.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WeightedEdge:
    target: str
    weight: float
    source: str = ""  # "wikilink", "co-occurrence", "domain", "tag_overlap"


class MechanismGraph:
    """Weighted graph connecting mechanisms via multiple edge types."""

    def __init__(self):
        # {source: {target: total_weight}}
        self._edges: dict[str, dict[str, float]] = defaultdict(dict)
        # {source: {target: [WeightedEdge]}}
        self._edge_sources: dict[str, dict[str, list[WeightedEdge]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._meta: dict[str, dict] = {}  # name → {domain, tags, filepath}

    def add_wikilink_edge(self, source: str, target: str) -> None:
        """Direct wikilink from concept A to concept B. Weight: 0.5."""
        self._add_edge(source, target, 0.5, "wikilink")

    def add_cooccurrence_edge(self, source: str, target: str, count: int) -> None:
        """Both mechanisms invoked together in count forecasts. Weight: log(count+1)/10."""
        w = min(0.8, math.log(count + 1) / 6)
        self._add_edge(source, target, w, "co-occurrence")

    def add_domain_edge(self, source: str, target: str) -> None:
        """Same domain boost. Weight: 0.3."""
        self._add_edge(source, target, 0.3, "domain")

    def add_tag_overlap_edge(self, source: str, target: str, overlap: int) -> None:
        """Share overlap Polymarket tags. Weight: min(0.4, overlap/10)."""
        w = min(0.4, overlap / 10)
        self._add_edge(source, target, w, "tag_overlap")

    def _add_edge(self, source: str, target: str, weight: float,
                  edge_type: str) -> None:
        if source == target:
            return
        # Bidirectional
        for s, t in [(source, target), (target, source)]:
            self._edges[s][t] = self._edges[s].get(t, 0.0) + weight
            self._edge_sources[s][t].append(
                WeightedEdge(target=t, weight=weight, source=edge_type)
            )

    def get_neighbors(self, name: str, min_weight: float = 0.2) -> dict[str, float]:
        """Return {neighbor: weight} for edges above threshold."""
        raw = self._edges.get(name, {})
        return {k: v for k, v in raw.items() if v >= min_weight}

    def set_metadata(self, name: str, domain: str, tags: list[str]) -> None:
        self._meta[name] = {"domain": domain, "tags": tags}

    def get_metadata(self, name: str) -> dict:
        return self._meta.get(name, {})

    def to_dict(self) -> dict:
        return {
            "edges": {
                s: {t: round(w, 3) for t, w in targets.items()}
                for s, targets in self._edges.items()
            },
            "n_nodes": len(self._meta),
            "n_edges": sum(len(v) for v in self._edges.values()),
        }

    # ── build from vault ──────────────────────────────────────────────

    @classmethod
    def from_vault(cls, vault_dir: str | Path,
                   trails: list | None = None) -> "MechanismGraph":
        """Build graph from vault concept files + agent trails."""
        graph = cls()
        vault = Path(vault_dir)

        # Phase 1: wikilinks from concept files
        for md_file in vault.rglob("_concept.md"):
            cls._process_concept_file(graph, md_file)
        for md_file in vault.rglob("*.md"):
            if md_file.name.startswith("_concept") or md_file.name == "_index.md":
                continue
            text = md_file.read_text(encoding="utf-8")
            if "type: concept" in text[:200]:
                cls._process_concept_file(graph, md_file)

        # Phase 2: domain edges
        domains: dict[str, list[str]] = defaultdict(list)
        for name, meta in graph._meta.items():
            d = meta.get("domain", "")
            if isinstance(d, list):
                d = d[0] if d else ""
            if d and isinstance(d, str):
                domains[d].append(name)
        for names in domains.values():
            for i, a in enumerate(names):
                for b in names[i+1:]:
                    graph.add_domain_edge(a, b)

        # Phase 3: co-occurrence from agent trails
        if trails:
            cooccur = defaultdict(int)
            for trail in trails:
                mechs = [
                    trail._normalize_mechanism(m) if hasattr(trail, '_normalize_mechanism')
                    else _normalize_mechanism(m)
                    for m in trail.mechanisms
                ]
                for i, a in enumerate(mechs):
                    for b in mechs[i+1:]:
                        pair = tuple(sorted([a, b]))
                        cooccur[pair] += 1
            for (a, b), count in cooccur.items():
                graph.add_cooccurrence_edge(a, b, count)

        return graph

    @staticmethod
    def _process_concept_file(graph: "MechanismGraph", filepath: Path) -> None:
        text = filepath.read_text(encoding="utf-8")
        fm = _parse_yaml_frontmatter(text)
        if not fm:
            return

        name = fm.get("slug", "") or fm.get("title", "")
        if not name:
            name = filepath.parent.name if filepath.name == "_concept.md" else filepath.stem
        name = name.lower().replace(" ", "-")[:60]

        graph.set_metadata(
            name,
            domain=fm.get("domain", ""),
            tags=fm.get("tags", []),
        )

        # Add wikilink edges
        related = fm.get("related_concepts") or []
        if isinstance(related, str):
            related = [related]
        for r in related:
            m = re.search(r'\[\[(?:.*?/)?([^/\]]+?)(?:/_concept)?\]\]', str(r))
            target = m.group(1).lower().replace(" ", "-") if m else str(r).lower().replace(" ", "-")[:60]
            graph.add_wikilink_edge(name, target)


def _normalize_mechanism(raw: str) -> str:
    name = raw.split(':')[0].strip()
    name = name.split('(')[0].strip()
    return name.lower().replace(' ', '-')[:60]


def _parse_yaml_frontmatter(text: str) -> dict | None:
    match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None
