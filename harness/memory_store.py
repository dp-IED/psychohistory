from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from harness.memory_schema import ConceptualPattern, EpisodicRecord, StructuralFact


class MemoryStore(ABC):
    """Abstract memory store for episodic, pattern, and factual data."""

    @abstractmethod
    def write_episode(self, episode: EpisodicRecord) -> None: ...

    @abstractmethod
    def update_episode_brier(
        self, job_id: str, brier_score: float, misses: list[str]
    ) -> None: ...

    @abstractmethod
    def write_pattern(self, pattern: ConceptualPattern) -> None: ...

    @abstractmethod
    def write_fact(self, fact: StructuralFact) -> None: ...

    @abstractmethod
    def read_recent_episodes(
        self, market_family: str, limit: int
    ) -> list[EpisodicRecord]: ...

    @abstractmethod
    def read_episode_by_id(self, job_id: str) -> EpisodicRecord | None: ...

    @abstractmethod
    def read_patterns(
        self, market_family: str
    ) -> list[ConceptualPattern]: ...

    @abstractmethod
    def read_facts(self, subject: str) -> list[StructuralFact]: ...


class NullMemoryStore(MemoryStore):
    """No-op store: all writes succeed silently, all reads return empty."""

    def write_episode(self, episode: EpisodicRecord) -> None:
        pass

    def update_episode_brier(
        self, job_id: str, brier_score: float, misses: list[str]
    ) -> None:
        pass

    def write_pattern(self, pattern: ConceptualPattern) -> None:
        pass

    def write_fact(self, fact: StructuralFact) -> None:
        pass

    def read_recent_episodes(
        self, market_family: str, limit: int
    ) -> list[EpisodicRecord]:
        return []

    def read_episode_by_id(self, job_id: str) -> EpisodicRecord | None:
        return None

    def read_patterns(self, market_family: str) -> list[ConceptualPattern]:
        return []

    def read_facts(self, subject: str) -> list[StructuralFact]:
        return []


class JsonlMemoryStore(MemoryStore):
    """File-backed memory store using JSONL files."""

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    @property
    def _episodes_path(self) -> Path:
        return self._base / "episodes.jsonl"

    @property
    def _patterns_path(self) -> Path:
        return self._base / "patterns.jsonl"

    @property
    def _facts_path(self) -> Path:
        return self._base / "facts.jsonl"

    # ── write ────────────────────────────────────────────────────────

    def write_episode(self, episode: EpisodicRecord) -> None:
        with self._episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode.to_dict(), default=str) + "\n")

    def update_episode_brier(
        self, job_id: str, brier_score: float, misses: list[str]
    ) -> None:
        episodes = self._read_all_episodes()
        if job_id not in episodes:
            raise KeyError(job_id)
        # Rewrite: update the matching episode in place
        updated = []
        for ep in episodes.values():
            if ep.job_id == job_id:
                ep = EpisodicRecord(
                    job_id=ep.job_id,
                    market_id=ep.market_id,
                    market_family=ep.market_family,
                    question=ep.question,
                    resolution_date=ep.resolution_date,
                    cutoff_date=ep.cutoff_date,
                    blind_spot_checks_fired=ep.blind_spot_checks_fired,
                    blind_spot_checks_skipped=ep.blind_spot_checks_skipped,
                    tool_calls=ep.tool_calls,
                    subgraph_node_count=ep.subgraph_node_count,
                    gnn_score_trajectory=ep.gnn_score_trajectory,
                    final_p_yes=ep.final_p_yes,
                    confidence_interval=ep.confidence_interval,
                    brier_score=brier_score,
                    misses=misses,
                    notes=ep.notes,
                )
            updated.append(ep)
        self._write_all_episodes(updated)

    def write_pattern(self, pattern: ConceptualPattern) -> None:
        with self._patterns_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pattern.to_dict(), default=str) + "\n")

    def write_fact(self, fact: StructuralFact) -> None:
        with self._facts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fact.to_dict(), default=str) + "\n")

    # ── read ─────────────────────────────────────────────────────────

    def read_recent_episodes(
        self, market_family: str, limit: int
    ) -> list[EpisodicRecord]:
        episodes = self._read_all_episodes()
        matching = [
            ep
            for ep in episodes.values()
            if ep.market_family == market_family
        ]
        matching.sort(key=lambda ep: ep.resolution_date, reverse=True)
        return matching[:limit]

    def read_episode_by_id(self, job_id: str) -> EpisodicRecord | None:
        return self._read_all_episodes().get(job_id)

    def read_patterns(self, market_family: str) -> list[ConceptualPattern]:
        if not self._patterns_path.exists():
            return []
        result = []
        for line in _safe_lines(self._patterns_path):
            pattern = ConceptualPattern.from_dict(json.loads(line))
            if market_family in pattern.applicable_market_families:
                result.append(pattern)
        return result

    def read_facts(self, subject: str) -> list[StructuralFact]:
        if not self._facts_path.exists():
            return []
        result = []
        for line in _safe_lines(self._facts_path):
            fact = StructuralFact.from_dict(json.loads(line))
            if fact.subject == subject:
                result.append(fact)
        return result

    # ── internal ─────────────────────────────────────────────────────

    def _read_all_episodes(self) -> dict[str, EpisodicRecord]:
        if not self._episodes_path.exists():
            return {}
        result: dict[str, EpisodicRecord] = {}
        for line in _safe_lines(self._episodes_path):
            ep = EpisodicRecord.from_dict(json.loads(line))
            result[ep.job_id] = ep
        return result

    def _write_all_episodes(self, episodes: list[EpisodicRecord]) -> None:
        with self._episodes_path.open("w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep.to_dict(), default=str) + "\n")


def _safe_lines(path: Path) -> list[str]:
    """Read lines, raise JSONDecodeError on any corrupt line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # validate parseability
        json.loads(line)
    return lines
