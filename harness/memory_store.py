"""Memory-store protocol and local JSONL backends for the agentic harness."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from harness.memory_schema import ConceptualPattern, EpisodicRecord, StructuralFact

T = TypeVar("T")


@runtime_checkable
class MemoryStore(Protocol):
    def write_episode(self, record: EpisodicRecord) -> None: ...

    def update_episode_brier(self, job_id: str, brier_score: float, misses: list[str]) -> None: ...

    def read_recent_episodes(self, market_family: str, n: int) -> list[EpisodicRecord]: ...

    def write_pattern(self, pattern: ConceptualPattern) -> None: ...

    def read_patterns(self, market_family: str) -> list[ConceptualPattern]: ...

    def write_fact(self, fact: StructuralFact) -> None: ...

    def read_facts(self, subject: str) -> list[StructuralFact]: ...


class NullMemoryStore:
    def write_episode(self, record: EpisodicRecord) -> None:
        _ = record

    def update_episode_brier(self, job_id: str, brier_score: float, misses: list[str]) -> None:
        _ = (job_id, brier_score, misses)

    def read_recent_episodes(self, market_family: str, n: int) -> list[EpisodicRecord]:
        _ = (market_family, n)
        return []

    def write_pattern(self, pattern: ConceptualPattern) -> None:
        _ = pattern

    def read_patterns(self, market_family: str) -> list[ConceptualPattern]:
        _ = market_family
        return []

    def write_fact(self, fact: StructuralFact) -> None:
        _ = fact

    def read_facts(self, subject: str) -> list[StructuralFact]:
        _ = subject
        return []


class JsonlMemoryStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = self._base_dir / "episodes.jsonl"
        self._patterns_path = self._base_dir / "patterns.jsonl"
        self._facts_path = self._base_dir / "facts.jsonl"

    def write_episode(self, record: EpisodicRecord) -> None:
        self._append_jsonl(self._episodes_path, record.to_dict())

    def update_episode_brier(self, job_id: str, brier_score: float, misses: list[str]) -> None:
        episodes = self._read_jsonl(self._episodes_path, EpisodicRecord.from_dict)
        found = False
        updated: list[EpisodicRecord] = []

        for episode in episodes:
            if episode.job_id == job_id:
                payload = episode.to_dict()
                payload["brier_score"] = brier_score
                payload["misses"] = list(misses)
                updated.append(EpisodicRecord.from_dict(payload))
                found = True
            else:
                updated.append(episode)

        if not found:
            raise KeyError(f"episode with job_id '{job_id}' not found")

        # TODO(memory-store): consider append-only JSONL patch records for brier updates
        # to avoid full-file rewrites once write volume grows.
        self._rewrite_jsonl(self._episodes_path, [episode.to_dict() for episode in updated])

    def read_recent_episodes(self, market_family: str, n: int) -> list[EpisodicRecord]:
        episodes = self._read_jsonl(self._episodes_path, EpisodicRecord.from_dict)
        filtered = [episode for episode in episodes if episode.market_family == market_family]
        filtered.sort(key=lambda episode: episode.resolution_date, reverse=True)
        return filtered[:n]

    def write_pattern(self, pattern: ConceptualPattern) -> None:
        self._append_jsonl(self._patterns_path, pattern.to_dict())

    def read_patterns(self, market_family: str) -> list[ConceptualPattern]:
        patterns = self._read_jsonl(self._patterns_path, ConceptualPattern.from_dict)
        return [
            pattern
            for pattern in patterns
            if market_family in pattern.applicable_market_families
        ]

    def write_fact(self, fact: StructuralFact) -> None:
        self._append_jsonl(self._facts_path, fact.to_dict())

    def read_facts(self, subject: str) -> list[StructuralFact]:
        facts = self._read_jsonl(self._facts_path, StructuralFact.from_dict)
        return [fact for fact in facts if fact.subject == subject]

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")

    @staticmethod
    def _rewrite_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload))
                handle.write("\n")

    @staticmethod
    def _read_jsonl(path: Path, loader: Callable[[dict[str, object]], T]) -> list[T]:
        if not path.exists():
            return []

        out: list[T] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                out.append(loader(json.loads(stripped)))
        return out


__all__ = ["MemoryStore", "NullMemoryStore", "JsonlMemoryStore"]
