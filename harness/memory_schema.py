"""Three-layer memory schema contracts for the agentic harness.

This module defines immutable record types only. It intentionally contains no
storage/backend logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCallRecord:
    """Audit record for one tool call used inside an episode.

    Attributes document ownership so the loop and resolver can coordinate safely.
    """

    tool_name: str
    """Tool identifier written by the agent loop and read by audit/planning tools."""

    query: str
    """Query/request summary written by the agent loop and read during retrospectives."""

    as_of_time: str | None = None
    """PIT as-of marker written by retrieval tools and read by PIT auditors."""

    evidence_count: int = 0
    """Evidence volume written by tool wrappers and read by miss-analysis logic."""

    notes: str = ""
    """Free-form trace note written by loop execution and read by human reviewers."""

    def __post_init__(self) -> None:
        _require_non_empty(self.tool_name, "tool_name")
        _require_non_empty(self.query, "query")
        if not isinstance(self.evidence_count, int) or self.evidence_count < 0:
            raise ValueError("evidence_count must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "query": self.query,
            "as_of_time": self.as_of_time,
            "evidence_count": self.evidence_count,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolCallRecord:
        return cls(
            tool_name=payload["tool_name"],
            query=payload["query"],
            as_of_time=payload.get("as_of_time"),
            evidence_count=payload.get("evidence_count", 0),
            notes=payload.get("notes", ""),
        )


@dataclass(frozen=True)
class EpisodicRecord:
    """Per-job episode memory written at loop exit and updated at resolution."""

    job_id: str
    """Unique run id written by the agent loop and read by resolution listeners."""

    market_id: str
    """Market identifier written by loop setup and read by family/coverage retrieval."""

    market_family: str
    """Market family written by loop setup and read by memory-store family filters."""

    question: str
    """Market question text written at loop start and read during planning recall."""

    resolution_date: date
    """Resolution date written by market metadata ingest and read by PIT validators."""

    cutoff_date: date
    """Cutoff date written by loop setup and read by PIT-safe query generation."""

    blind_spot_checks_fired: list[str]
    """Checks executed by the loop and read by post-hoc miss attribution."""

    blind_spot_checks_skipped: list[str]
    """Checks not executed, written by loop planner and read by miss inference."""

    tool_calls: list[ToolCallRecord]
    """Tool trace written during research loop and read by auditing/diagnostics."""

    subgraph_node_count: int
    """Retrieved node count written by graph tooling and read by quality analysis."""

    gnn_score_trajectory: list[float]
    """Per-step scores written by gnn_score tool and read by convergence analysis."""

    final_p_yes: float
    """Final probability written by synthesis and read by Brier resolver."""

    confidence_interval: tuple[float, float] | None
    """Synthesis interval written by loop and read by calibration/resolution analysis."""

    brier_score: float | None
    """Resolution error written by resolver and read by policy-self-improvement."""

    misses: list[str]
    """Missed checks written post-resolution and read by policy patch generation."""

    notes: str
    """Free-form execution notes written by loop and read by human reviewers."""

    def __post_init__(self) -> None:
        _require_non_empty(self.job_id, "job_id")
        _require_non_empty(self.market_id, "market_id")
        _require_non_empty(self.market_family, "market_family")
        _require_non_empty(self.question, "question")

        if self.cutoff_date > self.resolution_date:
            raise ValueError("cutoff_date must be <= resolution_date")

        _require_list_of_str(self.blind_spot_checks_fired, "blind_spot_checks_fired")
        _require_list_of_str(self.blind_spot_checks_skipped, "blind_spot_checks_skipped")
        if set(self.blind_spot_checks_fired) & set(self.blind_spot_checks_skipped):
            raise ValueError("blind_spot_checks_fired and blind_spot_checks_skipped must be disjoint")

        if not isinstance(self.tool_calls, list) or not all(isinstance(t, ToolCallRecord) for t in self.tool_calls):
            raise ValueError("tool_calls must be a list[ToolCallRecord]")

        if not isinstance(self.subgraph_node_count, int) or self.subgraph_node_count < 0:
            raise ValueError("subgraph_node_count must be a non-negative int")

        if not isinstance(self.gnn_score_trajectory, list):
            raise ValueError("gnn_score_trajectory must be a list[float]")
        for idx, score in enumerate(self.gnn_score_trajectory):
            _require_float01(score, f"gnn_score_trajectory[{idx}]")

        _require_float01(self.final_p_yes, "final_p_yes")
        if self.confidence_interval is not None:
            if not isinstance(self.confidence_interval, tuple) or len(self.confidence_interval) != 2:
                raise ValueError("confidence_interval must be tuple[float, float] | None")
            lower, upper = self.confidence_interval
            _require_float01(lower, "confidence_interval[0]")
            _require_float01(upper, "confidence_interval[1]")
            if lower > upper:
                raise ValueError("confidence_interval lower bound must be <= upper bound")
        if self.brier_score is not None:
            _require_float01(self.brier_score, "brier_score")

        _require_list_of_str(self.misses, "misses")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "market_id": self.market_id,
            "market_family": self.market_family,
            "question": self.question,
            "resolution_date": self.resolution_date.isoformat(),
            "cutoff_date": self.cutoff_date.isoformat(),
            "blind_spot_checks_fired": list(self.blind_spot_checks_fired),
            "blind_spot_checks_skipped": list(self.blind_spot_checks_skipped),
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "subgraph_node_count": self.subgraph_node_count,
            "gnn_score_trajectory": list(self.gnn_score_trajectory),
            "final_p_yes": self.final_p_yes,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval is not None else None,
            "brier_score": self.brier_score,
            "misses": list(self.misses),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodicRecord:
        return cls(
            job_id=payload["job_id"],
            market_id=payload["market_id"],
            market_family=payload["market_family"],
            question=payload["question"],
            resolution_date=date.fromisoformat(payload["resolution_date"]),
            cutoff_date=date.fromisoformat(payload["cutoff_date"]),
            blind_spot_checks_fired=list(payload["blind_spot_checks_fired"]),
            blind_spot_checks_skipped=list(payload["blind_spot_checks_skipped"]),
            tool_calls=[
                t if isinstance(t, ToolCallRecord) else ToolCallRecord.from_dict(t)
                for t in payload["tool_calls"]
            ],
            subgraph_node_count=payload["subgraph_node_count"],
            gnn_score_trajectory=list(payload["gnn_score_trajectory"]),
            final_p_yes=payload["final_p_yes"],
            confidence_interval=(
                tuple(payload["confidence_interval"])
                if payload.get("confidence_interval") is not None
                else None
            ),
            brier_score=payload.get("brier_score"),
            misses=list(payload["misses"]),
            notes=payload["notes"],
        )


@dataclass(frozen=True)
class ConceptualPattern:
    """Named analytical pattern reinforced across episodes."""

    pattern_id: str
    """Pattern identifier written by policy/authoring flows and read by loop planner."""

    name: str
    """Human-readable label written by authoring flows and read in planning context."""

    description: str
    """Pattern description written by authors/patchers and read by planning/synthesis."""

    applicable_market_families: list[str]
    """Family scope written by pattern creators and read for retrieval filtering."""

    evidence_job_ids: list[str]
    """Supporting episodes written by policy patching and read by auditors/reviewers."""

    confidence: float
    """Pattern confidence written by curators/patch gate and read by planner ranking."""

    blind_spot_check_mapping: str | None
    """Optional check mapping written by authors and read by blind-spot planning."""

    created_at: datetime
    """Creation timestamp written on insert and read for chronology/decay logic."""

    last_reinforced_at: datetime | None
    """Last reinforcement timestamp written by updater and read by freshness filters."""

    source: Literal["hand_authored", "agent_proposed", "policy_patch"]
    """Provenance written by origin workflow and read by policy self-improvement gates."""

    def __post_init__(self) -> None:
        _require_non_empty(self.pattern_id, "pattern_id")
        _require_non_empty(self.name, "name")
        _require_non_empty(self.description, "description")
        _require_list_of_str(self.applicable_market_families, "applicable_market_families")
        if not self.applicable_market_families:
            raise ValueError("applicable_market_families must be non-empty")
        _require_list_of_str(self.evidence_job_ids, "evidence_job_ids")
        _require_float01(self.confidence, "confidence")
        if self.blind_spot_check_mapping is not None and not isinstance(self.blind_spot_check_mapping, str):
            raise ValueError("blind_spot_check_mapping must be str | None")
        if self.last_reinforced_at is not None and self.last_reinforced_at < self.created_at:
            raise ValueError("last_reinforced_at must be >= created_at")
        if self.source not in {"hand_authored", "agent_proposed", "policy_patch"}:
            raise ValueError("source must be one of: hand_authored, agent_proposed, policy_patch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "applicable_market_families": list(self.applicable_market_families),
            "evidence_job_ids": list(self.evidence_job_ids),
            "confidence": self.confidence,
            "blind_spot_check_mapping": self.blind_spot_check_mapping,
            "created_at": self.created_at.isoformat(),
            "last_reinforced_at": self.last_reinforced_at.isoformat() if self.last_reinforced_at else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConceptualPattern:
        return cls(
            pattern_id=payload["pattern_id"],
            name=payload["name"],
            description=payload["description"],
            applicable_market_families=list(payload["applicable_market_families"]),
            evidence_job_ids=list(payload["evidence_job_ids"]),
            confidence=payload["confidence"],
            blind_spot_check_mapping=payload.get("blind_spot_check_mapping"),
            created_at=datetime.fromisoformat(payload["created_at"]),
            last_reinforced_at=(
                datetime.fromisoformat(payload["last_reinforced_at"])
                if payload.get("last_reinforced_at") is not None
                else None
            ),
            source=payload["source"],
        )


@dataclass(frozen=True)
class StructuralFact:
    """Time-scoped structural fact cache for graph/planning reuse."""

    fact_id: str
    """Fact identifier written by fact producers and read by graph/planning lookups."""

    subject: str
    """Triple subject written by fact producers and read by query/reasoning tools."""

    predicate: str
    """Triple predicate written by fact producers and read by query/reasoning tools."""

    object: str
    """Triple object written by fact producers and read by query/reasoning tools."""

    confidence: float
    """Fact confidence written by extractors/verifiers and read by filtering logic."""

    source_url: str | None
    """Source pointer written by extractors and read by provenance auditors."""

    valid_from: date | None
    """Validity start written by extractors and read by temporal filtering."""

    valid_until: date | None
    """Validity end written by extractors and read by temporal filtering."""

    last_verified: date
    """Last verification date written by verifiers and read by freshness checks."""

    def __post_init__(self) -> None:
        _require_non_empty(self.fact_id, "fact_id")
        _require_non_empty(self.subject, "subject")
        _require_non_empty(self.predicate, "predicate")
        _require_non_empty(self.object, "object")
        _require_float01(self.confidence, "confidence")
        if self.source_url is not None and not isinstance(self.source_url, str):
            raise ValueError("source_url must be str | None")
        if self.valid_from is not None and self.valid_until is not None and self.valid_from > self.valid_until:
            raise ValueError("valid_from must be <= valid_until")
        if self.valid_from is not None and self.last_verified < self.valid_from:
            raise ValueError("last_verified must be >= valid_from when valid_from is provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "last_verified": self.last_verified.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StructuralFact:
        return cls(
            fact_id=payload["fact_id"],
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            confidence=payload["confidence"],
            source_url=payload.get("source_url"),
            valid_from=date.fromisoformat(payload["valid_from"]) if payload.get("valid_from") else None,
            valid_until=date.fromisoformat(payload["valid_until"]) if payload.get("valid_until") else None,
            last_verified=date.fromisoformat(payload["last_verified"]),
        )


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_list_of_str(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{field_name} must be a list[str]")


def _require_float01(value: Any, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite float in [0, 1]")
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{field_name} must be in [0, 1]")


__all__ = [
    "ConceptualPattern",
    "EpisodicRecord",
    "StructuralFact",
    "ToolCallRecord",
]
