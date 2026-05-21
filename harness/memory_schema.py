from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

# ── Tool call record ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    query: str
    as_of_time: str  # ISO 8601
    evidence_count: int
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "query": self.query,
            "as_of_time": self.as_of_time,
            "evidence_count": self.evidence_count,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCallRecord":
        return cls(
            tool_name=d["tool_name"],
            query=d["query"],
            as_of_time=d["as_of_time"],
            evidence_count=d["evidence_count"],
            notes=d["notes"],
        )


# ── Episodic record ──────────────────────────────────────────────────

_VALID_SOURCES = frozenset(
    {"policy_patch", "hand_authored", "observation_distill", "reflection_update"}
)


@dataclass(frozen=True)
class EpisodicRecord:
    job_id: str
    market_id: str
    market_family: str
    question: str
    resolution_date: date
    cutoff_date: date
    blind_spot_checks_fired: list[str] = field(default_factory=list)
    blind_spot_checks_skipped: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    subgraph_node_count: int = 0
    gnn_score_trajectory: list[float] = field(default_factory=list)
    final_p_yes: float = 0.5
    confidence_interval: tuple[float, float] | None = None
    brier_score: float | None = None
    misses: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.cutoff_date >= self.resolution_date:
            raise ValueError(
                f"cutoff_date ({self.cutoff_date}) must be before "
                f"resolution_date ({self.resolution_date})"
            )
        fired = set(self.blind_spot_checks_fired)
        skipped = set(self.blind_spot_checks_skipped)
        both = fired & skipped
        if both:
            raise ValueError(
                f"checks must be disjoint, shared: {sorted(both)}"
            )
        if not 0.0 <= self.final_p_yes <= 1.0:
            raise ValueError(
                f"final_p_yes must be 0-1, got {self.final_p_yes}"
            )
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(
                    f"confidence_interval must satisfy 0 <= lo <= hi <= 1, "
                    f"got ({lo}, {hi})"
                )

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
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "subgraph_node_count": self.subgraph_node_count,
            "gnn_score_trajectory": list(self.gnn_score_trajectory),
            "final_p_yes": self.final_p_yes,
            "confidence_interval": (
                list(self.confidence_interval)
                if self.confidence_interval is not None
                else None
            ),
            "brier_score": self.brier_score,
            "misses": list(self.misses),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodicRecord":
        ci = d.get("confidence_interval")
        if ci is not None and isinstance(ci, list) and len(ci) == 2:
            ci = (ci[0], ci[1])
        elif ci is not None and isinstance(ci, (list, tuple)):
            ci = tuple(ci)  # type: ignore[assignment]
        raw_calls = d.get("tool_calls", [])
        if raw_calls and isinstance(raw_calls[0], dict):
            calls = [ToolCallRecord.from_dict(tc) for tc in raw_calls]
        else:
            calls = list(raw_calls)
        return cls(
            job_id=d["job_id"],
            market_id=d["market_id"],
            market_family=d["market_family"],
            question=d["question"],
            resolution_date=_parse_date(d["resolution_date"]),
            cutoff_date=_parse_date(d["cutoff_date"]),
            blind_spot_checks_fired=list(d.get("blind_spot_checks_fired", [])),
            blind_spot_checks_skipped=list(d.get("blind_spot_checks_skipped", [])),
            tool_calls=calls,
            subgraph_node_count=d.get("subgraph_node_count", 0),
            gnn_score_trajectory=list(d.get("gnn_score_trajectory", [])),
            final_p_yes=d.get("final_p_yes", 0.5),
            confidence_interval=ci,
            brier_score=d.get("brier_score"),
            misses=list(d.get("misses", [])),
            notes=d.get("notes", ""),
        )


# ── Conceptual pattern ───────────────────────────────────────────────


@dataclass(frozen=True)
class ConceptualPattern:
    pattern_id: str
    name: str
    description: str
    applicable_market_families: list[str] = field(default_factory=list)
    evidence_job_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    blind_spot_check_mapping: str | None = None
    created_at: datetime | None = None
    last_reinforced_at: datetime | None = None
    source: str = "hand_authored"

    _SOURCES: tuple[str, ...] = (
        "policy_patch",
        "hand_authored",
        "observation_distill",
        "reflection_update",
    )

    def __post_init__(self) -> None:
        if self.source not in self._SOURCES:
            raise ValueError(
                f"source must be one of {self._SOURCES}, got {self.source!r}"
            )
        if (
            self.last_reinforced_at is not None
            and self.created_at is not None
            and self.last_reinforced_at < self.created_at
        ):
            raise ValueError(
                "last_reinforced_at must be >= created_at"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "applicable_market_families": list(self.applicable_market_families),
            "evidence_job_ids": list(self.evidence_job_ids),
            "confidence": self.confidence,
            "blind_spot_check_mapping": self.blind_spot_check_mapping,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_reinforced_at": (
                self.last_reinforced_at.isoformat() if self.last_reinforced_at else None
            ),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConceptualPattern":
        return cls(
            pattern_id=d["pattern_id"],
            name=d["name"],
            description=d["description"],
            applicable_market_families=list(d.get("applicable_market_families", [])),
            evidence_job_ids=list(d.get("evidence_job_ids", [])),
            confidence=d.get("confidence", 0.5),
            blind_spot_check_mapping=d.get("blind_spot_check_mapping"),
            created_at=_parse_dt(d.get("created_at")),
            last_reinforced_at=_parse_dt(d.get("last_reinforced_at")),
            source=d.get("source", "hand_authored"),
        )


# ── Structural fact ──────────────────────────────────────────────────


@dataclass(frozen=True)
class StructuralFact:
    fact_id: str
    subject: str
    predicate: str
    object: str  # noqa: A003 (shadow builtin intentionally)
    confidence: float = 0.5
    source_url: str | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    last_verified: date | None = None

    def __post_init__(self) -> None:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_from > self.valid_until
        ):
            raise ValueError("valid_from must be <= valid_until")

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
            "last_verified": (
                self.last_verified.isoformat() if self.last_verified else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StructuralFact":
        return cls(
            fact_id=d["fact_id"],
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            confidence=d.get("confidence", 0.5),
            source_url=d.get("source_url"),
            valid_from=_parse_date(d.get("valid_from")),
            valid_until=_parse_date(d.get("valid_until")),
            last_verified=_parse_date(d.get("last_verified")),
        )


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_date(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw))


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    return datetime.fromisoformat(str(raw))
