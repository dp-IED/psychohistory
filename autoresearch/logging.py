"""JSONL experiment logging with failure classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class FailureClass(str, Enum):
    syntax_failure = "syntax_failure"
    schema_invalid = "schema_invalid"
    eval_crash = "eval_crash"
    score_regression = "score_regression"
    incompatible_projection = "incompatible_projection"
    retrieval_regression = "retrieval_regression"


@dataclass
class ExperimentRecord:
    iteration: int
    candidate_id: str
    timestamp_utc: str
    composite_score: float
    score_parts: dict[str, float] = field(default_factory=dict)
    proposal: dict[str, Any] = field(default_factory=dict)
    eval_report_path: str | None = None
    failure_class: str | None = None
    notes: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def append_experiment(log_path: Path, record: ExperimentRecord) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not record.timestamp_utc:
        record.timestamp_utc = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(record.to_json_line() + "\n")


def read_prior_scores(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    scores: list[float] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                scores.append(float(obj.get("composite_score", 0.0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return scores
