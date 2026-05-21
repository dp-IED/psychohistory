from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from harness.memory_store import JsonlMemoryStore


class SyncFormatError(ValueError):
    """Raised when template output files have missing/bad fields."""


@dataclass(frozen=True)
class SyncResult:
    scanned: int
    imported: int
    skipped_existing: int
    resolved: int = 0
    resolve_skipped: int = 0


def sync_template_outputs(
    template_dir: Path,
    memory: JsonlMemoryStore,
) -> SyncResult:
    """Scan template output JSONL files and import new runs into memory.

    Skips runs already present (by job_id derived from question_id + timestamp).
    Resolves runs when outcome is present and not yet resolved.
    """
    scanned = 0
    imported = 0
    skipped_existing = 0
    resolved = 0
    resolve_skipped = 0

    # Track existing job_ids to skip duplicates
    for jsonl_path in sorted(template_dir.glob("*.jsonl")):
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scanned += 1

            # Validation: require run_timestamp
            timestamp = row.get("run_timestamp")
            if timestamp is None:
                raise SyncFormatError(
                    f"Missing 'run_timestamp' in {jsonl_path.name}"
                )

            qid = row["question_id"]
            job_id = f"q{qid}-{timestamp}"

            # Check if already imported
            existing = memory.read_episode_by_id(job_id)
            if existing is not None:
                # Already imported — check if resolution needed
                outcome = row.get("resolved_outcome")
                if outcome is not None and existing.brier_score is None:
                    # Resolve it
                    from harness.memory_schema import EpisodicRecord

                    memory.update_episode_brier(
                        job_id=job_id,
                        brier_score=_brier(existing.final_p_yes, outcome),
                        misses=[],
                    )
                    resolved += 1
                else:
                    resolve_skipped += 1
                skipped_existing += 1
                continue

            # Import new episode
            from harness.memory_schema import EpisodicRecord

            close_date = _parse_date(row.get("close_date"))
            resolution_date = _parse_date(row.get("resolution_date"))
            p_yes = float(row.get("posted_probability", 0.5))

            episode = EpisodicRecord(
                job_id=job_id,
                market_id=str(qid),
                market_family="metaculus_binary",
                question=row.get("question_text", ""),
                resolution_date=resolution_date,
                cutoff_date=close_date,
                final_p_yes=p_yes,
                confidence_interval=(max(0.0, p_yes - 0.1), min(1.0, p_yes + 0.1)),
                notes=f"Synced from {jsonl_path.name}",
            )
            memory.write_episode(episode)
            imported += 1

            # Check for resolution
            outcome = row.get("resolved_outcome")
            if outcome is not None:
                memory.update_episode_brier(
                    job_id=job_id,
                    brier_score=_brier(p_yes, outcome),
                    misses=[],
                )
                resolved += 1

    return SyncResult(
        scanned=scanned,
        imported=imported,
        skipped_existing=skipped_existing,
        resolved=resolved,
        resolve_skipped=resolve_skipped,
    )


def _brier(p_yes: float, outcome: object) -> float:
    target = 1.0 if outcome else 0.0
    return (p_yes - target) ** 2


def _parse_date(raw: object) -> date:
    if raw is None:
        return date.today()
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw)[:10])
