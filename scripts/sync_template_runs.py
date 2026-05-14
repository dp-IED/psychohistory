from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from harness.agent_loop import AgentToolset, GraphQueryResult, MarketContext
from harness.memory_schema import EpisodicRecord, ToolCallRecord
from harness.memory_store import JsonlMemoryStore, MemoryStore
from harness.resolution import AlreadyResolvedError, resolve_market


class SyncFormatError(ValueError):
    pass


@dataclass(frozen=True)
class SyncResult:
    scanned: int
    imported: int
    skipped_existing: int
    resolved: int
    resolve_skipped: int


def _parse_date(value: str | None, field: str) -> date:
    if not value:
        raise SyncFormatError(f"missing required date field: {field}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise SyncFormatError(f"invalid date value for {field}: {value}") from exc


def _extract_probability(payload: dict[str, Any]) -> float:
    value = payload.get("posted_probability")
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
        return float(value)
    raise SyncFormatError("missing or invalid required field: posted_probability")


def _extract_question_id(payload: dict[str, Any]) -> int:
    value = payload.get("question_id")
    if isinstance(value, int) and value > 0:
        return value
    raise SyncFormatError("missing or invalid required field: question_id")


def _extract_question_text(payload: dict[str, Any]) -> str:
    value = payload.get("question_text")
    if isinstance(value, str) and value.strip():
        return value
    raise SyncFormatError("missing or invalid required field: question_text")


def _extract_run_timestamp(payload: dict[str, Any]) -> str:
    value = payload.get("run_timestamp")
    if isinstance(value, str) and value.strip():
        return value
    raise SyncFormatError("missing or invalid required field: run_timestamp")


def _derive_job_id(question_id: int, run_timestamp: str) -> str:
    _ = run_timestamp
    return f"template-{question_id}"


def _build_episode(payload: dict[str, Any]) -> EpisodicRecord:
    question_id = _extract_question_id(payload)
    run_ts = _extract_run_timestamp(payload)
    job_id = _derive_job_id(question_id, run_ts)

    cutoff = _parse_date(
        payload.get("cutoff_date") or payload.get("close_date") or payload.get("scheduled_close_time"),
        "cutoff_date|close_date|scheduled_close_time",
    )
    resolution_date = _parse_date(
        payload.get("resolution_date") or payload.get("resolve_time") or payload.get("scheduled_resolve_time") or payload.get("close_date"),
        "resolution_date|resolve_time|scheduled_resolve_time|close_date",
    )
    if cutoff > resolution_date:
        cutoff = resolution_date

    p_yes = _extract_probability(payload)
    qtext = _extract_question_text(payload)

    return EpisodicRecord(
        job_id=job_id,
        market_id=f"metaculus-{question_id}",
        market_family="metaculus_binary",
        question=qtext,
        resolution_date=resolution_date,
        cutoff_date=cutoff,
        blind_spot_checks_fired=[],
        blind_spot_checks_skipped=[],
        tool_calls=[
            ToolCallRecord(
                tool_name="template_sync",
                query=qtext,
                as_of_time=f"{cutoff.isoformat()}T00:00:00Z",
                evidence_count=0,
                notes=f"synced from template run timestamp={run_ts}",
            )
        ],
        subgraph_node_count=0,
        gnn_score_trajectory=[p_yes],
        final_p_yes=p_yes,
        confidence_interval=None,
        brier_score=None,
        misses=[],
        notes="Imported from metac-bot-template output",
    )


def _is_resolved_true(payload: dict[str, Any]) -> bool | None:
    for key in ("resolved_outcome", "outcome", "resolution"):
        value = payload.get(key)
        if isinstance(value, bool):
            return value
    return None


def _resolution_toolset() -> AgentToolset:
    def web_search(_req):
        return []

    def graph_query(_q: str, _d: date) -> GraphQueryResult:
        return GraphQueryResult(node_count=0, notes="sync stub")

    def gnn_score(_step: int, _evidence: int, _nodes: int) -> float:
        return 0.5

    def analogues(_q: str) -> list[str]:
        return []

    def market_context(_q: str, cutoff: date, resolution: date) -> MarketContext:
        return MarketContext(
            market_id="sync",
            market_family="metaculus_binary",
            key_actors=[],
            region=None,
            cutoff_date=cutoff,
            resolution_date=resolution,
        )

    return AgentToolset(web_search=web_search, graph_query=graph_query, gnn_score=gnn_score, analogues=analogues, market_context=market_context)


def sync_template_outputs(template_output_dir: Path, memory: MemoryStore) -> SyncResult:
    scanned = imported = skipped_existing = resolved = resolve_skipped = 0
    toolset = _resolution_toolset()

    for jsonl_path in sorted(template_output_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                scanned += 1
                payload = json.loads(line)
                episode = _build_episode(payload)
                if memory.read_episode_by_id(episode.job_id) is not None:
                    skipped_existing += 1
                else:
                    memory.write_episode(episode)
                    imported += 1

                outcome = _is_resolved_true(payload)
                if outcome is not None:
                    try:
                        resolve_market(job_id=episode.job_id, outcome=outcome, memory=memory, tools=toolset)
                        resolved += 1
                    except AlreadyResolvedError:
                        resolve_skipped += 1

    return SyncResult(
        scanned=scanned,
        imported=imported,
        skipped_existing=skipped_existing,
        resolved=resolved,
        resolve_skipped=resolve_skipped,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync metac-bot-template JSONL outputs into harness memory store")
    parser.add_argument("--template-output-dir", required=True)
    parser.add_argument("--memory-dir", default=".harness_memory")
    args = parser.parse_args(argv)

    memory = JsonlMemoryStore(Path(args.memory_dir))
    result = sync_template_outputs(Path(args.template_output_dir), memory)
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
