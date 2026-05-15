"""Factory wiring real retrieval + synthesis tools into AgentToolset."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from harness.agent_loop import AgentToolset, GraphQueryResult, MarketContext
from harness.memory_schema import ToolCallRecord
from harness.memory_store import MemoryStore
from harness.policy_loader import PolicyConfig, load_policy
from harness.query_mapper import WebSearchRequest
from harness.tools.analogues import (
    analogues_to_tool_strings,
    find_analogues,
    find_vault_run_analogues,
)
from harness.tools.evidence_graph import EvidenceGraph, build_evidence_graph
from harness.tools.llm_synthesis import llm_synthesize_forecast, synthesis_policy_hint
from harness.tools.loop_context import (
    get_research_cutoff,
    get_research_market_family,
    get_research_question,
    get_research_tool_calls,
    get_vault_synthesis_context,
)
from harness.tools.market_context import resolve_market_context
from harness.tools.web_search import AskNewsSearchTool, rate_limited_asknews_call


def build_real_toolset(
    memory: MemoryStore,
    policy: PolicyConfig,
    *,
    vault_dir: Path | None = None,
) -> AgentToolset:
    """Close over memory; reads latest `policy.md` at each gnn_score for synthesis hints."""

    _ = policy
    vault_root = vault_dir.expanduser().resolve() if vault_dir else None

    api_key = os.environ.get("ASKNEWS_API_KEY", "").strip()
    ask: AskNewsSearchTool | None
    try:
        ask = AskNewsSearchTool(api_key) if api_key else None
    except ValueError:
        ask = None

    last_evidence: dict[str, EvidenceGraph | None] = {"g": None}

    def web_search(query: str, as_of_date: date) -> list[ToolCallRecord]:
        if ask is None:
            return [
                ToolCallRecord(
                    tool_name="web_search",
                    query=query,
                    as_of_time=f"{as_of_date.isoformat()}T00:00:00Z",
                    evidence_count=0,
                    notes="asknews_disabled_stub",
                )
            ]
        req = WebSearchRequest(
            query=query,
            as_of_date=as_of_date,
            market_family=(get_research_market_family() or "general"),
            blind_spot_check="live_search",
        )
        return rate_limited_asknews_call(ask, req)

    def graph_query(question: str, cutoff: date) -> GraphQueryResult:
        calls = list(get_research_tool_calls() or [])
        g = build_evidence_graph(calls, question, cutoff)
        last_evidence["g"] = g
        return GraphQueryResult(node_count=len(g.nodes), notes=g.summary[:2000])

    def gnn_score(step: int, evidence_count: int, nodes: int) -> float:
        g = last_evidence["g"]
        q = get_research_question() or ""
        co = get_research_cutoff()
        if co is None:
            raise RuntimeError(
                "gnn_score: research cutoff date was not set in contextvar. "
                "Cannot proceed without a known cutoff — date.today() would leak out-of-bounds information."
            )
        cutoff = co
        past: list = []
        fam = get_research_market_family()
        if fam:
            past = find_analogues(q, fam, memory, max_results=5)
            if vault_root is not None:
                v_rows = find_vault_run_analogues(q, fam, vault_root, max_results=5)
                merged: dict[str, dict] = {}
                for row in v_rows + past:
                    key = str(row.get("question", ""))
                    if key and key not in merged:
                        merged[key] = row
                past = list(merged.values())[:8]
        if g is None:
            g = build_evidence_graph([], q, cutoff)
        live_policy = load_policy()
        vault_ctx = (get_vault_synthesis_context() or "").strip()
        p_yes, _reason = llm_synthesize_forecast(
            q,
            cutoff,
            g,
            past,
            synthesis_policy_hint(live_policy),
            live_policy.shrinkage,
            step=step,
            evidence_count=evidence_count,
            node_count=nodes,
            vault_context=vault_ctx,
        )
        return float(p_yes)

    def analogues(question_text: str) -> list[str]:
        fam = get_research_market_family()
        if not fam:
            return []
        rows = find_analogues(question_text, fam, memory, max_results=5)
        if vault_root is not None:
            v_rows = find_vault_run_analogues(question_text, fam, vault_root, max_results=5)
            merged: dict[str, dict] = {}
            for row in v_rows + rows:
                key = str(row.get("question", ""))
                if key and key not in merged:
                    merged[key] = row
            rows = list(merged.values())[:8]
        return analogues_to_tool_strings(rows)

    def market_context(
        question_text: str, cutoff: date, resolution: date
    ) -> MarketContext:
        return resolve_market_context(question_text, cutoff, resolution)

    return AgentToolset(
        web_search=web_search,
        graph_query=graph_query,
        gnn_score=gnn_score,
        analogues=analogues,
        market_context=market_context,
    )


__all__ = ["build_real_toolset"]
