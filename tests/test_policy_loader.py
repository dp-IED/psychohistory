from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from harness.agent_loop import AgentToolset, GraphQueryResult, MarketContext, run_agent_loop
from harness.memory_schema import ToolCallRecord
from harness.memory_store import JsonlMemoryStore
from harness.policy_loader import DEFAULT_MAX_STEPS, PolicyConfig, load_policy, save_policy


def test_load_valid_policy(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        """---
blind_spot_checks:
  - electoral_legitimacy_check
max_steps: 2
convergence_epsilon: 0.02
shrinkage: 0.1
---

# Body here

Hello
""",
        encoding="utf-8",
    )
    cfg = load_policy(path)
    assert cfg.blind_spot_checks == ["electoral_legitimacy_check"]
    assert cfg.max_steps == 2
    assert cfg.convergence_epsilon == pytest.approx(0.02)
    assert cfg.shrinkage == pytest.approx(0.1)
    assert "# Body here" in cfg.body
    assert "Hello" in cfg.body


def test_load_missing_file_returns_default(tmp_path: Path) -> None:
    missing = tmp_path / "nope.md"
    cfg = load_policy(missing)
    assert cfg.blind_spot_checks == []
    assert cfg.body == ""
    assert cfg.max_steps == DEFAULT_MAX_STEPS


def test_load_partial_frontmatter_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        """---
blind_spot_checks: []
---

Only body.
""",
        encoding="utf-8",
    )
    cfg = load_policy(path)
    assert cfg.max_steps == DEFAULT_MAX_STEPS
    assert cfg.convergence_epsilon == 0.01
    assert cfg.shrinkage is None
    assert "Only body." in cfg.body


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    original = PolicyConfig(
        blind_spot_checks=["a", "b"],
        max_steps=3,
        convergence_epsilon=0.05,
        shrinkage=None,
        body="# Title\n\nText.\n",
    )
    save_policy(original, path)
    loaded = load_policy(path)
    assert loaded.blind_spot_checks == ["a", "b"]
    assert loaded.max_steps == 3
    assert loaded.convergence_epsilon == pytest.approx(0.05)
    assert loaded.shrinkage is None
    assert loaded.body.rstrip("\n") == original.body.rstrip("\n")


def test_run_agent_loop_accepts_policy_config(tmp_path: Path) -> None:
    policy = PolicyConfig(blind_spot_checks=[], max_steps=2, body="# Test policy")
    memory = JsonlMemoryStore(tmp_path / "memory")

    def ws(q: str, d: date) -> list[ToolCallRecord]:
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=q,
                as_of_time=f"{d.isoformat()}T00:00:00Z",
                evidence_count=1,
                notes="",
            )
        ]

    def gq(q: str, d: date) -> GraphQueryResult:
        _ = (q, d)
        return GraphQueryResult(node_count=8)

    def gs(s: int, e: int, n: int) -> float:
        _ = (e, n)
        return min(0.95, 0.52 + 0.01 * s)

    def an(q: str) -> list[str]:
        _ = q
        return []

    def mc(q: str, c: date, r: date) -> MarketContext:
        return MarketContext(
            market_id="test",
            market_family="test",
            key_actors=[],
            region=None,
            cutoff_date=c,
            resolution_date=r,
        )

    tools = AgentToolset(web_search=ws, graph_query=gq, gnn_score=gs, analogues=an, market_context=mc)
    result = run_agent_loop("Test Q?", date(2025, 1, 1), date(2025, 2, 1), policy, memory, tools)
    assert 0.0 <= result.final_p_yes <= 1.0
    assert "Policy context:" in result.reasoning_summary
    assert "# Test policy" in result.reasoning_summary


def test_unknown_blind_spot_check_skipped_for_policy_config(tmp_path: Path) -> None:
    policy = PolicyConfig(blind_spot_checks=["not_a_real_template"], max_steps=1, body="")
    memory = JsonlMemoryStore(tmp_path / "memory")

    def ws(q: str, d: date) -> list[ToolCallRecord]:
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=q,
                as_of_time=f"{d.isoformat()}T00:00:00Z",
                evidence_count=1,
                notes="",
            )
        ]

    def gq(q: str, d: date) -> GraphQueryResult:
        _ = (q, d)
        return GraphQueryResult(node_count=4)

    def gs(_s: int, _e: int, _n: int) -> float:
        return 0.55

    def an(q: str) -> list[str]:
        _ = q
        return []

    def mc(q: str, c: date, r: date) -> MarketContext:
        return MarketContext(
            market_id="test",
            market_family="test",
            key_actors=[],
            region=None,
            cutoff_date=c,
            resolution_date=r,
        )

    tools = AgentToolset(web_search=ws, graph_query=gq, gnn_score=gs, analogues=an, market_context=mc)
    result = run_agent_loop("Q?", date(2025, 1, 1), date(2025, 2, 1), policy, memory, tools)
    assert result.blind_spot_checks_fired == []
    assert "not_a_real_template" in result.blind_spot_checks_skipped
    assert "Unknown blind_spot checks skipped" in result.reasoning_summary
