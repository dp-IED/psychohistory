from __future__ import annotations

from pathlib import Path

from harness.tools.dataview_runtime import extract_dataview_queries, run_dataview_query
from harness.tools.strategy_runtime import build_vault_synthesis_bundle


def test_dataview_table_limit(tmp_path: Path) -> None:
    vault = tmp_path
    (vault / "runs").mkdir()
    (vault / "runs" / "a.md").write_text(
        "---\nquestion: Q1\ncategory: crypto\nbrier: 0.5\nhorizon_days: 10\napproach: x\n---\n",
        encoding="utf-8",
    )
    (vault / "runs" / "b.md").write_text(
        "---\nquestion: Q2\ncategory: crypto\nbrier: 0.2\nhorizon_days: 5\napproach: y\n---\n",
        encoding="utf-8",
    )
    q = 'TABLE question, brier\nFROM "runs"\nWHERE category = "crypto" AND horizon_days < 30\nSORT brier ASC\nLIMIT 1\n'
    out = run_dataview_query(vault, q)
    assert "Q2" in out
    assert "Q1" not in out or out.count("Q1") == 0


def test_strategy_bundle_includes_strategy_and_footer(tmp_path: Path) -> None:
    """Bundle includes strategy text and tool availability footer; Dataview queries
    are NOT pre-executed — the synthesis agent runs them itself via the CLI tool."""
    vault = tmp_path
    (vault / "runs").mkdir()
    (vault / "runs" / "r.md").write_text(
        '---\nquestion: Qz\ncategory: crypto\nbrier: 0.1\nhorizon_days: 7\n---\n',
        encoding="utf-8",
    )
    (vault / "_strategy.md").write_text(
        "# Strategy\n\nSynthesis protocol here.\n",
        encoding="utf-8",
    )
    bundle = build_vault_synthesis_bundle(vault, category="crypto", horizon_days=7)
    # Strategy content is included as plain text
    assert "# Strategy" in bundle
    assert "Synthesis protocol here" in bundle
    # Dataview runs are NOT pre-executed — no "Qz" in the bundle
    assert "Qz" not in bundle
    # Footer with tool availability is included
    assert "Available tools" in bundle
    assert "dataview_query" in bundle


def test_extract_dataview_queries() -> None:
    md = "Intro\n```dataview\nTABLE x\nFROM \"runs\"\n```\n"
    qs = extract_dataview_queries(md)
    assert len(qs) == 1
    assert "TABLE x" in qs[0]
