from __future__ import annotations

from datetime import date
from pathlib import Path

from harness.memory_schema import EpisodicRecord, ToolCallRecord
from scripts.export_to_obsidian import build_run_note, export_resolved_episodes


def test_build_run_note_frontmatter_and_reasoning(tmp_path: Path) -> None:
    ep = EpisodicRecord(
        job_id="job-abc123",
        market_id="m1",
        market_family="politics",
        question="Will X happen?",
        resolution_date=date(2026, 5, 14),
        cutoff_date=date(2026, 5, 10),
        blind_spot_checks_fired=["geopolitical_stability_check"],
        blind_spot_checks_skipped=[],
        tool_calls=[
            ToolCallRecord(
                tool_name="web_search",
                query="saudi trump",
                as_of_time="2026-05-10T00:00:00Z",
                evidence_count=2,
                notes="ok",
            )
        ],
        subgraph_node_count=3,
        gnn_score_trajectory=[0.52],
        final_p_yes=0.52,
        confidence_interval=(0.44, 0.60),
        brier_score=0.231,
        misses=[],
        notes="Full reasoning text here.",
    )
    md = build_run_note(ep, market_label="polymarket")
    assert md.startswith("---\n")
    assert "run_id: job-abc123" in md
    assert "category: politics" in md
    assert "p_yes:" in md
    assert "## Reasoning" in md
    assert "Full reasoning text here." in md
    assert "[[geopolitical_stability_check]]" in md


def test_export_resolved_episodes_filters_since(tmp_path: Path) -> None:
    mem = tmp_path / "mem"
    vault = tmp_path / "vault"
    store_dir = mem
    # minimal jsonl via store API
    from harness.memory_store import JsonlMemoryStore

    js = JsonlMemoryStore(store_dir)
    js.write_episode(
        EpisodicRecord(
            job_id="job-old",
            market_id="m",
            market_family="crypto",
            question="Old?",
            resolution_date=date(2026, 5, 1),
            cutoff_date=date(2026, 4, 20),
            blind_spot_checks_fired=[],
            blind_spot_checks_skipped=[],
            tool_calls=[],
            subgraph_node_count=0,
            gnn_score_trajectory=[0.5],
            final_p_yes=0.5,
            confidence_interval=None,
            brier_score=0.1,
            misses=[],
            notes="n",
        )
    )
    js.write_episode(
        EpisodicRecord(
            job_id="job-new",
            market_id="m",
            market_family="crypto",
            question="New?",
            resolution_date=date(2026, 5, 20),
            cutoff_date=date(2026, 5, 10),
            blind_spot_checks_fired=[],
            blind_spot_checks_skipped=[],
            tool_calls=[],
            subgraph_node_count=0,
            gnn_score_trajectory=[0.5],
            final_p_yes=0.5,
            confidence_interval=None,
            brier_score=0.2,
            misses=[],
            notes="n",
        )
    )
    out = export_resolved_episodes(store_dir, vault, runs_subdir="runs", since=date(2026, 5, 14))
    assert len(out) == 1
    assert out[0].name == "job-new.md"
