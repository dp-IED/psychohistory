from pathlib import Path

from ingest.polymarket_branch_builder import (
    build_graph_artifact_from_record,
    build_portfolios_from_resolved_record,
    evaluate_branch_builder_against_gold,
    infer_market_family,
    load_gold_branch_cases,
    write_graph_artifacts_jsonl,
)
from ingest.polymarket_resolved import ResolvedMarketRecord
from schemas.polymarket_agentic import BranchType, Direction, HypothesisSide, validate_portfolio_against_policy


GOLD_PATH = Path("data/polymarket/gold_branch_dataset.json")


def _record(question: str, description: str = "", slug: str = "test-market") -> ResolvedMarketRecord:
    return ResolvedMarketRecord(
        id="test",
        slug=slug,
        question=question,
        description=description,
        category=None,
        market_type=None,
        condition_id="0x0",
        outcomes=["Yes", "No"],
        terminal_outcome_prices=[1.0, 0.0],
        resolved_outcome="Yes",
        volume=None,
        liquidity=None,
        start_date=None,
        end_date="2024-01-01T00:00:00Z",
        closed_time="2024-01-02T00:00:00Z",
        created_at=None,
        updated_at=None,
        clob_token_ids=[],
        url="https://polymarket.com/market/test-market",
        gamma_url="https://gamma-api.polymarket.com/markets?slug=test-market",
    )


def test_institutional_markers_take_precedence_over_country_negotiation_terms() -> None:
    record = _record(
        "Will the Senate vote on Iran sanctions before August?",
        "This resolves based on a formal Senate vote on a sanctions bill.",
        "senate-vote-iran-sanctions",
    )

    assert infer_market_family(record).value == "institutional_process"


def test_branch_builder_uses_family_scaffold_without_random_context_factor_enrichment() -> None:
    record = _record(
        "Will there be a ceasefire before August 1?",
        "Hamas and Israel negotiations with a mediator before the deadline.",
        "ceasefire-before-august",
    )

    yes_portfolio, no_portfolio = build_portfolios_from_resolved_record(record, as_of_time="2024-01-01T00:00:00Z")
    labels = [
        element.label
        for portfolio in (yes_portfolio, no_portfolio)
        for branch in portfolio.branches
        for element in branch.seed_elements
    ]

    assert labels
    assert all(not label.startswith("context factor:") for label in labels)


def test_macro_policy_print_includes_required_disruptor_branch() -> None:
    record = _record(
        "Fed decreases interest rates by 25 bps after July 2024 meeting?",
        "The decision is made by the Federal Open Market Committee after the scheduled meeting.",
        "fed-decreases-interest-rates-by-25-bps-after-july-2024-meeting",
    )

    portfolios = build_portfolios_from_resolved_record(record, as_of_time="2024-01-01T00:00:00Z")

    for portfolio in portfolios:
        assert BranchType.DISRUPTOR in portfolio.branch_types_present()
        assert validate_portfolio_against_policy(portfolio) == []


def test_builder_matches_gold_family_branch_and_label_contracts() -> None:
    cases = load_gold_branch_cases(GOLD_PATH)

    for case in cases:
        record = case.record
        assert infer_market_family(record).value == case.expected_family

        portfolios = build_portfolios_from_resolved_record(record, as_of_time="2024-01-01T00:00:00Z")
        assert {portfolio.hypothesis.side for portfolio in portfolios} == {HypothesisSide.YES, HypothesisSide.NO}

        for portfolio in portfolios:
            assert portfolio.hypothesis.market_frame.resolved_outcome == record.resolved_outcome
            assert portfolio.hypothesis.market_frame.family.value == case.expected_family
            assert {branch.branch_type.value for branch in portfolio.branches} >= set(case.expected_required_branches)
            assert len(portfolio.prerequisites) >= case.expected_min_prerequisites
            assert validate_portfolio_against_policy(portfolio) == []

            local = next(branch for branch in portfolio.branches if branch.branch_type == BranchType.LOCAL)
            assert {Direction.FOR, Direction.AGAINST} <= local.directions_present()


def test_graph_artifact_materializes_gold_event_branches_and_target_labels() -> None:
    cases = load_gold_branch_cases(GOLD_PATH)

    for case in cases:
        artifact = build_graph_artifact_from_record(case.record, as_of_time="2024-01-01T00:00:00Z")
        node_types = {node.type for node in artifact.nodes}
        assert node_types >= set(case.expected_node_types)
        assert any(edge.type == "HAS_BRANCH" for edge in artifact.edges)
        assert any(edge.type == "SEEDS_ELEMENT" for edge in artifact.edges)
        assert any(edge.type == "GATES_OUTCOME" for edge in artifact.edges)

        target = next(item for item in artifact.target_table if item.name == "resolved_yes")
        assert target.value == case.expected_target_value
        assert target.metadata["resolved_outcome"] == case.record.resolved_outcome
        assert artifact.metadata["cutoff_policy"] == "terminal resolution is label only; PIT evidence must be attached downstream"


def test_graph_artifact_writer_materializes_jsonl(tmp_path: Path) -> None:
    cases = load_gold_branch_cases(GOLD_PATH)
    output = tmp_path / "branch_graphs.jsonl"

    count = write_graph_artifacts_jsonl([case.record for case in cases], output, as_of_time="2024-01-01T00:00:00Z")

    assert count == len(cases)
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == len(cases)
    assert '"artifact_format":"graph_artifact_v1"' in rows[0]


def test_gold_dataset_evaluation_requires_perfect_structural_and_semantic_match() -> None:
    report = evaluate_branch_builder_against_gold(GOLD_PATH)

    assert report.case_count == 30
    assert report.family_accuracy == 1.0
    assert report.branch_recall == 1.0
    assert report.target_accuracy == 1.0
    assert report.policy_pass_rate == 1.0
    assert report.content_coverage == 1.0
    assert report.branch_content_recall == 1.0
    assert report.expressiveness_score == 1.0
    assert report.failures == []
