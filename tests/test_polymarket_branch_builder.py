from pathlib import Path

from ingest.polymarket_branch_builder import (
    build_graph_artifact_from_record,
    build_portfolios_from_resolved_record,
    evaluate_branch_builder_against_gold,
    infer_market_family,
    load_gold_branch_cases,
    write_graph_artifacts_jsonl,
)
from schemas.polymarket_agentic import BranchType, Direction, HypothesisSide, validate_portfolio_against_policy


GOLD_PATH = Path("data/polymarket/gold_branch_dataset.json")


def test_builder_matches_gold_family_branch_and_label_contracts() -> None:
    cases = load_gold_branch_cases(GOLD_PATH)

    for case in cases:
        record = case.record
        assert infer_market_family(record).value == case.expected_family

        portfolios = build_portfolios_from_resolved_record(record, as_of_time="2024-01-01T00:00:00Z")
        assert {portfolio.hypothesis.side for portfolio in portfolios} == {HypothesisSide.YES, HypothesisSide.NO}

        observed = next(portfolio for portfolio in portfolios if portfolio.hypothesis.side.value == record.resolved_outcome.upper())
        assert observed.hypothesis.market_frame.resolved_outcome == record.resolved_outcome
        assert observed.hypothesis.market_frame.family.value == case.expected_family
        assert {branch.branch_type.value for branch in observed.branches} >= set(case.expected_required_branches)
        assert len(observed.prerequisites) >= case.expected_min_prerequisites
        assert validate_portfolio_against_policy(observed) == []

        local = next(branch for branch in observed.branches if branch.branch_type == BranchType.LOCAL)
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


def test_gold_dataset_evaluation_requires_perfect_structural_match() -> None:
    report = evaluate_branch_builder_against_gold(GOLD_PATH)

    assert report.case_count == 3
    assert report.family_accuracy == 1.0
    assert report.branch_recall == 1.0
    assert report.target_accuracy == 1.0
    assert report.policy_pass_rate == 1.0
    assert report.failures == []
