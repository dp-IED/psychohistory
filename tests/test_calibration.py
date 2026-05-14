from __future__ import annotations

from datetime import date

import pytest

from harness.calibration import apply_shrinkage, compute_calibration
from harness.memory_schema import EpisodicRecord
from harness.memory_store import JsonlMemoryStore, NullMemoryStore


def test_empty_memory_returns_insufficient_data() -> None:
    memory = NullMemoryStore()
    report = compute_calibration(memory)
    assert report.insufficient_data is True
    assert report.resolved_count == 0
    assert report.suggested_shrinkage == 0.0


def test_fewer_than_10_episodes_insufficient_data(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")

    for i in range(5):
        e = _make_episode(job_id=f"job-{i}", brier=0.16, p_yes=0.6)
        memory.write_episode(e)
        memory.update_episode_brier(f"job-{i}", 0.16, [])

    report = compute_calibration(memory)
    assert report.insufficient_data is True
    assert report.resolved_count == 5
    assert report.suggested_shrinkage == 0.0


def test_shrinkage_increases_with_overconfidence(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")

    for i in range(10):
        e = _make_episode(job_id=f"job-{i}", brier=0.81, p_yes=0.9)
        memory.write_episode(e)
        memory.update_episode_brier(f"job-{i}", 0.81, [])

    report = compute_calibration(memory)
    assert not report.insufficient_data
    assert report.overconfidence_bias == pytest.approx(0.4)
    assert report.suggested_shrinkage == pytest.approx(0.24)


def test_shrinkage_capped_at_0_3(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")

    for i in range(10):
        e = _make_episode(job_id=f"job-{i}", brier=1.0, p_yes=1.0)
        memory.write_episode(e)
        memory.update_episode_brier(f"job-{i}", 1.0, [])

    report = compute_calibration(memory)
    assert report.overconfidence_bias == pytest.approx(0.5)
    assert report.suggested_shrinkage == pytest.approx(0.3)


def test_by_category_groups_correctly(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")

    for i in range(5):
        e = _make_episode(job_id=f"pol-{i}", market_family="politics", brier=0.16, p_yes=0.6)
        memory.write_episode(e)
        memory.update_episode_brier(f"pol-{i}", 0.16, [])
    for i in range(5):
        e = _make_episode(job_id=f"econ-{i}", market_family="economics", brier=0.04, p_yes=0.8)
        memory.write_episode(e)
        memory.update_episode_brier(f"econ-{i}", 0.04, [])

    report = compute_calibration(memory)
    assert "politics" in report.by_category
    assert "economics" in report.by_category
    assert report.by_category["politics"] == pytest.approx(0.16)
    assert report.by_category["economics"] == pytest.approx(0.04)


def test_overall_brier_averages_correctly(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")

    pairs = [
        (0.6, 0.16),
        (0.7, 0.09),
        (0.9, 0.81),
        (0.5, 0.25),
        (0.8, 0.04),
        (0.6, 0.16),
        (0.7, 0.09),
        (0.9, 0.01),
        (0.3, 0.09),
        (0.4, 0.36),
    ]
    for i, (p_yes, brier) in enumerate(pairs):
        e = _make_episode(job_id=f"job-{i}", brier=brier, p_yes=p_yes)
        memory.write_episode(e)
        memory.update_episode_brier(f"job-{i}", brier, [])

    report = compute_calibration(memory)
    expected_mean = sum([0.16, 0.09, 0.81, 0.25, 0.04, 0.16, 0.09, 0.01, 0.09, 0.36]) / 10
    assert report.overall_brier == pytest.approx(expected_mean)
    assert report.resolved_count == 10
    assert not report.insufficient_data


def test_apply_shrinkage_pulls_toward_0_5() -> None:
    assert apply_shrinkage(0.8, 0.0) == pytest.approx(0.8)
    assert apply_shrinkage(0.2, 0.0) == pytest.approx(0.2)

    assert apply_shrinkage(0.8, 1.0) == pytest.approx(0.5)
    assert apply_shrinkage(0.2, 1.0) == pytest.approx(0.5)

    assert apply_shrinkage(0.8, 0.25) == pytest.approx(0.725)
    assert apply_shrinkage(0.2, 0.25) == pytest.approx(0.275)


def test_apply_shrinkage_symmetric() -> None:
    shrinkage = 0.2
    result_high = apply_shrinkage(0.8, shrinkage)
    result_low = apply_shrinkage(0.2, shrinkage)
    assert result_high == pytest.approx(1.0 - result_low)


def _make_episode(
    *,
    job_id: str = "job-0",
    market_family: str = "test_family",
    market_id: str = "market-0",
    brier: float | None = None,
    p_yes: float = 0.5,
    cutoff: date = date(2024, 1, 1),
    resolution_date: date = date(2024, 6, 1),
) -> EpisodicRecord:
    return EpisodicRecord(
        job_id=job_id,
        market_id=market_id,
        market_family=market_family,
        question="Test question",
        resolution_date=resolution_date,
        cutoff_date=cutoff,
        blind_spot_checks_fired=[],
        blind_spot_checks_skipped=[],
        tool_calls=[],
        subgraph_node_count=0,
        gnn_score_trajectory=[],
        final_p_yes=p_yes,
        confidence_interval=None,
        brier_score=brier,
        misses=[],
        notes="",
    )
