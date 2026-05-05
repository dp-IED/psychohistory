from pathlib import Path

from baselines.weimar_graph_jepa_pilot import DOMAINS, build_weimar_states, run_pilot


def test_weimar_states_cover_tri_domain_and_pit_order():
    states = build_weimar_states()
    assert len(states) == 15
    assert [s.year for s in states] == sorted(s.year for s in states)
    assert DOMAINS == ("economic", "cultural", "socio_political")
    for state in states:
        assert state.matrix.shape == (3, 3)
        assert state.matrix.min() >= 0.0
        assert state.matrix.max() <= 1.0
        assert state.events
        assert state.regime_label


def test_weimar_graph_jepa_pilot_writes_artifacts(tmp_path: Path):
    summary = run_pilot(tmp_path)
    assert summary["train_examples"] == 33
    assert summary["eval_examples"] == 9
    assert set(summary["by_domain"]) == set(DOMAINS)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "eval_rows.csv").exists()
    assert (tmp_path / "weimar_states.jsonl").exists()
    assert -1.0 <= summary["overall"]["jepa_cosine"] <= 1.0
