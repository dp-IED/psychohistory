from __future__ import annotations

from pathlib import Path

import pytest

from evals.wikidata_grounding import apply_wikidata_grounding, ground_snapshot_nodes


def test_ground_snapshot_nodes_skips_non_entity_types() -> None:
    nodes = [{"id": "e1", "type": "Event", "label": "x"}]
    out, stats = ground_snapshot_nodes(nodes, cache_path=None, request_delay_s=0.0)
    assert out == nodes
    assert stats["attempted"] == 0


def test_ground_snapshot_nodes_skips_existing_qid() -> None:
    nodes = [
        {
            "id": "a1",
            "type": "Actor",
            "label": "Test Actor",
            "external_ids": {"wikidata": "Q12345"},
        }
    ]
    out, stats = ground_snapshot_nodes(nodes, cache_path=None, request_delay_s=0.0)
    assert stats["skipped_existing_qid"] == 1
    assert out[0]["external_ids"]["wikidata"] == "Q12345"


def test_ground_snapshot_nodes_resolves_via_monkeypatched_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.wikidata_grounding as wg

    def fake_search(label: str, **kwargs: object) -> dict[str, object]:
        return {"id": "Q999", "label": label}

    monkeypatch.setattr(wg, "search_wikidata_entity", fake_search)

    cache = tmp_path / "cache.json"
    nodes = [
        {
            "id": "a1",
            "type": "Actor",
            "label": "Unique Label XYZ",
            "attributes": {"country_code": "FR"},
        }
    ]
    out, stats = ground_snapshot_nodes(nodes, cache_path=cache, request_delay_s=0.0)
    assert stats["resolved"] == 1
    assert stats["api_calls"] == 1
    assert out[0]["external_ids"]["wikidata"] == "Q999"
    assert cache.exists()


def test_apply_wikidata_grounding_sets_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import evals.wikidata_grounding as wg

    monkeypatch.setattr(
        wg,
        "search_wikidata_entity",
        lambda label, **kwargs: {"id": "Q1", "label": label},
    )

    payload = {
        "nodes": [{"id": "a", "type": "Actor", "label": "L", "attributes": {"country_code": "FR"}}],
        "metadata": {},
    }
    apply_wikidata_grounding(payload, cache_path=tmp_path / "c.json")
    assert "wikidata_grounding" in payload["metadata"]
    assert payload["metadata"]["wikidata_grounding"]["resolved"] >= 1
