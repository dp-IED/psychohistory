from __future__ import annotations

import json
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


def test_ground_snapshot_nodes_resolves_from_dump_manifest_without_api(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "wikidata"
    cache = tmp_path / "cache.json"
    nodes = [
        {
            "id": "a1",
            "type": "Actor",
            "label": "Emmanuel Macron",
            "attributes": {"country_code": "FR"},
        }
    ]

    out, stats = ground_snapshot_nodes(
        nodes,
        cache_path=cache,
        request_delay_s=0.0,
        log=False,
        origin="2021-06-07",
        dump_manifest_path=fixture_root / "manifest.json",
        api_fallback=False,
    )

    assert out[0]["external_ids"]["wikidata"] == "Q3052772"
    assert stats["resolved"] == 1
    assert stats["dump_resolved"] == 1
    assert stats["api_calls"] == 0
    cached = json.loads(cache.read_text(encoding="utf-8"))
    entry = cached["Actor:emmanuel macron@wd-mini-2021-01-01"]
    assert entry["method"] == "dump_slice"
    assert entry["dump_id"] == "wd-mini-2021-01-01"
    assert entry["dump_date"] == "2021-01-01"


def test_ground_snapshot_nodes_selects_latest_dump_at_or_before_origin(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "wikidata"
    nodes = [{"id": "a1", "type": "Actor", "label": "Francois Bayrou"}]

    out, stats = ground_snapshot_nodes(
        nodes,
        cache_path=tmp_path / "cache.json",
        request_delay_s=0.0,
        log=False,
        origin="2022-06-06",
        dump_manifest_path=fixture_root / "manifest.json",
        api_fallback=False,
    )

    assert out[0]["external_ids"]["wikidata"] == "Q313250"
    assert stats["dump_id"] == "wd-mini-2022-01-01"
    assert stats["dump_date"] == "2022-01-01"


def test_apply_wikidata_grounding_writes_dump_metadata(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "wikidata"
    payload = {
        "nodes": [
            {
                "id": "a1",
                "type": "Actor",
                "label": "Emmanuel Macron",
                "attributes": {"country_code": "FR"},
            }
        ],
        "metadata": {},
    }

    apply_wikidata_grounding(
        payload,
        cache_path=tmp_path / "cache.json",
        request_delay_s=0.0,
        log=False,
        origin="2021-06-07",
        dump_manifest_path=fixture_root / "manifest.json",
        api_fallback=False,
    )

    block = payload["metadata"]["wikidata_grounding"]
    assert block["resolved"] == 1
    assert block["dump_id"] == "wd-mini-2021-01-01"
    assert block["dump_date"] == "2021-01-01"
