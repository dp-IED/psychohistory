from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pit_subgraph_audit import _country_ok
from scripts.warehouse_quality_gate import infer_country_from_admin1, validate_manifest


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def _row(
    *,
    admin1_code: str | None,
    first_seen: str | None = "2011-01-01",
    entity_hint_keys: list[str] | None = None,
) -> dict:
    extensions = {}
    if entity_hint_keys is not None:
        extensions["entity_hint_keys"] = entity_hint_keys
    return {
        "node_id": f"test|{admin1_code or 'none'}",
        "admin1_code": admin1_code,
        "first_seen": first_seen,
        "extensions": extensions,
    }


@pytest.mark.parametrize(
    ("admin1_code", "expected_country", "source"),
    [
        ("EG", "EG", "native_iso2"),
        ("EG-C", "EG", "native_prefixed"),
        ("Cairo", "EG", "alias_map"),
        ("North Sinai", "EG", "alias_map"),
        ("Tripolitania", None, "unmapped_label"),
        (None, None, "missing"),
        ("  ", None, "blank"),
    ],
)
def test_quality_gate_country_inference_sources(admin1_code, expected_country, source) -> None:
    assert infer_country_from_admin1(admin1_code) == (expected_country, source)


@pytest.mark.parametrize(
    ("admin1_code", "expected_country", "ok"),
    [
        ("EG", "EG", True),
        ("EG-C", "EG", True),
        ("Cairo", "EG", True),
        ("North Sinai", "EG", True),
        ("Cairo", "LY", False),
        ("Tripolitania", "LY", False),
        (None, "EG", False),
    ],
)
def test_pit_audit_country_ok_uses_same_alias_contract(admin1_code, expected_country, ok) -> None:
    assert _country_ok(admin1_code, expected_country) is ok


def test_pit_audit_country_ok_skips_unknown_expected_country() -> None:
    assert _country_ok("Cairo", None) is None


def test_quality_gate_passes_alias_mapped_admin_labels_with_complete_hints(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _row(admin1_code="Cairo", entity_hint_keys=["cairo protest"]),
            _row(admin1_code="North Sinai", entity_hint_keys=["sinai"]),
            _row(admin1_code="EG-C", entity_hint_keys=["egypt"]),
        ],
    )
    result = validate_manifest(manifest)
    assert result["passed"] is True
    assert result["metrics"]["invalid_country_rows"] == 0
    assert result["metrics"]["unmapped_admin1_rows"] == 0
    assert result["metrics"]["missing_entity_hint_rows"] == 0
    assert result["country_source_counts"] == {"alias_map": 2, "native_prefixed": 1}


def test_quality_gate_fails_closed_on_unmapped_missing_first_seen_or_missing_hints(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _row(admin1_code="Tripolitania", entity_hint_keys=["tripoli"]),
            _row(admin1_code="Cairo", first_seen=None, entity_hint_keys=["cairo"]),
            _row(admin1_code="EG-C", entity_hint_keys=[]),
        ],
    )
    result = validate_manifest(manifest)
    assert result["passed"] is False
    assert result["checks"] == {
        "country_inference_complete": False,
        "no_unmapped_admin1_labels": False,
        "first_seen_complete": False,
        "entity_hint_keys_complete": False,
    }
    assert result["metrics"]["invalid_country_rows"] == 1
    assert result["metrics"]["unmapped_admin1_rows"] == 1
    assert result["metrics"]["missing_first_seen_rows"] == 1
    assert result["metrics"]["missing_entity_hint_rows"] == 1
