from __future__ import annotations

import pytest

from evals import wikidata_linking
from evals.graph_artifact_contract import GRAPH_ARTIFACT_FORMAT, GraphArtifactV1, normalize_graph_artifact
from schemas.base_schema import get_seed_graph_schema
from schemas.schema_registry import load_graph_schema


def test_seed_schema_is_narrow_forecasting_scope() -> None:
    schema = get_seed_graph_schema()

    assert schema.name == "psychohistory_forecast_ir"
    assert set(schema.node_types) == {
        "Actor",
        "Location",
        "Event",
        "Narrative",
        "Market",
        "Source",
    }
    assert "participates_in" in schema.edge_types
    assert schema.extensions["forecast_target"] == "next_7_day_geopolitical_contention"


def test_schema_registry_loads_seed_schema() -> None:
    schema = load_graph_schema("schemas.base_schema")

    assert schema.version == "0.2.0"
    assert schema.projection.required is True


def test_graph_artifact_contract_rejects_dangling_edges() -> None:
    with pytest.raises(ValueError, match="edge target not found"):
        GraphArtifactV1.model_validate(
            {
                "artifact_format": GRAPH_ARTIFACT_FORMAT,
                "probe_id": "forecast-slice",
                "nodes": [{"id": "event-1", "type": "Event"}],
                "edges": [{"source": "event-1", "target": "missing", "type": "occurs_in"}],
            }
        )


def test_normalize_graph_artifact_preserves_temporal_fields() -> None:
    artifact = normalize_graph_artifact(
        probe_id="forecast-slice",
        schema_version="0.2.0",
        graph={
            "nodes": [
                {
                    "id": "event-1",
                    "type": "Event",
                    "time": {"start": "2024-01-01", "end": "2024-01-07", "granularity": "day"},
                    "source_event_id": "source:1",
                },
                {"id": "loc-1", "type": "Location"},
            ],
            "edges": [{"source": "event-1", "target": "loc-1", "type": "occurs_in"}],
        },
        default_split="train",
    )

    assert artifact.artifact_format == GRAPH_ARTIFACT_FORMAT
    assert artifact.nodes[0].time.start == "2024-01-01"
    assert artifact.nodes[0].attributes["source_event_id"] == "source:1"
    assert artifact.edges[0].train_eval_split == "train"


def test_wikidata_qid_extraction_accepts_common_shapes() -> None:
    assert wikidata_linking.wikidata_qid_for_node({"external_ids": {"wikidata": "Q42"}}) == "Q42"
    assert wikidata_linking.wikidata_qid_for_node({"same_as": ["https://www.wikidata.org/wiki/Q123"]}) == "Q123"
    assert wikidata_linking.wikidata_qid_for_node({"id": "wd:Q456"}) == "Q456"


def test_wikidata_enrichment_updates_existing_seed_entity_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search(label: str, *, timeout: float = 10.0, max_retries: int = 2) -> dict[str, str]:
        return {"id": "Q220", "label": label}

    monkeypatch.setattr(wikidata_linking, "search_wikidata_entity", fake_search)
    seed_data = {"seed_entities": {"places": ["Rome"]}}
    graph_payload = {
        "nodes": [{"id": "seed_entity__rome", "type": "Location", "label": "Rome", "seed_entity": True}],
        "edges": [],
    }

    enriched = wikidata_linking.enrich_graph_with_wikidata(
        seed_data,
        graph_payload,
        request_delay_s=0.0,
    )

    assert len(enriched["nodes"]) == 1
    assert enriched["nodes"][0]["external_ids"]["wikidata"] == "Q220"
