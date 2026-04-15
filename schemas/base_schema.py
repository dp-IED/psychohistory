"""Seed schema for the first forecasting loop.

The active project target is a temporally clean geopolitical contention
forecast, not a broad historical ontology. Keep this schema small until real
data and rolling backtests prove that more vocabulary is needed.
"""

from __future__ import annotations

from schemas.schema_types import EdgeSpec, GraphSchema, Layer, NodeSpec, ProjectionSpec


def _node(
    name: str,
    layer: Layer,
    description: str,
    attributes: dict[str, str] | None = None,
) -> NodeSpec:
    return NodeSpec(
        name=name,
        primary_layer=layer,
        description=description,
        attributes=attributes or {},
    )


def _edge(
    name: str,
    description: str,
    source_layers: list[Layer],
    target_layers: list[Layer],
) -> EdgeSpec:
    return EdgeSpec(
        name=name,
        description=description,
        allowed_source_layers=source_layers,
        allowed_target_layers=target_layers,
    )


def get_seed_graph_schema() -> GraphSchema:
    """Return the narrow graph IR needed for Stage 0/1 forecasting work."""

    nodes = [
        _node(
            "Actor",
            Layer.INSTITUTIONAL,
            "State, organization, armed group, party, movement, or other event participant.",
            {"canonical_id": "str", "aliases": "list[str]", "actor_kind": "str"},
        ),
        _node(
            "Location",
            Layer.MATERIAL,
            "Country, region, city, or coordinate-bearing area used for event aggregation.",
            {"country_code": "str", "admin_level": "str", "latitude": "float", "longitude": "float"},
        ),
        _node(
            "Event",
            Layer.MATERIAL,
            "Observed dated interaction or incident from an event source.",
            {
                "event_type": "str",
                "event_date": "date",
                "severity": "float",
                "source_event_id": "str",
            },
        ),
        _node(
            "Narrative",
            Layer.IDEOLOGICAL,
            "Evidence-backed issue frame or topic cluster linked to actors or events.",
            {"topic_id": "str", "stance": "str", "window_start": "date", "window_end": "date"},
        ),
        _node(
            "Market",
            Layer.EPISTEMIC,
            "Prediction market or other belief signal treated as a feature, not ground truth.",
            {"market_id": "str", "question": "str", "close_date": "date"},
        ),
        _node(
            "Source",
            Layer.EPISTEMIC,
            "Dataset, feed, article cluster, or other provenance source.",
            {"source_name": "str", "retrieved_at": "datetime"},
        ),
    ]

    edges = [
        _edge(
            "participates_in",
            "Links an actor to an observed event.",
            [Layer.INSTITUTIONAL],
            [Layer.MATERIAL],
        ),
        _edge(
            "targets",
            "Links an event to its target actor when source data supports one.",
            [Layer.MATERIAL],
            [Layer.INSTITUTIONAL],
        ),
        _edge(
            "occurs_in",
            "Links an event to a location.",
            [Layer.MATERIAL],
            [Layer.MATERIAL],
        ),
        _edge(
            "located_in",
            "Links actors or places to containing locations.",
            [Layer.INSTITUTIONAL, Layer.MATERIAL],
            [Layer.MATERIAL],
        ),
        _edge(
            "reports",
            "Links a source to an event it reports or codes.",
            [Layer.EPISTEMIC],
            [Layer.MATERIAL],
        ),
        _edge(
            "mentions",
            "Links a source or narrative to an actor, event, market, or location.",
            [Layer.EPISTEMIC, Layer.IDEOLOGICAL],
            [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.EPISTEMIC],
        ),
        _edge(
            "frames",
            "Links a narrative to the actor, event, or location it frames.",
            [Layer.IDEOLOGICAL],
            [Layer.INSTITUTIONAL, Layer.MATERIAL],
        ),
        _edge(
            "prices_belief_about",
            "Links a market to the event, actor, or location its question concerns.",
            [Layer.EPISTEMIC],
            [Layer.MATERIAL, Layer.INSTITUTIONAL],
        ),
    ]

    return GraphSchema(
        name="psychohistory_forecast_ir",
        version="0.2.0",
        benchmark_pack="forecast-charter-v1",
        ontology_compatibility="stage-0",
        node_types={node.name: node for node in nodes},
        edge_types={edge.name: edge for edge in edges},
        projection=ProjectionSpec(
            dim=32,
            required=True,
            per_type_optional={
                "Actor": 8,
                "Location": 4,
                "Event": 12,
                "Narrative": 8,
                "Market": 6,
            },
        ),
        layers_declared=[
            Layer.MATERIAL,
            Layer.INSTITUTIONAL,
            Layer.IDEOLOGICAL,
            Layer.EPISTEMIC,
        ],
        extensions={
            "forecast_target": "next_7_day_geopolitical_contention",
            "stage": "charter_and_temporal_data_substrate",
        },
    )
