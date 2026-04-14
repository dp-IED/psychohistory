"""
Seed graph IR schema (human- and agent-editable).

Autoresearch copies this module into candidate sandboxes; keep changes coherent.
Node/edge names cover the union of current benchmark probes (must_represent +
may_represent); extend via autoresearch proposals when new probes land.
"""

from __future__ import annotations

from typing import Any

from schemas.schema_types import (
    EpistemicSpec,
    GraphSchema,
    Layer,
    NodeSpec,
    EdgeSpec,
    ProjectionSpec,
    TemporalSpec,
)

# Union of probe YAML `schema_requirements.must_represent.node_types` (+ may_represent).
_NODE_NAMES: tuple[str, ...] = (
    "Actor",
    "AdministrativeRegion",
    "AidChannel",
    "ArmedGroup",
    "BusinessAssociation",
    "CivicMovement",
    "Claim",
    "Coalition",
    "Commodity",
    "ConstitutionalBody",
    "Counterweight",
    "DebtRegime",
    "DependencyRegime",
    "DevelopmentProgram",
    "EconomicSector",
    "Empire",
    "Enterprise",
    "Event",
    "ExternalConstraint",
    "ExternalPower",
    "ExtractionChannel",
    "ExtractiveChannel",
    "FrontierPeople",
    "FrontierPolity",
    "FrontierRegion",
    "FrontierZone",
    "HumanitarianPopulation",
    "InheritanceBundle",
    "Institution",
    "InstitutionalLineage",
    "InternationalOrganization",
    "LaborUnion",
    "LegacyAggregate",
    "LineageClaim",
    "LocalElite",
    "MassOrganization",
    "MediaOutlet",
    "MilitaryFormation",
    "MiningSite",
    "Narrative",
    "Norm",
    "Office",
    "Party",
    "Policy",
    "Province",
    "ReligiousInstitution",
    "Resource",
    "SecurityApparatus",
    "SecurityForce",
    "Shock",
    "SocialMovement",
    "SuccessorPolity",
    "SupplyChainNode",
    "TransitCorridor",
    "TransmissionChannel",
    "TributaryRelation",
    "Contract",
    "Firm",
    "FundingProgram",
    "Infrastructure",
    "Investor",
    "Laboratory",
    "MarketActor",
    "MediaNarrative",
    "PlatformProduct",
    "ProcurementChannel",
    "Region",
    "RegulatoryBody",
    "ResearchNetwork",
    "Technology",
    "University",
    "AdMarket",
    "Advertiser",
    "AppStore",
    "ChipSupplyNode",
    "CloudInfrastructure",
    "ComputeResource",
    "DataAsset",
    "DeveloperEcosystem",
    "ExecutiveOffice",
    "LaborPool",
    "LegislativeBody",
    "ModelLab",
    "Platform",
    "PoliticalCoalition",
    "Standard",
    "UserPopulation",
    "DistributionChannel",
    "PlatformComponent",
)

_EDGE_NAMES: tuple[str, ...] = (
    "administers",
    "aligns_with",
    "allies_with",
    "appeals_to",
    "backs",
    "bargains_with",
    "benefits",
    "buffers",
    "centralizes",
    "channels_through",
    "coalitions_with",
    "commands",
    "conditions",
    "conquers",
    "constrains",
    "contests",
    "controls",
    "coopts",
    "coordinates_with",
    "decentralizes",
    "delegates_to",
    "delegitimizes",
    "depends_on",
    "disciplines",
    "displaces",
    "excludes",
    "externalizes_to",
    "extracts_from",
    "fails_to_check",
    "federates_with",
    "finances",
    "fragments",
    "funds",
    "governs",
    "grants_status_to",
    "imposes_order_on",
    "inherits_from",
    "integrates",
    "intervenes_in",
    "invades",
    "isolates",
    "leaves_legacy_via",
    "legitimizes",
    "loots",
    "mediates",
    "mobilizes",
    "monitors",
    "nationalizes",
    "negotiates_with",
    "overthrows",
    "polarizes",
    "pressures",
    "presupposes",
    "reconfigures",
    "recruits",
    "redistributes",
    "reforms",
    "regulates",
    "resists",
    "responds_to",
    "restores",
    "rolls_back",
    "sabotages",
    "sacralizes",
    "sanctions",
    "secedes_from",
    "secures",
    "settles_within",
    "smuggles_through",
    "strikes_against",
    "subordinates",
    "supersedes_while_claiming_continuity",
    "supports",
    "survives_as",
    "taxes",
    "trades_with",
    "trains",
    "transfers_to_center",
    "transforms_into",
    "transmits_through",
    "undermines",
    "usurps",
    "votes_against",
    "withdraws_support_from",
    "alignswith",
    "brandsthrough",
    "builds",
    "clustersin",
    "collaborateswith",
    "contractswith",
    "dependson",
    "diffusesinto",
    "embedsin",
    "founds",
    "integrateswith",
    "investsin",
    "marketsas",
    "procuresfrom",
    "recruitsfrom",
    "researches",
    "respondsTo",
    "scales",
    "scrutinizes",
    "spinsoutfrom",
    "standardizes",
    "captures",
    "competeswith",
    "deplatforms",
    "extractsfrom",
    "governsaccessTo",
    "hosts",
    "intermediates",
    "moderates",
    "monetizes",
    "ranks",
    "restrictsexportTo",
    "securitizes",
    "supplies",
    "tracks",
    "trainson",
    "acquires",
    "distributesthrough",
    "locksin",
)

# Optional attribute type hints for high-leverage node kinds (retrieval / adapters; open vocabulary).
_ATTR_HINTS: dict[str, dict[str, str]] = {
    "Event": {"when": "iso8601_span", "granularity": "str"},
    "Shock": {"severity_hint": "float", "recovery_horizon": "iso8601_span"},
    "Narrative": {"stance": "str", "audience": "str"},
    "MediaNarrative": {"framing": "str", "medium": "str"},
    "Institution": {"jurisdiction_hint": "str", "charter_kind": "str"},
    "Policy": {"instrument": "str", "scope": "str"},
    "Platform": {"business_model": "str", "gatekeeping": "str"},
    "Contract": {"kind": "str", "counterparty_class": "str"},
    "Firm": {"sector_hint": "str", "listing": "str"},
    "Technology": {"maturity": "str", "supply_chain_role": "str"},
    "Investor": {"vehicle": "str", "horizon": "str"},
    "Commodity": {"unit": "str", "benchmark": "str"},
    "DataAsset": {"sensitivity_tier": "str", "lineage": "str"},
    "ComputeResource": {"tier": "str", "region_hint": "str"},
    "ExternalPower": {"alignment_hint": "str", "power_projection": "str"},
    "ExternalConstraint": {"constraint_class": "str", "source_actor_class": "str"},
}

# Edge-level extensions for contested, persistence, and governance relations (no probe string changes).
_EDGE_EXTENSIONS: dict[str, dict[str, Any]] = {
    "contests": {"relational_role": "ideological_contestation"},
    "legitimizes": {"relational_role": "legitimation"},
    "delegitimizes": {"relational_role": "delegitimation"},
    "polarizes": {"relational_role": "polarization"},
    "inherits_from": {"relational_role": "persistence_lineage"},
    "transmits_through": {"relational_role": "persistence_transmission"},
    "leaves_legacy_via": {"relational_role": "persistence_legacy"},
    "regulates": {"relational_role": "institutional_governance"},
    "constrains": {"relational_role": "structural_constraint"},
    "supersedes_while_claiming_continuity": {"relational_role": "contested_succession"},
    "integrates": {"relational_role": "integration_stack"},
    "integrateswith": {"relational_role": "integration_stack"},
    "survives_as": {"relational_role": "persistence_identity_continuity"},
    "transforms_into": {"relational_role": "regime_form_reconfiguration"},
    "allies_with": {"relational_role": "alliance_commitment"},
    "supports": {"relational_role": "political_or_material_support"},
    "backs": {"relational_role": "backing_commitment"},
}


# Parallel probe tokens (snake_case vs concatenated/camel) share semantics; adapters should normalize via family id.
_EDGE_NORMALIZATION_FAMILY: dict[str, str] = {
    "aligns_with": "alignment",
    "alignswith": "alignment",
    "depends_on": "dependency",
    "dependson": "dependency",
    "responds_to": "response_channel",
    "respondsTo": "response_channel",
    "extracts_from": "extraction",
    "extractsfrom": "extraction",
    "integrates": "integration",
    "integrateswith": "integration",
    "trains": "training_relation",
    "trainson": "training_relation",
    "recruits": "recruitment_channel",
    "recruitsfrom": "recruitment_channel",
    "governsaccessTo": "governance_access",
    "restrictsexportTo": "export_control_gate",
    "marketsas": "market_framing",
    "procuresfrom": "procurement_channel",
    "collaborateswith": "collaboration_channel",
    "contractswith": "contracting_channel",
    "clustersin": "cluster_attachment",
    "diffusesinto": "diffusion_path",
    "embedsin": "embedding_path",
    "spinsoutfrom": "spinout_lineage",
    "brandsthrough": "brand_channel",
}

# Cross-layer compatibility hints for contested, governance, and persistence relations.
_EDGE_LAYER_HINTS: dict[str, tuple[list[Layer], list[Layer]]] = {
    "contests": (
        [Layer.IDEOLOGICAL, Layer.INSTITUTIONAL, Layer.COUNTERWEIGHT, Layer.EPISTEMIC],
        [Layer.IDEOLOGICAL, Layer.INSTITUTIONAL, Layer.EPISTEMIC],
    ),
    "delegitimizes": (
        [Layer.IDEOLOGICAL, Layer.INSTITUTIONAL, Layer.COUNTERWEIGHT],
        [Layer.IDEOLOGICAL, Layer.INSTITUTIONAL, Layer.EPISTEMIC],
    ),
    "polarizes": (
        [Layer.IDEOLOGICAL, Layer.INSTITUTIONAL],
        [Layer.IDEOLOGICAL, Layer.COUNTERWEIGHT],
    ),
    "legitimizes": (
        [Layer.IDEOLOGICAL, Layer.INSTITUTIONAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "regulates": (
        [Layer.INSTITUTIONAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "constrains": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.EPISTEMIC],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "inherits_from": (
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL],
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL],
    ),
    "transmits_through": (
        [Layer.PERSISTENCE],
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "leaves_legacy_via": (
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL],
        [Layer.PERSISTENCE, Layer.MATERIAL],
    ),
    "captures": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.IDEOLOGICAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.COUNTERWEIGHT],
    ),
    "supplies": (
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "tracks": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.IDEOLOGICAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL, Layer.COUNTERWEIGHT],
    ),
    "hosts": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL, Layer.COUNTERWEIGHT],
    ),
    "monetizes": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.MATERIAL, Layer.COUNTERWEIGHT, Layer.INSTITUTIONAL],
    ),
    "ranks": (
        [Layer.INSTITUTIONAL, Layer.IDEOLOGICAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL, Layer.COUNTERWEIGHT],
    ),
    "deplatforms": (
        [Layer.INSTITUTIONAL, Layer.IDEOLOGICAL],
        [Layer.INSTITUTIONAL, Layer.COUNTERWEIGHT, Layer.MATERIAL],
    ),
    "competeswith": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "governsaccessTo": (
        [Layer.INSTITUTIONAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL, Layer.COUNTERWEIGHT],
    ),
    "intermediates": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.COUNTERWEIGHT],
    ),
    "moderates": (
        [Layer.INSTITUTIONAL, Layer.IDEOLOGICAL],
        [Layer.COUNTERWEIGHT, Layer.INSTITUTIONAL],
    ),
    "securitizes": (
        [Layer.INSTITUTIONAL, Layer.IDEOLOGICAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "integrateswith": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "investsin": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "acquires": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "distributesthrough": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.MATERIAL, Layer.COUNTERWEIGHT],
    ),
    "locksin": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "restrictsexportTo": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "depends_on": (
        [Layer.MATERIAL, Layer.INSTITUTIONAL, Layer.EPISTEMIC],
        [Layer.MATERIAL, Layer.INSTITUTIONAL, Layer.EPISTEMIC],
    ),
    "responds_to": (
        [Layer.INSTITUTIONAL, Layer.IDEOLOGICAL, Layer.COUNTERWEIGHT],
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.EPISTEMIC],
    ),
    "extracts_from": (
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
        [Layer.MATERIAL, Layer.INSTITUTIONAL],
    ),
    "supersedes_while_claiming_continuity": (
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL],
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "integrates": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "integrateswith": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "survives_as": (
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "transforms_into": (
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL, Layer.MATERIAL],
        [Layer.PERSISTENCE, Layer.INSTITUTIONAL, Layer.MATERIAL],
    ),
    "allies_with": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.COUNTERWEIGHT],
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.COUNTERWEIGHT],
    ),
    "supports": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.IDEOLOGICAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.COUNTERWEIGHT],
    ),
    "backs": (
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.IDEOLOGICAL],
        [Layer.INSTITUTIONAL, Layer.MATERIAL, Layer.COUNTERWEIGHT],
    ),

}

_PERSISTENCE_LINEAGE_NODES: frozenset[str] = frozenset(
    {
        "LineageClaim",
        "InstitutionalLineage",
        "InheritanceBundle",
        "TransmissionChannel",
        "LegacyAggregate",
    }
)

_PERSPECTIVE_DIPLOMACY_EDGES: frozenset[str] = frozenset(
    {
        "negotiates_with",
        "mediates",
        "bargains_with",
        "appeals_to",
        "coalitions_with",
        "coordinates_with",
        "allies_with",
        "backs",
        "supports",
    }
)

_TEMPORAL_EDGE_DEFAULTS: frozenset[str] = frozenset(
    {
        "negotiates_with",
        "mediates",
        "bargains_with",
        "finances",
        "funds",
        "invades",
        "intervenes_in",
        "conquers",
        "sanctions",
        "trades_with",
        "nationalizes",
        "reforms",
        "allies_with",
        "backs",
        "supports",
        "pressures",
    }
)


def _temporal_profile_for_node(name: str) -> str:
    """Semantic time-scoping hint per node kind (consumable by future metrics; feeds eval temporal_rich)."""
    if name in _PERSISTENCE_LINEAGE_NODES:
        return "multi_epoch_lineage"
    if name == "Claim":
        return "proposition_validity_window"
    if name in ("Event", "Shock"):
        return "event_shock_window"
    if name in {"Narrative", "Norm", "MediaOutlet", "MediaNarrative"}:
        return "discourse_horizon"
    layer = _layer_for_node(name)
    if layer is None:
        return "unscoped"
    if layer == Layer.INSTITUTIONAL:
        return "institutional_term"
    if layer == Layer.COUNTERWEIGHT:
        return "mobilization_wave"
    return "epoch_scoped"


def _default_node_extensions(name: str) -> dict[str, Any]:
    """Minimal structured hooks so adapters can shard without duplicating primary_layer elsewhere."""
    layer = _layer_for_node(name)
    shard = layer.value if layer is not None else "unclassified"
    return {
        "retrieve_shard": shard,
        "ir_registry": "benchmark_union_v0",
        "temporal_profile": _temporal_profile_for_node(name),
    }


def _layer_for_node(name: str) -> Layer | None:
    persistence = {
        "LineageClaim",
        "InstitutionalLineage",
        "InheritanceBundle",
        "TransmissionChannel",
        "LegacyAggregate",
    }
    epistemic = {"Claim"}
    ideological = {"Narrative", "Norm", "MediaOutlet", "MediaNarrative"}
    counter = {
        "Counterweight",
        "Shock",
        "CivicMovement",
        "SocialMovement",
        "MassOrganization",
        "LaborUnion",
        "HumanitarianPopulation",
        "UserPopulation",
    }
    institutional = {
        "Institution",
        "Office",
        "Party",
        "Coalition",
        "Policy",
        "Province",
        "AdministrativeRegion",
        "ConstitutionalBody",
        "Empire",
        "SuccessorPolity",
        "ReligiousInstitution",
        "InternationalOrganization",
        "DebtRegime",
        "AidChannel",
        "DevelopmentProgram",
        "Enterprise",
        "BusinessAssociation",
        "SecurityApparatus",
        "SecurityForce",
        "RegulatoryBody",
        "LegislativeBody",
        "ExecutiveOffice",
        "PoliticalCoalition",
        "Platform",
        "PlatformProduct",
        "PlatformComponent",
        "AppStore",
        "University",
        "Laboratory",
        "ResearchNetwork",
        "ModelLab",
        "Standard",
        "Contract",
        "Firm",
        "FundingProgram",
        "Investor",
        "MarketActor",
        "ProcurementChannel",
        "DependencyRegime",
        "AdMarket",
        "Advertiser",
        "Technology",
        "DeveloperEcosystem",
        "DistributionChannel",
    }
    material = {
        "Resource",
        "MiningSite",
        "Commodity",
        "SupplyChainNode",
        "TransitCorridor",
        "ExtractiveChannel",
        "ExtractionChannel",
        "MilitaryFormation",
        "ArmedGroup",
        "FrontierPeople",
        "FrontierPolity",
        "FrontierRegion",
        "FrontierZone",
        "TributaryRelation",
        "LocalElite",
        "ExternalPower",
        "ExternalConstraint",
        "EconomicSector",
        "Actor",
        "Event",
        "Region",
        "Infrastructure",
        "ChipSupplyNode",
        "CloudInfrastructure",
        "ComputeResource",
        "DataAsset",
        "LaborPool",
    }

    if name in persistence:
        return Layer.PERSISTENCE
    if name in epistemic:
        return Layer.EPISTEMIC
    if name in ideological:
        return Layer.IDEOLOGICAL
    if name in counter:
        return Layer.COUNTERWEIGHT
    if name in institutional:
        return Layer.INSTITUTIONAL
    if name in material:
        return Layer.MATERIAL
    return None


def _contestation_epistemic() -> EpistemicSpec:
    return EpistemicSpec(tracks_contention=True, tracks_provenance=True, perspective_aware=True)


def _build_edges() -> dict[str, EdgeSpec]:
    edges: dict[str, EdgeSpec] = {}
    for e in _EDGE_NAMES:
        base = {"pack": "v0", "lineage": "probe_union"}
        ext = {**base, **_EDGE_EXTENSIONS.get(e, {})}
        fam = _EDGE_NORMALIZATION_FAMILY.get(e)
        if fam:
            ext = {
                **ext,
                "normalization_family": fam,
                "alias_registry": "iter2_alias_families_full_v1",
            }
        perspective_heavy = {
            "contests",
            "delegitimizes",
            "polarizes",
            "deplatforms",
            "monetizes",
            "supersedes_while_claiming_continuity",
        } | _PERSPECTIVE_DIPLOMACY_EDGES
        epi = _contestation_epistemic() if e in perspective_heavy else EpistemicSpec()
        lh = _EDGE_LAYER_HINTS.get(e)
        kw: dict[str, Any] = {"name": e, "epistemic": epi, "extensions": ext}
        if e in _TEMPORAL_EDGE_DEFAULTS:
            kw["temporal"] = TemporalSpec(
                node_time_scoped_default=False,
                edge_time_scoped_default=True,
                supports_slice_ids=True,
                iso8601_span=True,
            )
        if lh is not None:
            kw["allowed_source_layers"] = lh[0]
            kw["allowed_target_layers"] = lh[1]
        edges[e] = EdgeSpec(**kw)
    return edges


def get_seed_graph_schema() -> GraphSchema:
    """Baseline schema: benchmark-complete type registry, projection-ready, epistemically explicit."""

    nodes: dict[str, NodeSpec] = {}
    for n in _NODE_NAMES:
        ep = EpistemicSpec(min_distinct_perspectives_for_contested=2) if n == "Claim" else EpistemicSpec()
        if n == "Claim":
            nodes[n] = NodeSpec(
                name=n,
                primary_layer=_layer_for_node(n),
                epistemic=ep,
                extensions={
                    **_default_node_extensions(n),
                    "ir_role": "contested_atomic_fact",
                    "retrieval_axes": ["perspective", "confidence", "time"],
                    "evidence_linkage": "open_provenance_slots_v1",
                    "counterposition_hook": "explicit_dissent_cluster_v1",
                },
            )
        elif n == "Actor":
            nodes[n] = NodeSpec(
                name=n,
                primary_layer=_layer_for_node(n),
                epistemic=ep,
                attributes={"role_hint": "str", "scale": "str"},
                extensions={
                    **_default_node_extensions(n),
                    "ir_hint": "typed_attributes",
                    "material_bucket_semantics": "generic_material_actor_prefer_typed_entity_nodes",
                    "adapter_guidance": "disambiguate_via_incident_edges_and_subtyping",
                },
            )
        elif n == "SuccessorPolity":
            nodes[n] = NodeSpec(
                name=n,
                primary_layer=_layer_for_node(n),
                epistemic=ep,
                extensions={
                    **_default_node_extensions(n),
                    "succession_hook": "polity_continuity_and_supersedes_edges_v1",
                },
            )
        elif n in _ATTR_HINTS:
            nodes[n] = NodeSpec(
                name=n,
                primary_layer=_layer_for_node(n),
                epistemic=ep,
                attributes=_ATTR_HINTS[n],
                extensions={
                    **_default_node_extensions(n),
                    "ir_hint": "typed_attributes",
                },
            )
        elif n in _PERSISTENCE_LINEAGE_NODES:
            nodes[n] = NodeSpec(
                name=n,
                primary_layer=_layer_for_node(n),
                epistemic=ep,
                temporal=TemporalSpec(
                    node_time_scoped_default=True,
                    edge_time_scoped_default=True,
                    supports_slice_ids=True,
                    iso8601_span=True,
                ),
                extensions={
                    **_default_node_extensions(n),
                    "temporal_profile": "multi_epoch_lineage",
                    "slice_aware": True,
                },
            )
        elif n in ("ExtractionChannel", "ExtractiveChannel"):
            nodes[n] = NodeSpec(
                name=n,
                primary_layer=_layer_for_node(n),
                epistemic=ep,
                extensions={
                    **_default_node_extensions(n),
                    "dual_registry_equivalent": True,
                    "unified_semantic_role": "material_extraction_pathway",
                },
            )
        else:
            nodes[n] = NodeSpec(
                name=n,
                primary_layer=_layer_for_node(n),
                epistemic=ep,
                extensions=_default_node_extensions(n),
            )

    edges = _build_edges()

    return GraphSchema(
        name="psychohistory_graph_ir_seed",
        version="0.1.10",
        benchmark_pack="v0",
        ontology_compatibility="v1-patched",
        node_types=nodes,
        edge_types=edges,
        projection=ProjectionSpec(
            dim=32,
            required=True,
            per_type_optional={
                "Claim": 8,
                "Narrative": 8,
                "Norm": 8,
                "MediaNarrative": 8,
                "Institution": 8,
                "Policy": 8,
                "Office": 8,
                "Empire": 8,
                "SuccessorPolity": 8,
                "Contract": 8,
                "Firm": 8,
                "Technology": 8,
                "Platform": 8,
                "ExternalPower": 6,
                "ExternalConstraint": 6,
            },
        ),
        persistence_hooks_enabled=True,
        global_epistemic=EpistemicSpec(min_distinct_perspectives_for_contested=2),
        layers_declared=[
            Layer.MATERIAL,
            Layer.INSTITUTIONAL,
            Layer.IDEOLOGICAL,
            Layer.COUNTERWEIGHT,
            Layer.EPISTEMIC,
            Layer.PERSISTENCE,
        ],
        extensions={
            "notes": "Union of probe must_represent/may_represent through benchmark pack; extend via proposals.",
            "layering_revision": "gov_platform_rd_institutional_claim_head_v1",
            "cross_layer_edge_hints": "v2_platform_supply_governance_expanded",
            "persistence_temporal_profile": "multi_epoch_lineage_v1",
            "registry_enrichment": "iter0_node_edge_shard_and_projection_heads_v1",
            "temporal_profile_coverage": "iter0_all_nodes_via_default_extensions_v1",
            "layer_discipline": "iter1_no_implicit_material_fallback_for_platform_supply_contract_v1",
            "edge_alias_normalization": "iter2_snake_camel_pairs_v1",
            "extraction_channel_unification": "iter2_dual_node_tag_v1",
            "unknown_node_layer_policy": "iter2_explicit_buckets_only_return_none_outside_union_v1",
            "probe_yaml_dual_key_policy": "merge_schema_requirements_and_schemarequirements_v1",
            "iteration2_bundle": "projection_towers_external_geopolitics_alias_completion_diplomatic_temporal_v1",
            "relational_edge_extensions": "contestation_persistence_governance_succession_v2",
        },
    )


GRAPH_SCHEMA = get_seed_graph_schema()
