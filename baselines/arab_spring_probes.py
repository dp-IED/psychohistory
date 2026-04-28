"""Arab Spring probe corpus v0: hand-authored seeds + templated expansion (no LLM).

Probe rows align with ``schemas.graph_builder_probe.ProbeRecord`` for graph-builder
plumbing tests covering Tunisia, Egypt, and Libya propagation dynamics (2010–2013).

**Syria policy (Option A)**: Syria-specific probes are explicitly excluded from both
seeds and expansion. The corpus covers only Tunisia (TU), Egypt (EG), and Libya (LY)
geography/hints. No Syrian admin1 codes, no Syrian actors as primary entity_hints.
Rationale: Syria appears in GDELT for this context but not in ACLED; Syrian nodes risk
being retrieval-valid yet label-orphaned for ACLED-grounded Stage 2. Excluding Syria
keeps Tunisia→Egypt→Libya training aligned with corroborated tape until labels cover Syria.

``generation_meta.assumption_gate_coverage`` is populated on every emitted row;
the deprecated fallback to ``assumption_emphasis`` is never used here.
"""

from __future__ import annotations

from datetime import date
from itertools import product
from typing import Sequence

from baselines.france_plumbing_probes import validate_gate_coverage
from baselines.graph_builder_query_encoder import ENTITY_HINT_KEYS, normalize_hint
from schemas import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)
from schemas.graph_builder_warehouse import NodeWarehouseManifest

ARAB_SPRING_TRAINING_CONTEXT_ID = "arab_spring_primary"

_V0_ORIGIN = date(2011, 2, 1)
_GENERATOR_VERSION = "arab_spring_v0"
_GLOBAL_SEED = 137
_TEMPLATE_ID = "arab_spring_actor_state_likelihood_v0"

_GATES_ORDERED: tuple[AssumptionEmphasis, ...] = (
    AssumptionEmphasis.PERSISTENCE,
    AssumptionEmphasis.PROPAGATION,
    AssumptionEmphasis.PRECURSOR,
    AssumptionEmphasis.SUPPRESSION,
    AssumptionEmphasis.COORDINATION,
)

# --- Expansion axes (Cartesian product → >= 150 rows) ---

# Axis 1: Geography along propagation arc (Tunisia→Egypt→Libya; Syria excluded)
ARAB_SPRING_GEOGRAPHIES: tuple[str, ...] = ("Tunisia", "Egypt", "Libya")

# Axis 2: Temporal buckets mapping to concrete as_of dates within 2010-01-01..2013-12-31
# Keys are bucket labels; values are (as_of date, context_snippet)
_TEMPORAL_BUCKETS: tuple[tuple[str, date, str], ...] = (
    ("pre_rupture", date(2010, 12, 15), "pre-uprising regional unrest signals"),
    ("during", date(2011, 6, 15), "peak mobilisation and regime response"),
    ("post", date(2013, 6, 15), "transition consolidation or fragmentation"),
)
ARAB_SPRING_TEMPORAL_BUCKET_LABELS: tuple[str, ...] = tuple(b[0] for b in _TEMPORAL_BUCKETS)
ARAB_SPRING_TEMPORAL_BUCKET_DATES: tuple[date, ...] = tuple(b[1] for b in _TEMPORAL_BUCKETS)

# Axis 3: Granularity (country-only geo vs admin1 geo vs named entity_hints pattern)
# Encoded as (granularity_label, admin1_suffix_or_none, hint_mode)
# hint_mode: "none" = no entity hints, "actor" = actor-type hint, "named" = named entity
_GRANULARITY_MODES: tuple[tuple[str, str | None, str], ...] = (
    ("country_only", None, "none"),
    ("admin1_geo", "admin1", "actor"),
    ("named_entity", None, "named"),
)
ARAB_SPRING_GRANULARITY_LABELS: tuple[str, ...] = tuple(g[0] for g in _GRANULARITY_MODES)

# Named entity hints per country (not Syria). Use strings that can appear in GDELT
# ``actor1_name`` / ``actor2_name`` (stored lowercased in ``entity_hint_keys``) — not
# synthetic slugs with underscores, or training validation fails against a real mmap.
_NAMED_HINTS_BY_COUNTRY: dict[str, list[str]] = {
    "Tunisia": ["protester", "government", "military"],
    "Egypt": ["protester", "government", "military"],
    "Libya": ["protester", "government", "military"],
}

# Actor-type hints (for admin1_geo granularity): short tokens that commonly occur in GDELT tape.
_ACTOR_TYPE_HINTS: tuple[str, ...] = ("protester", "government", "military")

# Actor types used in probes
_ACTOR_TYPES: tuple[str, ...] = ("protest_group", "government", "opposition_coalition")

# State flags
_STATE_FLAGS: tuple[str, ...] = ("escalating", "sustained")

# Horizon days for expansion
_HORIZON_DAYS: tuple[int, ...] = (7, 14, 21)


def _nl_from_slots(
    *,
    actor_type: str,
    geography: str,
    state_flag: str,
    horizon_days: int,
    context_snippet: str,
) -> str:
    actor_phrase = actor_type.replace("_", " ")
    return (
        f"What is the likelihood that {actor_phrase} in {geography} will {state_flag} "
        f"over the next {horizon_days} days, given {context_snippet}?"
    )


def _probe_record(
    *,
    probe_id: str,
    nl_text: str,
    geography: list[str],
    actor_type: list[str],
    entity_hints: list[str],
    state_flags: list[str],
    horizon_days: int,
    context_snippet: str,
    as_of: date,
    assumption_emphasis: AssumptionEmphasis,
    template_id: str,
    seed: int | None,
) -> ProbeRecord:
    return ProbeRecord(
        probe_id=probe_id,
        origin=as_of,
        nl_text=nl_text,
        q_struct=QStructV0(
            actor_state=ActorStateQuery(
                geography=geography,
                actor_type=actor_type,
                entity_hints=entity_hints,
                state_flags=state_flags,
                as_of=as_of,
            ),
        ),
        lens_params=LensParamsV0(horizon_days=horizon_days, context_snippet=context_snippet),
        assumption_emphasis=assumption_emphasis,
        generation_meta=GenerationMeta(
            template_id=template_id,
            generator_version=_GENERATOR_VERSION,
            seed=seed,
            assumption_gate_coverage=assumption_emphasis,
        ),
    )


# Twenty hand-authored base seeds (4 per gate, covering Tunisia/Egypt/Libya only).
ARAB_SPRING_BASE_PROBE_DEFS: tuple[ProbeRecord, ...] = (
    # PERSISTENCE (×4)
    _probe_record(
        probe_id="ar_base_00",
        nl_text="Base seed: Tunisian protester wage grievance persistence post-Ben Ali.",
        geography=["Tunisia"],
        actor_type=["protest_group"],
        entity_hints=["protester"],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="Protester strike calendar after Ben Ali exit",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_01",
        nl_text="Base seed: Egyptian military maintaining order continuity.",
        geography=["Egypt"],
        actor_type=["government"],
        entity_hints=["military"],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="Military transitional decree cycle",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_02",
        nl_text="Base seed: Libyan government authority consolidation in western towns.",
        geography=["Libya"],
        actor_type=["government"],
        entity_hints=["government"],
        state_flags=["sustained"],
        horizon_days=21,
        context_snippet="Government municipal integration pressure",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_03",
        nl_text="Base seed: Tunisian government party institutional persistence after elections.",
        geography=["Tunisia"],
        actor_type=["opposition_coalition"],
        entity_hints=["government"],
        state_flags=["sustained"],
        horizon_days=7,
        context_snippet="Coalition negotiation continuity",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    # PROPAGATION (×4)
    _probe_record(
        probe_id="ar_base_04",
        nl_text="Base seed: protest wave propagating from Tunis to Egyptian urban centres.",
        geography=["Egypt"],
        actor_type=["protest_group"],
        entity_hints=[],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="cross-border social media cascade",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_05",
        nl_text="Base seed: Libyan rebel network echoing Egyptian Tahrir mobilisation frames.",
        geography=["Libya"],
        actor_type=["protest_group"],
        entity_hints=["rebel"],
        state_flags=["escalating"],
        horizon_days=14,
        context_snippet="Rebel communiqué referencing Cairo playbook",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_06",
        nl_text="Base seed: Tunisian opposition narrative spillover to Libyan civil society.",
        geography=["Libya"],
        actor_type=["opposition_coalition"],
        entity_hints=[],
        state_flags=["sustained"],
        horizon_days=21,
        context_snippet="Al Jazeera broadcast amplification",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_07",
        nl_text="Base seed: Egyptian Brotherhood framing propagating to Libyan Islamist factions.",
        geography=["Libya"],
        actor_type=["opposition_coalition"],
        entity_hints=["Muslim Brotherhood"],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="Brotherhood branch coordination signal",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    # PRECURSOR (×4)
    _probe_record(
        probe_id="ar_base_08",
        nl_text="Base seed: early Tunisian bread-price signals before Sidi Bouzid.",
        geography=["Tunisia"],
        actor_type=["protest_group"],
        entity_hints=[],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="food price spike in interior towns",
        as_of=date(2010, 12, 15),
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_09",
        nl_text="Base seed: Egyptian labour strikes as leading indicator before Tahrir.",
        geography=["Egypt"],
        actor_type=["protest_group"],
        entity_hints=[],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="Mahalla textile strike wave",
        as_of=date(2010, 12, 15),
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_10",
        nl_text="Base seed: Libyan civil society online assembly before February protests.",
        geography=["Libya"],
        actor_type=["protest_group"],
        entity_hints=[],
        state_flags=["escalating"],
        horizon_days=21,
        context_snippet="Facebook group membership acceleration",
        as_of=date(2010, 12, 15),
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_11",
        nl_text="Base seed: Tunisian protester network as precursor signal.",
        geography=["Tunisia"],
        actor_type=["opposition_coalition"],
        entity_hints=["protester"],
        state_flags=["sustained"],
        horizon_days=7,
        context_snippet="Protester activity velocity spike",
        as_of=date(2010, 12, 15),
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    # SUPPRESSION (×4)
    _probe_record(
        probe_id="ar_base_12",
        nl_text="Base seed: Tunisian security apparatus dampening street action.",
        geography=["Tunisia"],
        actor_type=["government"],
        entity_hints=["government"],
        state_flags=["sustained"],
        horizon_days=7,
        context_snippet="Government security deployment orders",
        as_of=date(2010, 12, 15),
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_13",
        nl_text="Base seed: Egyptian military capacity constraining Tahrir intensity.",
        geography=["Egypt"],
        actor_type=["government"],
        entity_hints=["military"],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="Military deployment schedule",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_14",
        nl_text="Base seed: Libyan military suppression in eastern Libya.",
        geography=["Libya"],
        actor_type=["government"],
        entity_hints=["military"],
        state_flags=["escalating"],
        horizon_days=21,
        context_snippet="Military deployment reports",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_15",
        nl_text="Base seed: Tunisian curfew and internet blackout dampening coordination.",
        geography=["Tunisia"],
        actor_type=["government"],
        entity_hints=[],
        state_flags=["sustained"],
        horizon_days=7,
        context_snippet="nationwide curfew enforcement period",
        as_of=date(2010, 12, 15),
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    # COORDINATION (×4)
    _probe_record(
        probe_id="ar_base_16",
        nl_text="Base seed: Tunisian protester-government coalition joint action timing.",
        geography=["Tunisia"],
        actor_type=["protest_group"],
        entity_hints=["protester", "government"],
        state_flags=["escalating"],
        horizon_days=14,
        context_snippet="Inter-faction demonstration route planning",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_17",
        nl_text="Base seed: Egyptian multi-faction coordination at Tahrir Square.",
        geography=["Egypt"],
        actor_type=["protest_group"],
        entity_hints=["muslim brotherhood"],
        state_flags=["escalating"],
        horizon_days=21,
        context_snippet="joint communiqué issuance",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_18",
        nl_text="Base seed: Libyan government and rebel tactical coordination across fronts.",
        geography=["Libya"],
        actor_type=["opposition_coalition"],
        entity_hints=["government", "rebel"],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="Military council front-line coordination",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="ar_base_19",
        nl_text="Base seed: cross-country Arab Spring activist network synchronisation.",
        geography=["Egypt"],
        actor_type=["opposition_coalition"],
        entity_hints=[],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="solidarity tweet timing pattern",
        as_of=_V0_ORIGIN,
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="arab_spring_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
)


def _entity_hints_for_expanded_row(
    expansion_index: int,
    country: str,
    granularity_label: str,
) -> list[str]:
    """Build entity hints based on granularity mode and country."""
    granularity_entry = _GRANULARITY_MODES[expansion_index % len(_GRANULARITY_MODES)]
    hint_mode = granularity_entry[2]
    if hint_mode == "none":
        return []
    if hint_mode == "actor":
        idx = expansion_index % len(_ACTOR_TYPE_HINTS)
        return [_ACTOR_TYPE_HINTS[idx]]
    # named
    hints = _NAMED_HINTS_BY_COUNTRY.get(country, [])
    if not hints:
        return []
    return [hints[expansion_index % len(hints)]]


def _expanded_probe(
    country: str,
    temporal_label: str,
    temporal_date: date,
    granularity_label: str,
    granularity_admin1: str | None,
    context_snippet: str,
    expansion_index: int,
) -> ProbeRecord:
    actor_type = _ACTOR_TYPES[expansion_index % len(_ACTOR_TYPES)]
    state_flag = _STATE_FLAGS[expansion_index % len(_STATE_FLAGS)]
    horizon_days = _HORIZON_DAYS[expansion_index % len(_HORIZON_DAYS)]
    entity_hints = _entity_hints_for_expanded_row(expansion_index, country, granularity_label)

    if granularity_admin1 is not None:
        geo_label = f"{country}-{granularity_admin1}{expansion_index % 3}"
    else:
        geo_label = country

    geography = [geo_label]
    nl = _nl_from_slots(
        actor_type=actor_type,
        geography=geo_label,
        state_flag=state_flag,
        horizon_days=horizon_days,
        context_snippet=context_snippet,
    )
    # Round-robin over five gates: N%5==0..4 cycles; when N is not a multiple of 5,
    # the first (N mod 5) gates get one extra row each (e.g. 243 rows → 49+49+49+48+48).
    emphasis = _GATES_ORDERED[expansion_index % len(_GATES_ORDERED)]
    probe_id = (
        f"ar_v0_{expansion_index:04d}_"
        f"{country.lower()}_{temporal_label}_{granularity_label}_h{horizon_days}"
    )
    return _probe_record(
        probe_id=probe_id,
        nl_text=nl,
        geography=geography,
        actor_type=[actor_type],
        entity_hints=entity_hints,
        state_flags=[state_flag],
        horizon_days=horizon_days,
        context_snippet=context_snippet,
        as_of=temporal_date,
        assumption_emphasis=emphasis,
        template_id=_TEMPLATE_ID,
        seed=_GLOBAL_SEED + expansion_index,
    )


def build_arab_spring_probe_corpus() -> list[ProbeRecord]:
    """Return the templated corpus (>= 150 rows).

    Axes: Geography (3) × Temporal bucket (3) × Granularity (3) × Actor type (3) × Horizon days (3) = 243 rows.
    The three primary axes (geography, temporal, granularity) are the structural dimensions;
    actor_type and horizon_days provide the cross-cutting signal variation.
    ``AssumptionEmphasis`` is assigned by round-robin over expansion order (``expansion_index % 5``),
    not by the Cartesian slot tuple, so gate counts differ by at most one per gate
    (243 rows → 49+49+49+48+48; three remainder slots go to the first three gates in
    ``_GATES_ORDERED``), not skewed by any single axis.
    """
    rows: list[ProbeRecord] = []
    expansion_index = 0
    combos = list(product(ARAB_SPRING_GEOGRAPHIES, _TEMPORAL_BUCKETS, _GRANULARITY_MODES, _ACTOR_TYPES, _HORIZON_DAYS))
    for country, (t_label, t_date, t_ctx), (g_label, g_admin1, _hint_mode), _actor, _horizon in combos:
        rows.append(
            _expanded_probe(
                country=country,
                temporal_label=t_label,
                temporal_date=t_date,
                granularity_label=g_label,
                granularity_admin1=g_admin1,
                context_snippet=t_ctx,
                expansion_index=expansion_index,
            )
        )
        expansion_index += 1
    return rows


def validate_arab_spring_probe_gate_annotations(probes: Sequence[ProbeRecord]) -> None:
    """Require ``generation_meta.assumption_gate_coverage`` on every row (Arab Spring corpus).

    Raises a single ``ValueError`` listing every ``probe_id`` with missing coverage so
    batch audits do not require hunting row-by-row.
    """
    missing_ids = [
        p.probe_id for p in probes if p.generation_meta.assumption_gate_coverage is None
    ]
    if missing_ids:
        listed = ", ".join(missing_ids)
        raise ValueError(
            "Arab Spring corpus requires generation_meta.assumption_gate_coverage for gate "
            f"starvation audits; missing on probe_id(s): {listed}",
        )


def manifest_entity_hint_key_set(manifest: NodeWarehouseManifest) -> set[str]:
    """All ``normalize_hint(k)`` values from ``entity_hint_keys`` across rows (set union).

    ``build_hint_index`` is one-hint-wins; for *existence* checks we only need whether a
    probe hint string appears in **any** row’s keys, which matches what retrieval can use
    when that string exists on at least one node.
    """
    s: set[str] = set()
    for row in manifest.rows or []:
        raw = row.extensions.get(ENTITY_HINT_KEYS) if row.extensions else None
        aliases: list[str] = raw if isinstance(raw, list) else []
        for k in aliases:
            if isinstance(k, str):
                s.add(normalize_hint(k))
    return s


def validate_probe_hints_against_manifest(
    probes: Sequence[ProbeRecord],
    manifest: NodeWarehouseManifest,
) -> None:
    """Validate that every entity_hint in probes appears in the manifest’s hint-key union.

    Args:
        probes: Probe records to validate.
        manifest: ``NodeWarehouseManifest`` instance; must have a ``.rows`` attribute.

    Raises:
        ValueError listing all (probe_id, hint) pairs that do not resolve, in a single error.
    """
    key_set = manifest_entity_hint_key_set(manifest)
    failures: list[str] = []
    for probe in probes:
        for hint in probe.q_struct.actor_state.entity_hints:
            nk = normalize_hint(hint)
            if nk not in key_set:
                failures.append(f"probe_id={probe.probe_id!r} hint={hint!r} (normalized: {nk!r})")
    if failures:
        listed = "; ".join(failures)
        raise ValueError(
            f"entity_hint(s) not found in NodeWarehouseManifest hint key union: {listed}",
        )


__all__ = [
    "ARAB_SPRING_BASE_PROBE_DEFS",
    "ARAB_SPRING_GEOGRAPHIES",
    "ARAB_SPRING_GRANULARITY_LABELS",
    "ARAB_SPRING_TEMPORAL_BUCKET_DATES",
    "ARAB_SPRING_TEMPORAL_BUCKET_LABELS",
    "ARAB_SPRING_TRAINING_CONTEXT_ID",
    "build_arab_spring_probe_corpus",
    "validate_arab_spring_probe_gate_annotations",
    "validate_gate_coverage",
    "validate_probe_hints_against_manifest",
]
