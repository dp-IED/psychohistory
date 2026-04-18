"""France harness v0: hand-authored probe seeds + templated expansion (no LLM).

Probe rows align with ``schemas.graph_builder_probe.ProbeRecord`` for graph-builder plumbing tests.
"""

from __future__ import annotations

from datetime import date
from itertools import product
from typing import Sequence

from schemas import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)

FRANCE_PLUMBING_TRAINING_CONTEXT_ID = "france_harness"

_V0_ORIGIN = date(2019, 6, 1)
_GENERATOR_VERSION = "france_plumbing_v0"
_GLOBAL_SEED = 42
_TEMPLATE_ID = "france_actor_state_likelihood_v0"

_GATES_ORDERED: tuple[AssumptionEmphasis, ...] = (
    AssumptionEmphasis.PERSISTENCE,
    AssumptionEmphasis.PROPAGATION,
    AssumptionEmphasis.PRECURSOR,
    AssumptionEmphasis.SUPPRESSION,
    AssumptionEmphasis.COORDINATION,
)

GEOGRAPHIES_V0: tuple[str, ...] = ("France", "Île-de-France")
ACTOR_TYPES_V0: tuple[str, ...] = ("labour_union", "government", "student_group")
STATE_FLAGS_V0: tuple[str, ...] = ("escalating", "sustained")
HORIZON_DAYS_V0: tuple[int, ...] = (7, 14, 21)
CONTEXT_SNIPPETS_V0: tuple[str, ...] = (
    "ongoing pension reform debate",
    "metro service disruptions in Paris",
    "university exam calendar pressure",
    "rural fuel tax protests echoing",
    "EU summit week in Brussels",
    "local council elections approaching",
)


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
    assumption_emphasis: AssumptionEmphasis,
    template_id: str,
    seed: int | None,
) -> ProbeRecord:
    return ProbeRecord(
        probe_id=probe_id,
        origin=_V0_ORIGIN,
        nl_text=nl_text,
        q_struct=QStructV0(
            actor_state=ActorStateQuery(
                geography=geography,
                actor_type=actor_type,
                entity_hints=entity_hints,
                state_flags=state_flags,
                as_of=_V0_ORIGIN,
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


# Twenty hand-authored seeds (each a valid ProbeRecord). Expansion below reuses their entity-hint
# patterns on a round-robin basis while varying template slots for the ~200-row corpus.
FRANCE_BASE_PROBE_DEFS: tuple[ProbeRecord, ...] = (
    _probe_record(
        probe_id="fr_plumb_base_00",
        nl_text="Base seed: CGT-led wage pressure in France (persistence).",
        geography=["France"],
        actor_type=["labour_union"],
        entity_hints=["CGT"],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="national strike calendar",
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_01",
        nl_text="Base seed: government communications posture in Île-de-France.",
        geography=["Île-de-France"],
        actor_type=["government"],
        entity_hints=[],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="prefecture briefing cycle",
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_02",
        nl_text="Base seed: student unions before exams in France.",
        geography=["France"],
        actor_type=["student_group"],
        entity_hints=["UNEF"],
        state_flags=["escalating"],
        horizon_days=21,
        context_snippet="campus occupation rumours",
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_03",
        nl_text="Base seed: labour actions cooling after peak (suppression read).",
        geography=["France"],
        actor_type=["labour_union"],
        entity_hints=["FO"],
        state_flags=["sustained"],
        horizon_days=7,
        context_snippet="negotiation corridor open",
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_04",
        nl_text="Base seed: cross-actor coordination in Paris basin.",
        geography=["Île-de-France"],
        actor_type=["student_group"],
        entity_hints=["SUD Étudiant"],
        state_flags=["escalating"],
        horizon_days=14,
        context_snippet="shared demonstration route planning",
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_05",
        nl_text="Base seed: ministry staffing continuity vs shock.",
        geography=["France"],
        actor_type=["government"],
        entity_hints=["Matignon"],
        state_flags=["sustained"],
        horizon_days=21,
        context_snippet="reshuffle rumours",
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_06",
        nl_text="Base seed: union locals echoing national calls.",
        geography=["Île-de-France"],
        actor_type=["labour_union"],
        entity_hints=["CGT"],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="RATP work-to-rule pattern",
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_07",
        nl_text="Base seed: early signals from student assemblies.",
        geography=["France"],
        actor_type=["student_group"],
        entity_hints=[],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="AG attendance uptick",
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_08",
        nl_text="Base seed: riot control capacity dampening street intensity.",
        geography=["France"],
        actor_type=["government"],
        entity_hints=["CRS"],
        state_flags=["sustained"],
        horizon_days=7,
        context_snippet="banlieue deployment schedule",
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_09",
        nl_text="Base seed: federated strike timing across unions.",
        geography=["France"],
        actor_type=["labour_union"],
        entity_hints=["CFDT", "CGT"],
        state_flags=["escalating"],
        horizon_days=21,
        context_snippet="intersyndicale press conference",
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_10",
        nl_text="Base seed: institutional memory vs flash protest.",
        geography=["Île-de-France"],
        actor_type=["government"],
        entity_hints=[],
        state_flags=["escalating"],
        horizon_days=14,
        context_snippet="mayoral security orders",
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_11",
        nl_text="Base seed: narrative spillover from Lyon to Paris belt.",
        geography=["France"],
        actor_type=["student_group"],
        entity_hints=["FAGE"],
        state_flags=["sustained"],
        horizon_days=7,
        context_snippet="social media hashtag velocity",
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_12",
        nl_text="Base seed: police union statements as leading indicators.",
        geography=["Île-de-France"],
        actor_type=["labour_union"],
        entity_hints=["Alliance"],
        state_flags=["escalating"],
        horizon_days=14,
        context_snippet="Alliance communiqués",
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_13",
        nl_text="Base seed: permit denials and kettle tactics.",
        geography=["France"],
        actor_type=["government"],
        entity_hints=["Paris Police Prefecture"],
        state_flags=["sustained"],
        horizon_days=21,
        context_snippet="manifestation route restrictions",
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_14",
        nl_text="Base seed: lycée blocus coordination with unions.",
        geography=["Île-de-France"],
        actor_type=["student_group"],
        entity_hints=["FIDL"],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="lycée blockade map",
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_15",
        nl_text="Base seed: long-horizon grievance stock in industrial north.",
        geography=["France"],
        actor_type=["labour_union"],
        entity_hints=["CGT"],
        state_flags=["sustained"],
        horizon_days=21,
        context_snippet="plant closure timeline",
        assumption_emphasis=AssumptionEmphasis.PERSISTENCE,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_16",
        nl_text="Base seed: ministerial tweet cadence during crisis week.",
        geography=["France"],
        actor_type=["government"],
        entity_hints=["Interior"],
        state_flags=["escalating"],
        horizon_days=7,
        context_snippet="Twitter/X cadence spike",
        assumption_emphasis=AssumptionEmphasis.PROPAGATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_17",
        nl_text="Base seed: small campus sit-in before national call.",
        geography=["Île-de-France"],
        actor_type=["student_group"],
        entity_hints=[],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="single-site occupation",
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_18",
        nl_text="Base seed: wage policy dampening on public sector.",
        geography=["France"],
        actor_type=["government"],
        entity_hints=["Bercy"],
        state_flags=["sustained"],
        horizon_days=14,
        context_snippet="index point offer",
        assumption_emphasis=AssumptionEmphasis.SUPPRESSION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
    _probe_record(
        probe_id="fr_plumb_base_19",
        nl_text="Base seed: dockers timing with student street days.",
        geography=["France"],
        actor_type=["labour_union"],
        entity_hints=["CGT Ports"],
        state_flags=["escalating"],
        horizon_days=21,
        context_snippet="port strike overlap window",
        assumption_emphasis=AssumptionEmphasis.COORDINATION,
        template_id="france_base_seed_v0",
        seed=_GLOBAL_SEED,
    ),
)


def _entity_hints_for_expanded_row(expansion_index: int) -> list[str]:
    return list(FRANCE_BASE_PROBE_DEFS[expansion_index % 20].q_struct.actor_state.entity_hints)


def _expanded_probe(slot_row: tuple[str, str, str, int, str], expansion_index: int) -> ProbeRecord:
    geography, actor_type, state_flag, horizon_days, context_snippet = slot_row
    nl = _nl_from_slots(
        actor_type=actor_type,
        geography=geography,
        state_flag=state_flag,
        horizon_days=horizon_days,
        context_snippet=context_snippet,
    )
    emphasis = _GATES_ORDERED[expansion_index % len(_GATES_ORDERED)]
    probe_id = (
        f"fr_plumb_v0_{expansion_index:04d}_"
        f"{geography.replace(' ', '_')}_{actor_type}_{state_flag}_h{horizon_days}"
    )
    return _probe_record(
        probe_id=probe_id,
        nl_text=nl,
        geography=[geography],
        actor_type=[actor_type],
        entity_hints=_entity_hints_for_expanded_row(expansion_index),
        state_flags=[state_flag],
        horizon_days=horizon_days,
        context_snippet=context_snippet,
        assumption_emphasis=emphasis,
        template_id=_TEMPLATE_ID,
        seed=_GLOBAL_SEED + expansion_index,
    )


def build_france_plumbing_probe_corpus() -> list[ProbeRecord]:
    """Return the templated corpus (~216 rows): Cartesian slots × France v0 template."""
    combos = list(product(GEOGRAPHIES_V0, ACTOR_TYPES_V0, STATE_FLAGS_V0, HORIZON_DAYS_V0, CONTEXT_SNIPPETS_V0))
    return [_expanded_probe(row, i) for i, row in enumerate(combos)]


def validate_france_plumbing_gate_annotations(probes: Sequence[ProbeRecord]) -> None:
    """Require ``generation_meta.assumption_gate_coverage`` on every row (France harness).

    Call this when ingesting France plumbing JSONL or before Stage 1 so coverage
    can be verified from metadata alone without re-deriving from free text.
    """
    for p in probes:
        if p.generation_meta.assumption_gate_coverage is None:
            raise ValueError(
                f"probe_id={p.probe_id!r}: France plumbing requires "
                "generation_meta.assumption_gate_coverage for gate starvation audits",
            )


def validate_gate_coverage(probes: Sequence[ProbeRecord], *, training_context_id: str) -> None:
    """Ensure each ``AssumptionEmphasis`` appears at least once (soft-gate batch coverage).

    Counts use ``generation_meta.assumption_gate_coverage`` when present so audits
    and Stage-1 wiring can rely on metadata alone; ``assumption_emphasis`` is used
    only as a fallback for older rows.
    """
    if not training_context_id.strip():
        raise ValueError("training_context_id must be a non-empty string")
    counts: dict[AssumptionEmphasis, int] = {g: 0 for g in AssumptionEmphasis}
    for p in probes:
        gate = p.generation_meta.assumption_gate_coverage
        if gate is None:
            gate = p.assumption_emphasis
        counts[gate] = counts.get(gate, 0) + 1
    missing = [g.value for g in AssumptionEmphasis if counts[g] == 0]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"AssumptionEmphasis gate coverage failed for training_context_id={training_context_id!r}: "
            f"missing gate(s): {joined}",
        )


__all__ = [
    "ACTOR_TYPES_V0",
    "CONTEXT_SNIPPETS_V0",
    "FRANCE_BASE_PROBE_DEFS",
    "FRANCE_PLUMBING_TRAINING_CONTEXT_ID",
    "GEOGRAPHIES_V0",
    "HORIZON_DAYS_V0",
    "STATE_FLAGS_V0",
    "build_france_plumbing_probe_corpus",
    "validate_france_plumbing_gate_annotations",
    "validate_gate_coverage",
]
