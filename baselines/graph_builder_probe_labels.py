"""Gate-aware probe labels from event tape rows (pure Python; no torch).

``ProbeRecord.origin`` in the graph-builder schema is the forecast origin **t**
(same calendar day as evidence cutoff / weekly origin). Label windows below use
that **t** as the start of the post-origin horizon unless noted.

**Actor match (persistence / suppression):** compare ``actor1_name`` to the
target string using casefold equality, or substring containment after stripping
(casefold on both sides for containment checks).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ingest.event_records import load_event_records
from ingest.event_tape import EventTapeRecord
from ingest.paths import resolve_data_root, warehouse_path
from schemas.cameo_escalation_v0 import cameo_tier
from schemas.graph_builder_probe import AssumptionEmphasis, ProbeRecord

SUPPRESSION_BASELINE_DAYS = 30

LABEL_VERSION_V0 = "graph_builder_probe_labels_v0"

# Minimal adjacency for tests (e.g. FR11 Île-de-France ↔ FR24 Centre-Val de Loire).
FR_ADMIN1_ADJACENCY: dict[str, frozenset[str]] = {
    "FR11": frozenset({"FR24"}),
    "FR24": frozenset({"FR11"}),
}

PERSISTENCE_PRE_DAYS = 14
PERSISTENCE_HORIZON_DAYS = 7

COORDINATION_WINDOW = 3


class ProbeLabelSidecar(BaseModel):
    """Per-probe fields not carried on ``EventTapeRecord`` rows.

    ``ProbeRecord.origin`` is **t** (forecast origin); align tape slicing to that date.
    """

    probe_id: str
    target_admin1: str
    actor_a: str | None = Field(default=None, description="Primary actor (persistence target; suppression actor A).")
    actor_b: str | None = Field(default=None, description="Suppression actor B (paired with actor_a).")
    reference_event_root_code: str | None = Field(
        default=None,
        description=(
            "Probe anchor CAMEO root for Precursor: y=1 iff some horizon event has "
            "tier strictly greater than tier(this root). Use a tier-0 root (e.g. ``01``) "
            "when the probe's baseline is verbal; hand-authored sidecars should set the "
            "probe's own event class for spec-faithful escalation checks."
        ),
    )


class ProbeLabelRow(BaseModel):
    probe_id: str
    label_version: str
    gate: AssumptionEmphasis
    y: bool
    t0: str
    meta: dict[str, Any] = Field(default_factory=dict)


def _actor_matches(actor1_name: str | None, target: str | None) -> bool:
    if target is None or actor1_name is None:
        return False
    t = target.casefold().strip()
    a = actor1_name.casefold().strip()
    if not t or not a:
        return False
    if a == t:
        return True
    return t in a or a in t


def _dates_inclusive(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    d = start
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def label_persistence_y(
    records: list[EventTapeRecord],
    t: dt.date,
    *,
    target_admin1: str,
    target_actor: str | None,
    gate: AssumptionEmphasis,
) -> bool | None:
    """Positive iff target-actor daily counts in ``target_admin1`` meet persistence rule.

    **Pre-**t baseline window: ``[t-14, t-1]`` inclusive — daily mean of matched
    actor event counts. **Horizon:** ``[t, t+6]`` inclusive (7 days). **y=1** iff
    every horizon day has count **≥** that baseline mean.

    If ``target_actor`` is ``None``, returns ``False``. If ``gate`` is not
    ``PERSISTENCE``, returns ``False`` (caller should pass the active gate).

    If **no** tape evidence matches the actor in ``target_admin1`` over the pre-t
    window, ``baseline_mean`` is ``0`` — that is a **data gap**, not evidence of
    persistence. In that case returns ``None`` (unanswerable / skip), not ``True``.
    """
    if gate != AssumptionEmphasis.PERSISTENCE or target_actor is None:
        return False

    pre_start = t - dt.timedelta(days=PERSISTENCE_PRE_DAYS)
    pre_end = t - dt.timedelta(days=1)
    pre_days = _dates_inclusive(pre_start, pre_end)
    pre_counts = [0 for _ in pre_days]
    for rec in records:
        if rec.admin1_code != target_admin1:
            continue
        if not _actor_matches(rec.actor1_name, target_actor):
            continue
        if rec.event_date in pre_days:
            idx = (rec.event_date - pre_start).days
            pre_counts[idx] += 1

    baseline_mean = sum(pre_counts) / float(len(pre_counts)) if pre_counts else 0.0
    if baseline_mean == 0.0:
        return None

    horizon_start = t
    horizon_end = t + dt.timedelta(days=PERSISTENCE_HORIZON_DAYS - 1)
    horizon_days = _dates_inclusive(horizon_start, horizon_end)
    h_counts = {d: 0 for d in horizon_days}
    for rec in records:
        if rec.admin1_code != target_admin1:
            continue
        if not _actor_matches(rec.actor1_name, target_actor):
            continue
        if rec.event_date in h_counts:
            h_counts[rec.event_date] += 1

    return all(h_counts[d] >= baseline_mean for d in horizon_days)


def label_propagation_y(
    records: list[EventTapeRecord],
    t: dt.date,
    *,
    target_admin1: str,
    gate: AssumptionEmphasis,
    adjacency: dict[str, frozenset[str]] | None = None,
) -> bool:
    """Positive iff protest-class propagation to an adjacent admin1 is observed.

    Uses ``event_class`` on records for class equality. Let **F** be the
    calendar date of the **first** tape event in ``target_admin1`` with
    ``event_date`` in ``[t, t+6]``. If none, **y=0**.

    **y=1** iff there exists an event with the same ``event_class`` as an event
    on **F** in ``target_admin1``, in some admin1 **adjacent** to ``target_admin1``,
    with event date **D** such that ``1 <= (D - F).days <= 7`` and ``D <= t+6``
    (propagation stays inside the forecast week anchored at **t**).
    """
    if gate != AssumptionEmphasis.PROPAGATION:
        return False

    adj = adjacency if adjacency is not None else FR_ADMIN1_ADJACENCY
    horizon_end = t + dt.timedelta(days=PERSISTENCE_HORIZON_DAYS - 1)

    in_target = [r for r in records if r.admin1_code == target_admin1 and t <= r.event_date <= horizon_end]
    if not in_target:
        return False
    first_date = min(r.event_date for r in in_target)
    classes_on_first = {r.event_class for r in in_target if r.event_date == first_date}
    if not classes_on_first:
        return False

    neighbors = adj.get(target_admin1, frozenset())
    if not neighbors:
        return False

    for ref_class in classes_on_first:
        for r in records:
            if r.admin1_code not in neighbors:
                continue
            if r.event_class != ref_class:
                continue
            lag = (r.event_date - first_date).days
            if 1 <= lag <= 7 and r.event_date <= horizon_end:
                return True
    return False


def label_precursor_y(
    records: list[EventTapeRecord],
    t: dt.date,
    *,
    target_admin1: str,
    reference_event_root_code: str | None,
    gate: AssumptionEmphasis,
) -> bool:
    """Positive iff some same-region event in the week escalates above the reference CAMEO tier.

    Horizon ``[t, t+6]``. **y=1** iff some event with ``admin1_code == target_admin1``
    has ``cameo_tier(event_root_code) > cameo_tier(reference_event_root_code)``.
    """
    if gate != AssumptionEmphasis.PRECURSOR or reference_event_root_code is None:
        return False

    ref_tier = cameo_tier(reference_event_root_code)
    horizon_end = t + dt.timedelta(days=PERSISTENCE_HORIZON_DAYS - 1)
    for r in records:
        if r.admin1_code != target_admin1:
            continue
        if not (t <= r.event_date <= horizon_end):
            continue
        if cameo_tier(r.event_root_code) > ref_tier:
            return True
    return False


# Suppression: joint same-day presence must exceed baseline expectation by a fixed margin.
SUPPRESSION_EXPECTED_MARGIN = 1.0


def label_suppression_y(
    records: list[EventTapeRecord],
    t: dt.date,
    *,
    target_admin1: str,
    actor_a: str | None,
    actor_b: str | None,
    gate: AssumptionEmphasis,
) -> bool:
    """Positive iff simultaneous same-day co-occurrence spikes vs a 30-day pre-t baseline.

    Baseline window: ``[t-30, t-1]`` inclusive (**SUPPRESSION_BASELINE_DAYS**).
    For each day, record whether **both** actors match (via ``_actor_matches``) at
    least once in ``target_admin1``. Let ``b`` = count of such baseline days.

    Horizon ``[t, t+6]``: let ``h`` = count of days where both appear. Expected
    under stationary daily rate: ``E = 7 * (b / 30)``. **Anomaly threshold:**
    ``h >= E + SUPPRESSION_EXPECTED_MARGIN`` and ``h >= 2`` (need at least two
    joint days to call a pattern). If ``b == 0`` and ``h >= 2``, **y=1** (emergent
    co-occurrence).
    """
    if gate != AssumptionEmphasis.SUPPRESSION or actor_a is None or actor_b is None:
        return False

    base_start = t - dt.timedelta(days=SUPPRESSION_BASELINE_DAYS)
    base_end = t - dt.timedelta(days=1)
    base_days = _dates_inclusive(base_start, base_end)

    def joint_on_day(day: dt.date) -> bool:
        a_hit = False
        b_hit = False
        for rec in records:
            if rec.admin1_code != target_admin1 or rec.event_date != day:
                continue
            if _actor_matches(rec.actor1_name, actor_a):
                a_hit = True
            if _actor_matches(rec.actor1_name, actor_b):
                b_hit = True
            if a_hit and b_hit:
                return True
        return False

    b = sum(1 for d in base_days if joint_on_day(d))
    horizon_end = t + dt.timedelta(days=PERSISTENCE_HORIZON_DAYS - 1)
    h_days = _dates_inclusive(t, horizon_end)
    h = sum(1 for d in h_days if joint_on_day(d))

    expected = 7.0 * (b / float(SUPPRESSION_BASELINE_DAYS))
    if b == 0:
        return h >= 2
    return h >= 2 and h >= expected + SUPPRESSION_EXPECTED_MARGIN


def label_coordination_y(
    records: list[EventTapeRecord],
    t: dt.date,
    *,
    target_admin1: str,
    gate: AssumptionEmphasis,
) -> bool:
    """Simplified coordination: aligned count deltas for two actors in a 3-day subwindow.

    Within ``[t, t+6]``, consider each consecutive 3-day window ``(d0,d1,d2)``.
    For each unordered pair of distinct ``actor1_name`` values (non-None) with at
    least one event in ``target_admin1`` in the horizon, let ``c_a(d)`` be the
    count of events for actor **a** on day ``d``. Define ``Δ_a = c_a(d2) - c_a(d0)``.

    **y=1** iff some window and some pair have ``Δ_a * Δ_b > 0`` and both
    ``Δ_a, Δ_b`` non-zero (same sign, non-flat deltas). This is a minimal proxy
    for “same-direction moves” without a full independence test.
    """
    if gate != AssumptionEmphasis.COORDINATION:
        return False

    horizon_end = t + dt.timedelta(days=PERSISTENCE_HORIZON_DAYS - 1)
    horizon_days = _dates_inclusive(t, horizon_end)
    if len(horizon_days) < COORDINATION_WINDOW:
        return False

    def count_for(actor: str, day: dt.date) -> int:
        n = 0
        for rec in records:
            if rec.admin1_code != target_admin1 or rec.event_date != day:
                continue
            if rec.actor1_name == actor:
                n += 1
        return n

    actors_in_horizon: set[str] = set()
    for rec in records:
        if rec.admin1_code != target_admin1 or not (t <= rec.event_date <= horizon_end):
            continue
        if rec.actor1_name:
            actors_in_horizon.add(rec.actor1_name)

    actor_list = sorted(actors_in_horizon)
    for i in range(len(horizon_days) - COORDINATION_WINDOW + 1):
        d0 = horizon_days[i]
        d2 = horizon_days[i + COORDINATION_WINDOW - 1]
        for ai in range(len(actor_list)):
            for bi in range(ai + 1, len(actor_list)):
                a = actor_list[ai]
                b = actor_list[bi]
                da = count_for(a, d2) - count_for(a, d0)
                db = count_for(b, d2) - count_for(b, d0)
                if da != 0 and db != 0 and da * db > 0:
                    return True
    return False


def compute_probe_label_y(
    records: list[EventTapeRecord],
    t: dt.date,
    gate: AssumptionEmphasis,
    sidecar: ProbeLabelSidecar,
    *,
    adjacency: dict[str, frozenset[str]] | None = None,
) -> bool | None:
    """Dispatch label by ``gate`` using ``sidecar`` fields.

    Returns ``None`` only when Persistence is **unanswerable** (zero pre-t baseline).
    """
    if gate == AssumptionEmphasis.PERSISTENCE:
        return label_persistence_y(
            records,
            t,
            target_admin1=sidecar.target_admin1,
            target_actor=sidecar.actor_a,
            gate=gate,
        )
    if gate == AssumptionEmphasis.PROPAGATION:
        return label_propagation_y(records, t, target_admin1=sidecar.target_admin1, gate=gate, adjacency=adjacency)
    if gate == AssumptionEmphasis.PRECURSOR:
        return label_precursor_y(
            records,
            t,
            target_admin1=sidecar.target_admin1,
            reference_event_root_code=sidecar.reference_event_root_code,
            gate=gate,
        )
    if gate == AssumptionEmphasis.SUPPRESSION:
        return label_suppression_y(
            records,
            t,
            target_admin1=sidecar.target_admin1,
            actor_a=sidecar.actor_a,
            actor_b=sidecar.actor_b,
            gate=gate,
        )
    if gate == AssumptionEmphasis.COORDINATION:
        return label_coordination_y(records, t, target_admin1=sidecar.target_admin1, gate=gate)
    return False


def france_plumbing_sidecar_v0(probe: ProbeRecord) -> ProbeLabelSidecar:
    """Heuristic ``ProbeLabelSidecar`` for the France plumbing harness (v0).

    **Not ground truth:** maps coarse geography to ``FR11``, derives actors from
    ``entity_hints`` / ``actor_type`` so offline label stats can run without a
    hand-authored sidecar file. Replace with real probe→actor/admin1 mappings
    when training on serious regimes.

    **Precursor:** ``reference_event_root_code`` is tier **0** (``01``) so
    escalation is vs a verbal anchor — protest (tier 1) in the horizon counts as
    positive on a protest-heavy tape. Hand-authored sidecars should set the
    probe's own event-class root for spec-faithful ``tier(event) > tier(probe)``.
    """
    target_admin1 = "FR11"
    hints = [h.strip() for h in probe.q_struct.actor_state.entity_hints if h.strip()]
    actor_a = hints[0] if hints else probe.q_struct.actor_state.actor_type[0].replace("_", " ")
    actor_b = hints[1] if len(hints) > 1 else "Law Enforcement"
    return ProbeLabelSidecar(
        probe_id=probe.probe_id,
        target_admin1=target_admin1,
        actor_a=actor_a,
        actor_b=actor_b,
        reference_event_root_code="01",
    )


def compute_france_harness_label_distribution(
    records: list[EventTapeRecord],
    probes: list[ProbeRecord],
) -> tuple[Counter[tuple[str, int]], dict[str, Any]]:
    """Compute ``(gate value, y int)`` counts for each probe using :func:`france_plumbing_sidecar_v0`.

    Returns ``(counter, meta)`` where ``meta`` includes ``n_probes``, ``skipped``, and
    a warning string if any gate has zero positives. Probes with Persistence
    ``None`` (zero pre-t baseline — unanswerable) are omitted from the counter and
    counted in ``skipped_unanswerable_persistence``.
    """
    counter: Counter[tuple[str, int]] = Counter()
    skipped = 0
    skipped_unanswerable_persistence = 0
    for probe in probes:
        cov = probe.generation_meta.assumption_gate_coverage
        if cov is None:
            skipped += 1
            continue
        sidecar = france_plumbing_sidecar_v0(probe)
        y = compute_probe_label_y(
            records,
            probe.origin,
            cov,
            sidecar,
            adjacency=FR_ADMIN1_ADJACENCY,
        )
        if y is None:
            skipped_unanswerable_persistence += 1
            continue
        counter[(cov.value, int(y))] += 1

    positives_by_gate: dict[str, int] = {}
    for g in AssumptionEmphasis:
        positives_by_gate[g.value] = sum(
            n for (gk, yi), n in counter.items() if gk == g.value and yi == 1
        )
    total_by_gate: dict[str, int] = {g.value: sum(n for (gk, _), n in counter.items() if gk == g.value) for g in AssumptionEmphasis}

    warn: str | None = None
    low = [g for g, p in positives_by_gate.items() if total_by_gate.get(g, 0) > 0 and p == 0]
    if low:
        warn = f"gates with zero positives on this tape (check sidecar/heuristics): {', '.join(sorted(low))}"

    meta = {
        "n_probes": len(probes),
        "skipped_missing_coverage": skipped,
        "skipped_unanswerable_persistence": skipped_unanswerable_persistence,
        "positives_by_gate": positives_by_gate,
        "total_by_gate": total_by_gate,
        "warning": warn,
    }
    return counter, meta


def _main_france_harness_stats(*, data_root: Path | None, warehouse: Path | None) -> None:
    from baselines.france_plumbing_probes import build_france_plumbing_probe_corpus

    db = warehouse if warehouse is not None else warehouse_path(resolve_data_root(data_root))
    records = load_event_records(warehouse_db_path=db)
    probes = build_france_plumbing_probe_corpus()
    _, meta = compute_france_harness_label_distribution(records, probes)
    print(json.dumps(meta, indent=2, sort_keys=True))
    if meta.get("warning"):
        print(meta["warning"], file=sys.stderr)


def write_probe_labels_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | ProbeLabelRow]) -> None:
    """Append one JSON object per line (UTF-8). Accepts ``ProbeLabelRow`` or dicts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, ProbeLabelRow):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = [
    "COORDINATION_WINDOW",
    "FR_ADMIN1_ADJACENCY",
    "LABEL_VERSION_V0",
    "PERSISTENCE_HORIZON_DAYS",
    "PERSISTENCE_PRE_DAYS",
    "ProbeLabelRow",
    "ProbeLabelSidecar",
    "SUPPRESSION_BASELINE_DAYS",
    "SUPPRESSION_EXPECTED_MARGIN",
    "compute_france_harness_label_distribution",
    "compute_probe_label_y",
    "france_plumbing_sidecar_v0",
    "label_coordination_y",
    "label_persistence_y",
    "label_precursor_y",
    "label_propagation_y",
    "label_suppression_y",
    "write_probe_labels_jsonl",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe label utilities (offline tape queries).")
    parser.add_argument(
        "--france-harness-stats",
        action="store_true",
        help="Load DuckDB warehouse + France plumbing corpus; print gate label distribution JSON.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override PSYCHOHISTORY_DATA_ROOT / default shared data root.",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=None,
        help="Explicit path to events.duckdb (overrides --data-root).",
    )
    args = parser.parse_args()
    if args.france_harness_stats:
        _main_france_harness_stats(data_root=args.data_root, warehouse=args.warehouse)
    else:
        parser.print_help()
        sys.exit(2)
