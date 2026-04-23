"""Binary risk labels from ACLED ``fatalities`` in the Arab Spring warehouse (ACLED ground truth)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

# Syria: exclude from labels until ACLED coverage is confirmed; distribution may differ.
_EXCLUDED_COUNTRY: frozenset[str] = frozenset({"SY"})


def _acled_fatality_value(raw: dict[str, object]) -> int:
    v = raw.get("fatalities")
    if v is None:
        return 0
    s = str(v).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def build_arab_spring_labels(
    warehouse_path: Path,
    as_of: date,
    country: str,
    *,
    threshold: int = 5,
    horizon_days: int = 7,
    data_start: date = date(2010, 1, 1),
    data_end: date = date(2013, 12, 31),
) -> dict[date, int]:
    """
    For each **query date** ``d`` from ``data_start`` through the last day with a full
    **forward** window still ending on or before ``data_end`` (i.e. ``d <= data_end - horizon_days``,
    and ``d <= as_of - horizon_days`` as needed for a fully observed forward window in ``[d+1, d+horizon_days]``):

    * Value **1** if there is at least one **ACLED** event for ``country`` with
      ``_acled_fatality_value >= threshold`` and ``d < event_date <= d + horizon_days``
      (strictly after ``d``; no events on ``d`` count toward the label, so there is
      no leakage from the query day into the **forward** block).
    * **0** otherwise.

    **GDELT** rows in the warehouse are ignored (fatalities are not used as ground truth).
    **Syria** (``"SY"``) is not supported: returns an empty dict until EDA is re-run.
    """
    p = Path(warehouse_path)
    if country.upper() in _EXCLUDED_COUNTRY:
        return {}

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb is required; install project dependencies first") from exc

    c2 = country.strip().upper()
    if len(c2) != 2:
        raise ValueError("country must be a 2-letter FIPS/ISO code (e.g. EG)")

    last_query = min(as_of, data_end) - timedelta(days=horizon_days)
    if last_query < data_start:
        return {}

    con = duckdb.connect(str(p), read_only=True)
    rows = con.execute(
        """
        SELECT event_date, raw_json
        FROM events
        WHERE source_name IN ('acled', 'acled_v3')
          AND country_code = ?
          AND event_date > ?
          AND event_date <= ?
        ORDER BY event_date
        """,
        [c2, data_start, data_end],
    ).fetchall()
    con.close()

    events: list[tuple[date, int]] = []
    for ed, rj in rows:
        if isinstance(ed, str):
            ed2 = date.fromisoformat(str(ed)[:10])
        else:
            ed2 = ed  # type: ignore[assignment]
        raw = json.loads(rj) if isinstance(rj, str) else dict(rj)
        events.append((ed2, _acled_fatality_value(raw)))

    by_day: dict[date, int] = {}
    qd = data_start
    while qd <= last_query:
        w_end = qd + timedelta(days=horizon_days)
        label = 0
        for ev_d, f in events:
            if qd < ev_d <= w_end and f >= threshold:
                label = 1
                break
        by_day[qd] = label
        qd += timedelta(days=1)
    return by_day
