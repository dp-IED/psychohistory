"""Past-only weekly feature histories aligned with flat location sample rows."""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass

import torch
from tqdm import tqdm

from baselines.features import FeatureRow, extract_features_for_origin
from ingest.event_tape import EventTapeRecord


def _monday_on_or_before(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def _filter_records_by_sources(
    records: list[EventTapeRecord],
    source_names: set[str] | None,
) -> list[EventTapeRecord]:
    if source_names is None:
        return records
    return [record for record in records if record.source_name in source_names]


@dataclass(frozen=True)
class LocationWeeklyHistorySamples:
    """x_seq time order: index 0 is oldest week (forecast_origin - 7*T)."""

    x_seq: torch.Tensor
    time_mask: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    origins: tuple[dt.date, ...]
    admin1_codes: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.x_seq.shape[0])


def collect_location_weekly_history(
    *,
    records: list[EventTapeRecord],
    origins: tuple[dt.date, ...],
    admin1_codes: tuple[str, ...],
    y: torch.Tensor,
    mask: torch.Tensor,
    scoring_universe: list[str],
    source_names: set[str] | None,
    feature_names: list[str],
    excluded_admin1: set[str],
    history_weeks: int,
    progress: bool = False,
    progress_label: str = "loc_weekly_history",
) -> LocationWeeklyHistorySamples:
    if history_weeks < 1:
        raise ValueError(f"history_weeks must be >= 1, got {history_weeks}")
    if len(origins) != len(admin1_codes):
        raise ValueError("origins and admin1_codes length mismatch")
    if y.shape[0] != len(origins) or mask.shape[0] != len(origins):
        raise ValueError("y/mask length mismatch with origins")

    filtered = _filter_records_by_sources(records, source_names)
    if filtered:
        data_start = _monday_on_or_before(min(r.event_date for r in filtered))
    else:
        data_start = dt.date.max

    n = len(origins)
    f = len(feature_names)
    t = history_weeks
    xs = torch.zeros((n, t, f), dtype=torch.float32)
    time_mask = torch.zeros((n, t), dtype=torch.bool)

    # ``extract_features_for_origin(hist)`` is identical for all rows sharing the same ``hist``
    # Monday; cache avoids O(n * T) full-tape rescans (was the dominant cost for GRU history).
    fr_by_hist: dict[dt.date, dict[str, FeatureRow]] = {}

    def _rows_for_hist(hist: dt.date) -> dict[str, FeatureRow]:
        cached = fr_by_hist.get(hist)
        if cached is not None:
            return cached
        rows = extract_features_for_origin(
            records=filtered,
            origin_date=hist,
            scoring_universe=scoring_universe,
            source_names=source_names,
        )
        out = {r.admin1_code: r for r in rows}
        fr_by_hist[hist] = out
        return out

    worst_case_visits = n * t
    if progress and n > 0:
        tqdm.write(
            f"[{progress_label}] past-only history: n_rows={n} history_weeks={t} "
            f"worst-case_cell_visits={worst_case_visits} (distinct history Mondays are cached; "
            "row bar shows scan progress)",
            file=sys.stderr,
        )

    row_pbar = tqdm(
        range(n),
        desc=f"{progress_label} · rows",
        unit="row",
        file=sys.stderr,
        disable=not progress or n == 0,
        leave=False,
        mininterval=0.15,
        smoothing=0.05,
    )
    for i in row_pbar:
        origin = origins[i]
        code = admin1_codes[i]
        for ti in range(t):
            d = history_weeks - ti
            hist = origin - dt.timedelta(days=7 * d)
            time_mask[i, ti] = hist >= data_start
            fr_by_code = _rows_for_hist(hist)
            fr = fr_by_code.get(code)
            if fr is None:
                continue
            missing = [name for name in feature_names if name not in fr.features]
            if missing:
                raise ValueError(
                    "feature_names contains keys not present on FeatureRow: "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            xs[i, ti] = torch.tensor(
                [float(fr.features[name]) for name in feature_names],
                dtype=torch.float32,
            )
        if progress and n > 0 and (i + 1 == n or (i + 1) % max(1, n // 25) == 0):
            row_pbar.set_postfix(uniq_hist=len(fr_by_hist))

    if progress and n > 0:
        tqdm.write(
            f"[{progress_label}] done unique_hist_mondays={len(fr_by_hist)}",
            file=sys.stderr,
        )

    return LocationWeeklyHistorySamples(
        x_seq=xs,
        time_mask=time_mask,
        y=y.detach().clone(),
        mask=mask.detach().clone(),
        origins=origins,
        admin1_codes=admin1_codes,
    )
