"""
Minimal linear training loop on tape-derived features (`next_steps.md` §2.2 step D).

Validates optimizer, loss, device, and batching before WM / GNN complexity. Not the production model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from baselines.backtest import weekly_origins
from baselines.features import FEATURE_NAMES, extract_features_for_origin
from baselines.metrics import brier_score
from baselines.recurrence import ForecastRow
from baselines.training_slice import (
    FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_END,
    FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_START,
    FRANCE_SCAFFOLD_TRAIN_ORIGIN_END,
    FRANCE_SCAFFOLD_TRAIN_ORIGIN_START,
)
from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES, build_snapshot_payload

LINEAR_SKELETON_MODEL_NAME = "linear_skeleton"


def assert_mondays(*dates: dt.date, label: str = "date") -> None:
    for d in dates:
        if d.weekday() != 0:
            raise ValueError(f"{label} must be a Monday (weekly_origins invariant): {d.isoformat()}")


def _filter_records_by_sources(
    records: list[EventTapeRecord],
    source_names: set[str] | None,
) -> list[EventTapeRecord]:
    if source_names is None:
        return records
    return [record for record in records if record.source_name in source_names]


def build_scoring_universe(
    records: list[EventTapeRecord],
    *,
    excluded_admin1: set[str],
) -> list[str]:
    return sorted(
        {
            r.admin1_code
            for r in records
            if r.admin1_code not in excluded_admin1
        }
    )


@dataclass(frozen=True)
class LocationSamples:
    """Stacked (origin, admin1) rows for one or more weekly origins."""

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    origins: tuple[dt.date, ...]
    admin1_codes: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.x.shape[0])


def collect_samples_for_origins(
    *,
    records: list[EventTapeRecord],
    origins: list[dt.date],
    scoring_universe: list[str],
    source_names: set[str] | None,
    feature_names: list[str],
    excluded_admin1: set[str],
) -> LocationSamples:
    """
    Build supervised rows from ``extract_features_for_origin`` + ``build_snapshot_payload`` targets.

    ``mask[i]`` is True iff ``target_occurs_next_7d`` is present for that (origin, admin1), matching
    :func:`baselines.gnn.build_graph_from_snapshot` semantics.
    """
    for origin in origins:
        assert_mondays(origin, label="forecast_origin")

    xs: list[list[float]] = []
    ys: list[float] = []
    masks: list[bool] = []
    row_origins: list[dt.date] = []
    row_codes: list[str] = []

    for origin in origins:
        feature_rows = extract_features_for_origin(
            records=records,
            origin_date=origin,
            scoring_universe=scoring_universe,
            source_names=source_names,
        )
        payload = build_snapshot_payload(
            records=records,
            origin_date=origin,
            source_names=source_names,
        )
        target_occurs: dict[str, bool] = {}
        target_counts: dict[str, int] = {}
        for row in payload["target_table"]:
            code = row["metadata"]["admin1_code"]
            if code in excluded_admin1:
                continue
            if row["name"] == "target_count_next_7d":
                target_counts[code] = int(row["value"])
            elif row["name"] == "target_occurs_next_7d":
                target_occurs[code] = bool(row["value"])

        for fr in feature_rows:
            code = fr.admin1_code
            if code in excluded_admin1:
                continue
            xs.append([float(fr.features[name]) for name in feature_names])
            row_origins.append(origin)
            row_codes.append(code)
            if code in target_occurs:
                ys.append(float(target_occurs[code]))
                masks.append(True)
            else:
                ys.append(0.0)
                masks.append(False)

    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)
    m_t = torch.tensor(masks, dtype=torch.bool)
    return LocationSamples(
        x=x_t,
        y=y_t,
        mask=m_t,
        origins=tuple(row_origins),
        admin1_codes=tuple(row_codes),
    )


class LinearOccurrenceModel(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


def _train_one_epoch(
    model: LinearOccurrenceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb, mb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        if mb.sum() == 0:
            continue
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits[mb], yb[mb])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    if n_batches == 0:
        print(
            "[train_loop_skeleton] warning: no training batches with masked labels this epoch.",
            file=sys.stderr,
        )
    return total_loss / max(n_batches, 1)


def _forecast_rows_from_samples(
    model: LinearOccurrenceModel,
    samples: LocationSamples,
    *,
    device: torch.device,
    model_name: str,
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
) -> list[ForecastRow]:
    model.eval()
    rows: list[ForecastRow] = []
    with torch.no_grad():
        logits = model(samples.x.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    for i in range(len(samples)):
        if not samples.mask[i].item():
            continue
        origin = samples.origins[i]
        code = samples.admin1_codes[i]
        tc, to = target_lookup.get((origin, code), (0, False))
        rows.append(
            ForecastRow(
                forecast_origin=origin,
                admin1_code=code,
                model_name=model_name,
                predicted_count=float(probs[i]),
                predicted_occurrence_probability=float(probs[i]),
                target_count_next_7d=int(tc),
                target_occurs_next_7d=bool(to),
            )
        )
    return rows


def build_target_lookup_for_origins(
    *,
    records: list[EventTapeRecord],
    origins: list[dt.date],
    source_names: set[str] | None,
    excluded_admin1: set[str],
) -> dict[tuple[dt.date, str], tuple[int, bool]]:
    lookup: dict[tuple[dt.date, str], tuple[int, bool]] = {}
    for origin in origins:
        payload = build_snapshot_payload(
            records=records,
            origin_date=origin,
            source_names=source_names,
        )
        for row in payload["target_table"]:
            code = row["metadata"]["admin1_code"]
            if code in excluded_admin1:
                continue
            key = (origin, code)
            existing = lookup.get(key, (0, False))
            if row["name"] == "target_count_next_7d":
                lookup[key] = (int(row["value"]), existing[1])
            elif row["name"] == "target_occurs_next_7d":
                lookup[key] = (existing[0], bool(row["value"]))
    return lookup


def run_linear_skeleton_cli(
    *,
    tape_path: Path,
    train_origin_start: dt.date,
    train_origin_end: dt.date,
    holdout_origin_start: dt.date,
    holdout_origin_end: dt.date,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    source_names: set[str] | None,
    feature_names: list[str],
    excluded_admin1: set[str],
) -> dict[str, Any]:
    assert_mondays(
        train_origin_start,
        train_origin_end,
        holdout_origin_start,
        holdout_origin_end,
        label="window boundary",
    )
    if train_origin_end >= holdout_origin_start:
        raise ValueError(
            "train_origin_end must be before holdout_origin_start "
            f"({train_origin_end} >= {holdout_origin_start})"
        )

    records = load_event_tape(tape_path)
    filtered = _filter_records_by_sources(records, source_names)
    scoring_universe = build_scoring_universe(filtered, excluded_admin1=excluded_admin1)

    train_origins = weekly_origins(train_origin_start, train_origin_end)
    holdout_origins = weekly_origins(holdout_origin_start, holdout_origin_end)

    train_samples = collect_samples_for_origins(
        records=filtered,
        origins=train_origins,
        scoring_universe=scoring_universe,
        source_names=source_names,
        feature_names=feature_names,
        excluded_admin1=excluded_admin1,
    )
    holdout_samples = collect_samples_for_origins(
        records=filtered,
        origins=holdout_origins,
        scoring_universe=scoring_universe,
        source_names=source_names,
        feature_names=feature_names,
        excluded_admin1=excluded_admin1,
    )

    n_train = len(train_samples)
    if batch_size >= n_train and n_train > 0:
        print(
            f"[train_loop_skeleton] warning: batch_size ({batch_size}) >= train rows ({n_train}); "
            "batching is not exercised meaningfully.",
            file=sys.stderr,
        )

    masked_y = train_samples.y[train_samples.mask]
    if masked_y.numel() > 0:
        prevalence = float(masked_y.mean().item())
        print(
            f"[train_loop_skeleton] train label prevalence (masked rows): "
            f"{prevalence:.4f} ({int(masked_y.sum().item())}/{int(train_samples.mask.sum().item())})",
            file=sys.stderr,
        )
    else:
        print(
            "[train_loop_skeleton] warning: zero masked training rows — loss may be empty.",
            file=sys.stderr,
        )

    dim = len(feature_names)
    model = LinearOccurrenceModel(dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(train_samples.x, train_samples.y, train_samples.mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    all_origins = train_origins + holdout_origins
    target_lookup = build_target_lookup_for_origins(
        records=filtered,
        origins=all_origins,
        source_names=source_names,
        excluded_admin1=excluded_admin1,
    )

    last_train_loss = float("nan")
    last_holdout_brier = float("nan")

    for epoch in range(1, epochs + 1):
        last_train_loss = _train_one_epoch(model, loader, optimizer, device)
        holdout_rows = _forecast_rows_from_samples(
            model,
            holdout_samples,
            device=device,
            model_name=LINEAR_SKELETON_MODEL_NAME,
            target_lookup=target_lookup,
        )
        if holdout_rows:
            last_holdout_brier = brier_score(holdout_rows, LINEAR_SKELETON_MODEL_NAME)
        else:
            last_holdout_brier = float("nan")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": last_train_loss,
                    "holdout_brier": last_holdout_brier,
                }
            ),
            flush=True,
        )

    return {
        "epochs": epochs,
        "last_train_loss": last_train_loss,
        "last_holdout_brier": last_holdout_brier,
        "train_rows": n_train,
        "holdout_rows": len(holdout_samples),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Linear occurrence baseline training skeleton.")
    parser.add_argument("--tape", type=Path, required=True, help="Path to event tape JSONL.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--train-origin-start", default=FRANCE_SCAFFOLD_TRAIN_ORIGIN_START.isoformat())
    parser.add_argument("--train-origin-end", default=FRANCE_SCAFFOLD_TRAIN_ORIGIN_END.isoformat())
    parser.add_argument("--holdout-origin-start", default=FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_START.isoformat())
    parser.add_argument("--holdout-origin-end", default=FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_END.isoformat())
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    run_linear_skeleton_cli(
        tape_path=args.tape,
        train_origin_start=dt.date.fromisoformat(args.train_origin_start),
        train_origin_end=dt.date.fromisoformat(args.train_origin_end),
        holdout_origin_start=dt.date.fromisoformat(args.holdout_origin_start),
        holdout_origin_end=dt.date.fromisoformat(args.holdout_origin_end),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
    )


if __name__ == "__main__":
    main()
