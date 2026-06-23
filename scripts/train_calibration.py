#!/usr/bin/env python3
"""PIT-safe calibration training harness.

Trains TagCalibration at a given information cutoff, then evaluates
on markets resolved after that cutoff.  Ensures no future leakage:
training data = markets with end_date ≤ cutoff
test data     = markets with end_date > cutoff

Usage:
    python scripts/train_calibration.py                     # train on all data
    python scripts/train_calibration.py --cutoff 2024-06-01 # train at PIT cutoff
    python scripts/train_calibration.py --cv-folds 5        # time-series CV
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
sys.path.insert(0, str(TESTBED))

from harness.tag_calibration import TagCalibration, TagCalibrationResult

DATA_PATH = TESTBED / "data" / "polymarket" / "resolved_markets.jsonl"


def parse_date(s: str) -> date | None:
    """Parse ISO date from various formats."""
    if not s:
        return None
    try:
        # "2024-11-05T12:00:00+00:00"
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        pass
    try:
        # "2024-11-05"
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def load_markets(path: Path) -> list[dict]:
    markets = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            rec["_end_date_obj"] = parse_date(rec.get("end_date", ""))
            markets.append(rec)
    return markets


def split_by_cutoff(markets: list[dict], cutoff: date) -> tuple[list[dict], list[dict]]:
    train, test = [], []
    for m in markets:
        d = m["_end_date_obj"]
        if d is None:
            continue  # can't place without date
        if d <= cutoff:
            train.append(m)
        else:
            test.append(m)
    return train, test


def train_and_eval(train: list[dict], test: list[dict],
                   prior_alpha: float = 1.0, prior_beta: float = 1.0,
                   verbose: bool = True) -> dict:
    """Train on train set, evaluate on test set."""
    cal = TagCalibration(prior_alpha=prior_alpha, prior_beta=prior_beta)
    for m in train:
        cal.update(m["tags"], m["resolution"])
    for m in test:
        cal.update(m["tags"], m["resolution"])  # not used for eval, just building

    if not test:
        return {"n_train": len(train), "n_test": 0,
                "brier": None, "log_loss": None, "cal_error": None}

    squared_errors = []
    log_losses = []
    tag_buckets = defaultdict(list)  # tag → list of (pred, actual) pairs

    for m in test:
        tags = m["tags"]
        actual = 1.0 if m["resolution"] else 0.0

        result = cal.query(tags)
        pred = result.mean

        squared_errors.append((pred - actual) ** 2)
        if pred > 0 and pred < 1:
            log_losses.append(-(actual * math.log(pred) +
                                (1 - actual) * math.log(1 - pred)))

        for tag in tags:
            tag_buckets[tag].append((pred, actual))

    brier = sum(squared_errors) / len(squared_errors)
    log_loss = sum(log_losses) / len(log_losses) if log_losses else None

    # Calibration error: difference between mean prediction and outcome rate
    mean_pred = sum(p for p, _ in
                    [(r.mean, 1.0 if m["resolution"] else 0.0)
                     for m, r in
                     ((m, cal.query(m["tags"])) for m in test)]) / len(test)
    outcome_rate = sum(1 for m in test if m["resolution"]) / len(test)
    cal_error = mean_pred - outcome_rate

    result = {
        "n_train": len(train),
        "n_test": len(test),
        "brier": round(brier, 4),
        "log_loss": round(log_loss, 4) if log_loss else None,
        "cal_error": round(cal_error, 4),
        "mean_pred": round(mean_pred, 4),
        "outcome_rate": round(outcome_rate, 4),
    }

    if verbose:
        print(f"  train={len(train):>4}  test={len(test):>4}  "
              f"brier={brier:.4f}  log_loss={log_loss:.4f}  "
              f"cal_error={cal_error:+.4f}  "
              f"pred_rate={mean_pred:.3f}  actual={outcome_rate:.3f}")

    return result


def time_series_cv(markets: list[dict], folds: int = 5) -> list[dict]:
    """Time-series cross-validation: train on earlier periods, test on later."""
    dated = sorted(
        [m for m in markets if m["_end_date_obj"] is not None],
        key=lambda m: m["_end_date_obj"],
    )
    if not dated:
        return []

    results = []
    chunk_size = len(dated) // (folds + 1)
    if chunk_size < 10:
        print("Not enough dated markets for CV")
        return []

    for i in range(1, folds + 1):
        cutoff_idx = i * chunk_size
        train = dated[:cutoff_idx]
        test = dated[cutoff_idx:cutoff_idx + chunk_size]

        cutoff_date = dated[cutoff_idx]["_end_date_obj"]
        print(f"\nFold {i}/{folds}  cutoff={cutoff_date}  "
              f"train={len(train)}  test={len(test)}")

        r = train_and_eval(train, test, verbose=True)
        r["fold"] = i
        r["cutoff"] = str(cutoff_date)
        results.append(r)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PIT-safe calibration training"
    )
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Train at PIT cutoff (YYYY-MM-DD)")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="Time-series cross-validation folds (0 = skip)")
    parser.add_argument("--full", action="store_true",
                        help="Train on all data, print full tag summary")
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run fetch_calibration_data.py first.")
        return 1

    markets = load_markets(DATA_PATH)
    dated = [m for m in markets if m["_end_date_obj"] is not None]
    print(f"Loaded {len(markets)} markets ({len(dated)} with dates)")

    # ── Full training ──────────────────────────────────────────────
    if args.full or (not args.cutoff and not args.cv_folds):
        print("\n=== Full Training (all data) ===")
        cal = TagCalibration(prior_alpha=args.prior_alpha,
                             prior_beta=args.prior_beta)
        for m in markets:
            cal.update(m["tags"], m["resolution"])
        print(cal.summary())
        print(f"\nTag coverage: {cal.n_tags} tags across "
              f"{cal.n_markets} markets")

    # ── PIT cutoff training ────────────────────────────────────────
    if args.cutoff:
        cutoff = date.fromisoformat(args.cutoff)
        print(f"\n=== PIT Training (cutoff={cutoff}) ===")
        train, test = split_by_cutoff(markets, cutoff)
        if not train:
            print(f"  No training data before {cutoff}")
        elif not test:
            print(f"  No test data after {cutoff}")
        else:
            r = train_and_eval(train, test, verbose=True)
            print(f"\n  Result: {json.dumps(r, indent=2)}")

    # ── Time-series CV ─────────────────────────────────────────────
    if args.cv_folds:
        print(f"\n=== Time-Series CV ({args.cv_folds} folds) ===")
        cv_results = time_series_cv(markets, args.cv_folds)
        if cv_results:
            avg_brier = sum(r["brier"] for r in cv_results) / len(cv_results)
            avg_cal = sum(r["cal_error"] for r in cv_results) / len(cv_results)
            print(f"\nCV Summary ({len(cv_results)} folds):")
            print(f"  Avg Brier:     {avg_brier:.4f}")
            print(f"  Avg Cal Error: {avg_cal:+.4f}")
            print(f"  (negative = underconfident, positive = overconfident)")

            # Write CV results
            cv_path = TESTBED / "data" / "polymarket" / "cv_results.json"
            cv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cv_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "folds": args.cv_folds,
                    "avg_brier": round(avg_brier, 4),
                    "avg_cal_error": round(avg_cal, 4),
                    "fold_results": cv_results,
                }, f, indent=2)
            print(f"  Results written to {cv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
