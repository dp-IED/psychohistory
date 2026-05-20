#!/usr/bin/env python3
"""PIT probes: graph forecast p_yes vs Polymarket YES price at cutoff.

Geopolitics / meta / institutions focus — not crypto-Fed-sports PM churn.

Usage:
  python scripts/pit_market_calibration.py seed --from-gold
  python scripts/pit_market_calibration.py catalog
  python scripts/pit_market_calibration.py run --max-probes 3 --skip-existing
  python scripts/pit_market_calibration.py score
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import VAULT_DIR
from harness.pit_market_probe import (
    DEFAULT_ABLATED_RESULTS,
    DEFAULT_CATALOG,
    DEFAULT_MARKET_CALIBRATION_BAND,
    DEFAULT_RESULTS,
    MarketProbeSpec,
    completed_probe_ids,
    format_ablation_comparison,
    format_market_calibration_feedback,
    load_catalog,
    load_results,
    run_market_probe,
    score_ablation,
    seed_from_gold_dataset,
    summarize_results,
    write_catalog,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "polymarket" / "gold_branch_dataset.json"

_BUILTIN_GRAPH: list[MarketProbeSpec] = [
    MarketProbeSpec(
        probe_id="graph-iran-israel-strike-jun24",
        cutoff=date(2024, 3, 31),
        question="Will Israel conduct airstrikes on Iranian territory on or before June 30, 2024?",
        graph_question=(
            "Will Israel conduct airstrikes on Iranian territory on or before June 30, 2024?\n\n"
            "Use only PIT graph through 2024-03-31. Match the probability implied by "
            "escalation dynamics in the vault (proxy war → direct exchange), not post-hoc knowledge."
        ),
        kind="vault_stance",
        domain="geopolitics",
        vault_target_p_yes=0.65,
        resolution=True,
        vault_anchors=("threads/israel-iran-shadow-war-gaza-2023-2024.md", "timeline/2024-Q1.md"),
        notes="Vault-calibrated stance; no historical PM market for this exact contract",
    ),
]


def _append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_seed(args: argparse.Namespace) -> int:
    catalog_path = Path(args.catalog)
    existing = {s.probe_id: s for s in load_catalog(catalog_path)}
    added = 0

    for spec in _BUILTIN_GRAPH:
        if spec.probe_id not in existing:
            existing[spec.probe_id] = spec
            added += 1

    if args.from_gold:
        # Map --domain argument to domain_filter
        domain_map = {
            "geopolitics": "geopolitics",
            "economics": "economics",
            "culture": "culture",
            "all": None,
        }
        domain_filter = domain_map.get(args.domain, None)
        for spec in seed_from_gold_dataset(
            GOLD_PATH,
            max_probes=args.max_gold,
            domain_filter=domain_filter,
        ):
            if spec.probe_id not in existing:
                existing[spec.probe_id] = spec
                added += 1

    specs = list(existing.values())
    specs.sort(key=lambda s: (s.cutoff.isoformat(), s.probe_id))
    write_catalog(catalog_path, specs)
    print(f"Catalog {catalog_path}: {len(specs)} probes ({added} new)")
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    specs = load_catalog(Path(args.catalog))
    if not specs:
        print("Empty catalog — run: seed --from-gold")
        return 0
    for s in specs:
        m = s.market_yes_at_cutoff
        mstr = f"{m:.3f}" if m is not None else (s.polymarket_slug or "—")
        print(f"{s.probe_id}  {s.cutoff}  [{s.domain}/{s.kind}]  market={mstr}")
        print(f"  Q: {s.question[:90]}{'…' if len(s.question) > 90 else ''}")
    return 0


def cmd_retake(args: argparse.Namespace) -> int:
    """Re-run all catalog probes (optionally after archiving prior results)."""
    results_path = Path(args.results)
    catalog_path = Path(args.catalog)
    specs = load_catalog(catalog_path)
    if not specs:
        print("Empty catalog.")
        return 1

    if results_path.is_file() and not args.no_archive:
        from datetime import datetime as _dt

        archive = results_path.with_name(
            f"results-{_dt.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        )
        results_path.rename(archive)
        print(f"Archived prior results → {archive}")

    if args.probe_ids:
        id_set = set(args.probe_ids)
        specs = [s for s in specs if s.probe_id in id_set]

    return _run_specs(args, specs, results_path)


def _run_specs(
    args: argparse.Namespace,
    todo: list[MarketProbeSpec],
    results_path: Path,
) -> int:
    print(f"Running {len(todo)} market calibration probe(s)")
    vault = Path(args.vault).resolve()
    rows: list[dict] = []

    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec.probe_id} @ {spec.cutoff}…", flush=True)
        result = run_market_probe(spec, vault_dir=vault, band=args.band)
        row = result.to_dict()
        rows.append(row)
        mae = row.get("market_abs_error")
        mstr = f" mae={mae:.3f}" if mae is not None else " (no market price)"
        print(f"  p_yes={row['p_yes']:.3f}{mstr}  within_band={row.get('within_band')}", flush=True)

    write_results(results_path, rows)
    print(f"Wrote {len(rows)} rows → {results_path}")
    summary = summarize_results(rows, band=args.band)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    catalog_path = Path(args.catalog)
    results_path = Path(args.results)
    specs = load_catalog(catalog_path)
    if not specs:
        print("Empty catalog.")
        return 1

    skip = completed_probe_ids(results_path) if args.skip_existing else set()
    todo = [s for s in specs if s.probe_id not in skip]
    if args.domain:
        todo = [s for s in todo if s.domain in args.domain]
    if args.max_probes:
        todo = todo[: args.max_probes]

    print(f"Running {len(todo)} market calibration probe(s) (skip {len(skip)} done)")
    vault = Path(args.vault).resolve()

    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec.probe_id} @ {spec.cutoff}…", flush=True)
        result = run_market_probe(spec, vault_dir=vault, band=args.band)
        row = result.to_dict()
        _append_result(results_path, row)
        mae = row.get("market_abs_error")
        mstr = f" mae={mae:.3f}" if mae is not None else " (no market price)"
        print(f"  p_yes={row['p_yes']:.3f}{mstr}  within_band={row.get('within_band')}", flush=True)

    return 0


def cmd_score(args: argparse.Namespace) -> int:
    rows = load_results(Path(args.results))
    if not rows:
        print("No results.")
        return 0
    band = getattr(args, "band", DEFAULT_MARKET_CALIBRATION_BAND)
    summary = summarize_results(rows, band=band)
    print(json.dumps(summary, indent=2))
    print("\nLast 5:")
    for r in rows[-5:]:
        mae = r.get("market_abs_error")
        m = f" mae={mae:.3f}" if mae is not None else ""
        print(f"  {r['probe_id']} p={r['p_yes']:.3f} market={r.get('market_yes_at_cutoff')}{m}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Re-run all completed probes in no-vault (ablated) mode."""
    catalog_path = Path(args.catalog)
    results_path = Path(args.results)
    ablated_path = Path(args.ablated_results)

    specs = load_catalog(catalog_path)
    if not specs:
        print("Empty catalog.")
        return 1

    # Only re-run probes that have vault-augmented results
    vault_rows = {r["probe_id"] for r in load_results(results_path) if r.get("probe_id")}
    if not vault_rows:
        print(f"No vault-augmented results found in {results_path}. Run probes first with `run`.")
        return 1

    # Check which ablated results we already have
    skip_ids = completed_probe_ids(ablated_path) if args.skip_existing else set()
    todo = [s for s in specs if s.probe_id in vault_rows and s.probe_id not in skip_ids]

    if args.domain:
        todo = [s for s in todo if s.domain in args.domain]
    if args.max_probes:
        todo = todo[: args.max_probes]

    print(f"Ablating {len(todo)} probe(s) (no-vault mode)...")
    vault = Path(args.vault).resolve()

    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec.probe_id} @ {spec.cutoff} (no-vault)...", flush=True)
        result = run_market_probe(spec, vault_dir=vault, band=args.band, no_vault=True)
        row = result.to_dict()
        _append_result(ablated_path, row)
        mae = row.get("market_abs_error")
        mstr = f" mae={mae:.3f}" if mae is not None else " (no market price)"
        print(f"  p_yes={row['p_yes']:.3f}{mstr}  within_band={row.get('within_band')}", flush=True)

    print(f"Appended {len(todo)} ablated probe(s) → {ablated_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare vault-augmented vs vault-ablated results."""
    results_path = Path(args.results)
    ablated_path = Path(args.ablated_results)

    vault_rows = load_results(results_path)
    ablated_rows = load_results(ablated_path)

    if not vault_rows:
        print(f"No vault-augmented results found in {results_path}.")
        return 1
    if not ablated_rows:
        print(f"No ablated results found in {ablated_path}. Run `ablate` first.")
        return 1

    band = getattr(args, "band", DEFAULT_MARKET_CALIBRATION_BAND)
    ablation = score_ablation(vault_rows, ablated_rows, band=band)
    print(format_ablation_comparison(ablation))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PIT vs Polymarket price calibration probes.")
    parser.add_argument("--vault", default=str(VAULT_DIR))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--ablated-results", default=str(DEFAULT_ABLATED_RESULTS))
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="Build catalog from builtins + optional gold PM")
    p_seed.add_argument("--from-gold", action="store_true")
    p_seed.add_argument("--max-gold", type=int, default=30)
    p_seed.add_argument(
        "--domain", choices=["geopolitics", "economics", "culture", "all"],
        default="geopolitics",
        help="Domain filter for gold seeding (default: geopolitics; use 'all' for no filter)",
    )

    sub.add_parser("catalog", help="List catalog probes")

    p_run = sub.add_parser("run", help="Run graph-only PIT forecasts vs market price")
    p_run.add_argument("--max-probes", type=int, default=0, help="0 = all pending")
    p_run.add_argument("--skip-existing", action="store_true")
    p_run.add_argument(
        "--domain", nargs="+", default=None,
        choices=["geopolitics", "economics", "culture", "other"],
        help="Filter probes by domain(s) to run (e.g. --domain economics)",
    )
    p_run.add_argument(
        "--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND,
        help="± band for within_band (default 0.05 = 5pt)",
    )

    p_retake = sub.add_parser("retake", help="Re-run probes (archives prior results by default)")
    p_retake.add_argument("--all", action="store_true", help="Retake full catalog (default)")
    p_retake.add_argument("--probe-ids", nargs="+", default=None)
    p_retake.add_argument("--no-archive", action="store_true")
    p_retake.add_argument("--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND)

    p_reflect = sub.add_parser(
        "reflect",
        help="After calibration: pit_reflect on librarian + forecaster misses",
    )
    p_reflect.add_argument("--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND)
    p_reflect.add_argument("--dry-run", action="store_true")

    p_learn = sub.add_parser("learn", help="Alias for reflect")
    p_learn.add_argument("--dry-run", action="store_true")

    p_score = sub.add_parser("score", help="Aggregate calibration metrics")
    p_score.add_argument("--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND)

    p_ablate = sub.add_parser("ablate", help="Re-run completed probes without vault (ablated)")
    p_ablate.add_argument("--max-probes", type=int, default=0, help="0 = all pending")
    p_ablate.add_argument("--skip-existing", action="store_true", help="Skip probes already in ablated_results.jsonl")
    p_ablate.add_argument("--domain", nargs="+", default=None)
    p_ablate.add_argument(
        "--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND,
        help="± band for within_band (default 0.05 = 5pt)",
    )

    p_compare = sub.add_parser("compare", help="Compare vault-augmented vs vault-ablated results")
    p_compare.add_argument(
        "--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND,
    )

    p_cal = sub.add_parser(
        "calibrate",
        help="retake (librarian+forecaster) → score → reflect",
    )
    p_cal.add_argument("--no-archive", action="store_true")
    p_cal.add_argument("--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND)
    p_cal.add_argument("--skip-reflect", action="store_true")

    args = parser.parse_args()
    if args.command == "seed":
        return cmd_seed(args)
    if args.command == "catalog":
        return cmd_catalog(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "retake":
        return cmd_retake(args)
    if args.command in ("reflect", "learn"):
        return cmd_reflect(args)
    if args.command == "score":
        return cmd_score(args)
    if args.command == "ablate":
        return cmd_ablate(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    return 1


def cmd_reflect(args: argparse.Namespace) -> int:
    import subprocess

    results_path = Path(args.results)
    band = getattr(args, "band", DEFAULT_MARKET_CALIBRATION_BAND)
    feedback = format_market_calibration_feedback(results_path, band=band, for_reflect=True)
    print(feedback)
    feedback_path = results_path.parent / "last_calibration_feedback.txt"
    feedback_path.write_text(feedback, encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pit_reflect.py"),
        "--forecast-threshold",
        "0",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def cmd_calibrate(args: argparse.Namespace) -> int:
    retake_args = argparse.Namespace(
        vault=args.vault,
        catalog=args.catalog,
        results=args.results,
        no_archive=args.no_archive,
        probe_ids=None,
        band=args.band,
    )
    rc = cmd_retake(retake_args)
    if rc != 0:
        return rc
    cmd_score(args)
    if args.skip_reflect:
        return 0
    reflect_args = argparse.Namespace(
        vault=args.vault,
        catalog=args.catalog,
        results=args.results,
        band=args.band,
        dry_run=getattr(args, "dry_run", False),
    )
    return cmd_reflect(reflect_args)


if __name__ == "__main__":
    raise SystemExit(main())
