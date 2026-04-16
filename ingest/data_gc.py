"""Report and prune generated psychohistory data under the central data root."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from ingest.paths import resolve_data_root, warehouse_path


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _relative(data_root: Path, path: Path) -> str:
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return str(path)


def report(data_root: Path) -> dict[str, Any]:
    data_root = Path(data_root)
    entries: list[dict[str, Any]] = []
    wh_path = warehouse_path(data_root)
    if wh_path.exists():
        entries.append(
            {
                "path": _relative(data_root, wh_path),
                "bytes": _path_size(wh_path),
                "category": "warehouse",
            }
        )

    for raw_path in [
        data_root / "raw",
        data_root / "gdelt" / "raw",
        data_root / "acled" / "raw",
    ]:
        if raw_path.exists():
            entries.append(
                {
                    "path": _relative(data_root, raw_path),
                    "bytes": _path_size(raw_path),
                    "category": "raw",
                }
            )

    runs_path = data_root / "runs"
    if runs_path.exists():
        for run_path in sorted(child for child in runs_path.iterdir() if child.exists()):
            entries.append(
                {
                    "path": _relative(data_root, run_path),
                    "bytes": _path_size(run_path),
                    "category": "run",
                }
            )

    snapshots_path = data_root / "snapshots"
    if snapshots_path.exists():
        entries.append(
            {
                "path": _relative(data_root, snapshots_path),
                "bytes": _path_size(snapshots_path),
                "category": "snapshots",
            }
        )

    total = _path_size(data_root)
    return {
        "data_root": str(data_root),
        "total_bytes": total,
        "paths": entries,
    }


def _is_older_than(path: Path, older_than_days: int | None, *, now: dt.datetime) -> bool:
    if older_than_days is None:
        return True
    cutoff = now.timestamp() - older_than_days * 24 * 60 * 60
    return path.stat().st_mtime < cutoff


def _candidate_paths(
    data_root: Path,
    *,
    raw: bool,
    materialized_snapshots: bool,
    smoke_runs: bool,
    audits: bool,
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if raw:
        for fragments_dir in data_root.rglob("fragments"):
            if fragments_dir.is_dir():
                candidates.append((fragments_dir, "raw"))
    if materialized_snapshots:
        runs_path = data_root / "runs"
        if runs_path.exists():
            for snapshots_dir in runs_path.rglob("snapshots"):
                if snapshots_dir.is_dir():
                    candidates.append((snapshots_dir, "snapshots"))
    if smoke_runs:
        runs_path = data_root / "runs"
        if runs_path.exists():
            for run_path in sorted(
                (path for path in runs_path.rglob("*") if path.is_dir() and "smoke" in path.name),
                key=lambda path: len(path.parts),
            ):
                if run_path.is_dir() and "smoke" in run_path.name:
                    candidates.append((run_path, "run"))
    if audits:
        for audit_path in data_root.rglob("*.audit.json"):
            if audit_path.is_file():
                candidates.append((audit_path, "audit"))
    return candidates


def prune(
    data_root: Path,
    *,
    raw: bool = False,
    materialized_snapshots: bool = False,
    smoke_runs: bool = False,
    audits: bool = False,
    older_than_days: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    data_root = Path(data_root)
    protected = {warehouse_path(data_root).resolve()}
    now = dt.datetime.now(tz=dt.timezone.utc)
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for path, category in _candidate_paths(
        data_root,
        raw=raw,
        materialized_snapshots=materialized_snapshots,
        smoke_runs=smoke_runs,
        audits=audits,
    ):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            continue
        if resolved in protected or protected.intersection(set(resolved.parents)):
            skipped.append(
                {
                    "path": _relative(data_root, path),
                    "category": category,
                    "reason": "protected",
                }
            )
            continue
        if not _is_older_than(path, older_than_days, now=now):
            skipped.append(
                {
                    "path": _relative(data_root, path),
                    "category": category,
                    "reason": "newer_than_cutoff",
                }
            )
            continue
        size = _path_size(path)
        deleted.append(
            {
                "path": _relative(data_root, path),
                "category": category,
                "bytes": size,
            }
        )
        if not dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    return {
        "data_root": str(data_root),
        "dry_run": dry_run,
        "deleted_count": len(deleted),
        "deleted_bytes": sum(item["bytes"] for item in deleted),
        "deleted": deleted,
        "skipped": skipped,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--data-root", default=None)

    prune_parser = subparsers.add_parser("prune")
    prune_parser.add_argument("--data-root", default=None)
    prune_parser.add_argument("--raw", action="store_true")
    prune_parser.add_argument("--materialized-snapshots", action="store_true")
    prune_parser.add_argument("--smoke-runs", action="store_true")
    prune_parser.add_argument("--audits", action="store_true")
    prune_parser.add_argument("--older-than-days", type=int, default=None)
    prune_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        data_root = resolve_data_root(args.data_root)
        if args.command == "report":
            print(json.dumps(report(data_root), indent=2, sort_keys=True))
            return 0
        if args.command == "prune":
            print(
                json.dumps(
                    prune(
                        data_root,
                        raw=args.raw,
                        materialized_snapshots=args.materialized_snapshots,
                        smoke_runs=args.smoke_runs,
                        audits=args.audits,
                        older_than_days=args.older_than_days,
                        dry_run=args.dry_run,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
