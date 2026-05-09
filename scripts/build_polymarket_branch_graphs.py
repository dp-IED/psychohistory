#!/usr/bin/env python
"""Build agentic branch graph artifacts from resolved Polymarket records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.polymarket_branch_builder import load_resolved_records_json, write_graph_artifacts_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/polymarket/resolved_binary_markets.json"),
        help="JSON produced by scripts/fetch_polymarket_resolved.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/polymarket/resolved_branch_graphs.jsonl"),
        help="Output JSONL path with one graph_artifact_v1 object per market.",
    )
    parser.add_argument(
        "--as-of-time",
        default="benchmark-cutoff-placeholder",
        help="Cutoff timestamp to stamp on generated portfolio scaffolds.",
    )
    args = parser.parse_args(argv)

    records = load_resolved_records_json(args.input)
    count = write_graph_artifacts_jsonl(records, args.output, as_of_time=args.as_of_time)
    print(f"wrote {count} branch graph artifacts")
    print(f"jsonl: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
