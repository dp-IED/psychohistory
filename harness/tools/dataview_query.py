"""Dataview query CLI — callable by any agent via `python -m harness.tools.dataview_query`.

Usage:
    python -m harness.tools.dataview_query --vault /path/to/vault --category crypto --horizon 30
    python -m harness.tools.dataview_query --vault /path/to/vault --query "TABLE question, brier FROM runs WHERE category = 'crypto' SORT brier DESC LIMIT 5"

The --category + --horizon flags run the standard queries from _strategy.md.
The --query flag runs an arbitrary Dataview-DQL query against the vault.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness.tools.dataview_runtime import run_dataview_query


def _build_cmd(args: argparse.Namespace) -> str:
    if args.query:
        return args.query

    category = args.category or "general"
    horizon = args.horizon or 30
    query_type = args.type or "all"

    queries: list[str] = []

    if query_type in ("all", "recent"):
        queries.append(
            f'TABLE question, brier, p_yes, horizon_days, run_id\n'
            f'FROM "runs"\n'
            f'WHERE category = "{category}"\n'
            f'SORT brier DESC\n'
            f'LIMIT {args.limit or 10}'
        )

    if query_type in ("all", "short"):
        queries.append(
            f'TABLE question, brier, p_yes, horizon_days\n'
            f'FROM "runs"\n'
            f'WHERE category = "{category}" AND horizon_days < {horizon}\n'
            f'SORT brier DESC\n'
            f'LIMIT {args.limit or 6}'
        )

    if query_type in ("all", "long"):
        queries.append(
            f'TABLE question, brier, p_yes, horizon_days\n'
            f'FROM "runs"\n'
            f'WHERE category = "{category}" AND horizon_days >= {horizon}\n'
            f'SORT brier DESC\n'
            f'LIMIT {args.limit or 6}'
        )

    if not queries:
        query_type = "recent"
        queries.append(
            f'TABLE question, brier, p_yes, horizon_days, run_id\n'
            f'FROM "runs"\n'
            f'WHERE category = "{category}"\n'
            f'SORT brier DESC\n'
            f'LIMIT {args.limit or 10}'
        )

    # Return all queries joined — the runtime runs them sequentially
    return "\n--- NEXT ---\n".join(queries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query vault runs using Dataview-DQL syntax."
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path("vault"),
        help="Path to vault directory (default: ./vault)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Market family / category to filter by",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Horizon threshold in days for short/long split",
    )
    parser.add_argument(
        "--type",
        choices=("all", "recent", "short", "long"),
        default="all",
        help="Which standard queries to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max results per query",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Raw Dataview-DQL query (overrides --category/--horizon/--type)",
    )

    cli_args = parser.parse_args(argv)
    vault_root = cli_args.vault.expanduser().resolve()

    if not vault_root.is_dir():
        print(f"Error: vault directory not found: {vault_root}", file=sys.stderr)
        return 1

    query_text = _build_cmd(cli_args)

    # Split on the delimiter and run each query
    sub_queries = query_text.split("\n--- NEXT ---\n")
    all_results: list[str] = []

    for idx, q in enumerate(sub_queries, start=1):
        q = q.strip()
        if not q:
            continue
        if len(sub_queries) > 1:
            all_results.append(f"### Query {idx}")
        try:
            result = run_dataview_query(vault_root, q)
        except (ValueError, OSError) as exc:
            result = f"Error: {exc}\n"
        all_results.append(result)

    print("\n".join(all_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
