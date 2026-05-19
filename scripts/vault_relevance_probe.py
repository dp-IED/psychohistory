#!/usr/bin/env python3
"""Today-relevance probe for every vault node: fetch, explain, improve.

Verdicts:
  keep        — relevance >= floor and vault-grounded explanation OK
  expand      — relevance >= floor but content/retrieval weak → extend conjuncture
  reorganize  — relevance < floor (anecdote/subset) → merge into parent, demote, delete
  fail        — probe broken (no JSON, no score)

Usage:
  python scripts/vault_relevance_probe.py audit
  python scripts/vault_relevance_probe.py probe --types concepts --max-nodes 5 --relevance-floor 0.4
  python scripts/vault_relevance_probe.py improve
  python scripts/vault_relevance_probe.py reorganize --dry-run
  python scripts/vault_relevance_probe.py summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import VAULT_DIR
from harness.vault_probe import (
    ProbeResult,
    VaultNode,
    build_improvement_prompt,
    build_reorganize_prompt,
    call_hermes,
    enumerate_nodes,
    expand_candidates,
    load_results,
    probed_node_keys,
    probe_result_to_dict,
    reorganize_candidates,
    run_probe,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "data" / "vault_probe" / "results.jsonl"
_RESULTS_LOCK = threading.Lock()


def _append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _RESULTS_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def _rows_to_probe_results(rows: list[dict]) -> list[ProbeResult]:
    out: list[ProbeResult] = []
    for r in rows:
        nd = r["node"]
        out.append(
            ProbeResult(
                node=VaultNode(**nd),
                question=r.get("question", ""),
                today=r.get("today", date.today().isoformat()),
                passed=r.get("passed", False),
                verdict=r.get("verdict", "fail"),
                relevance_score=r.get("relevance_score"),
                relevance_floor=r.get("relevance_floor", 0.4),
                disposition=r.get("disposition", ""),
                merge_target=r.get("merge_target", ""),
                errors=r.get("errors", []),
                gaps=r.get("gaps", ""),
                explanation=r.get("explanation", ""),
            )
        )
    return out


def _followup_ns(args: argparse.Namespace, **overrides: object) -> argparse.Namespace:
    """Namespace for improve/reorganize when chained from probe (subcommands define extra fields)."""
    return argparse.Namespace(
        vault=args.vault,
        results=args.results,
        relevance_floor=args.relevance_floor,
        dry_run=False,
        **overrides,
    )


def _git_commit_vault(vault: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(vault), check=False, timeout=30)
    stat = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        capture_output=True, text=True, cwd=str(vault), timeout=30,
    )
    if stat.stdout.strip():
        subprocess.run(["git", "commit", "-m", message], cwd=str(vault), check=False, timeout=30)
        print(f"Committed: {stat.stdout.strip()}")


def cmd_audit(vault: Path, types: tuple[str, ...]) -> int:
    nodes = enumerate_nodes(vault, types=types)
    by_type: dict[str, int] = {}
    for n in nodes:
        by_type[n.node_type] = by_type.get(n.node_type, 0) + 1
    print(f"Vault nodes probeable: {len(nodes)}")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    vault = Path(args.vault).resolve()
    types = tuple(args.types)
    nodes = enumerate_nodes(vault, types=types)
    selected = nodes if args.all else nodes[: args.max_nodes]

    skip = probed_node_keys(Path(args.results)) if args.skip_existing else set()
    todo = [n for n in selected if n.rel_path not in skip]
    print(
        f"Probing {len(todo)} nodes (today={date.today().isoformat()}, "
        f"mode={args.mode}, relevance_floor={args.relevance_floor}, skip={len(skip)} done)"
    )
    if not todo:
        return 0

    counts = {"keep": 0, "expand": 0, "reorganize": 0, "fail": 0}

    def one(node):
        return run_probe(
            node, vault_dir=vault, min_chars=args.min_chars,
            relevance_floor=args.relevance_floor, mode=args.mode,
        )

    if args.concurrency <= 1:
        for i, node in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {node.rel_path}…", flush=True)
            result = one(node)
            _append_result(Path(args.results), probe_result_to_dict(result))
            counts[result.verdict] = counts.get(result.verdict, 0) + 1
            rel = f"{result.relevance_score:.2f}" if result.relevance_score is not None else "?"
            print(
                f"  {result.verdict.upper()} rel={rel} [{result.probe_mode}] — "
                f"{'; '.join(result.errors) or 'ok'}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(one, n): n for n in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                result = fut.result()
                _append_result(Path(args.results), probe_result_to_dict(result))
                counts[result.verdict] = counts.get(result.verdict, 0) + 1
                print(f"[{i}/{len(todo)}] {result.verdict.upper()} {result.node.rel_path}", flush=True)

    print(f"\nDone → {args.results}: {counts}")
    if args.improve:
        cmd_improve(_followup_ns(args, max_failures=args.max_failures))
    if args.reorganize:
        cmd_reorganize(_followup_ns(args, max_candidates=args.max_failures))
    return 0


def cmd_improve(args: argparse.Namespace) -> int:
    vault = Path(args.vault).resolve()
    rows = expand_candidates(Path(args.results))
    if not rows:
        print("No expand candidates (relevance >= floor but retrieval weak).")
        return 0
    rows = rows[: args.max_failures]
    print(f"Expanding {len(rows)} standalone-worthy node(s)…")
    prompt = build_improvement_prompt(_rows_to_probe_results(rows), vault)
    if args.dry_run:
        print(prompt[:4000])
        return 0
    print(call_hermes(prompt, vault_dir=vault)[:2000])
    _git_commit_vault(vault, f"probe: expand {len(rows)} relevance-worthy nodes")
    return 0


def cmd_reorganize(args: argparse.Namespace) -> int:
    vault = Path(args.vault).resolve()
    rows = reorganize_candidates(Path(args.results))
    if not rows:
        print("No reorganize candidates (nothing below relevance floor).")
        return 0
    rows = rows[: args.max_candidates]
    floor = rows[0].get("relevance_floor", args.relevance_floor)
    print(f"Reorganizing {len(rows)} below-floor node(s) (floor={floor})…")
    prompt = build_reorganize_prompt(_rows_to_probe_results(rows), vault)
    if args.dry_run:
        print(prompt[:4000])
        return 0
    print(call_hermes(prompt, vault_dir=vault)[:2000])
    _git_commit_vault(vault, f"probe: reorganize {len(rows)} below-floor nodes")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    rows = load_results(Path(args.results))
    if not rows:
        print("No results yet.")
        return 0
    by_verdict: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict", "fail")
        by_verdict[v] = by_verdict.get(v, 0) + 1
    print(f"Total probes: {len(rows)}")
    for v in ("keep", "expand", "reorganize", "fail"):
        if by_verdict.get(v):
            print(f"  {v}: {by_verdict[v]}")
    floor = rows[-1].get("relevance_floor", 0.4)
    print(f"\nRelevance floor (last run): {floor}")
    print("\nReorganize candidates (below floor):")
    for r in reorganize_candidates(Path(args.results))[-10:]:
        rel = r.get("relevance_score")
        print(f"  {r['node']['rel_path']} rel={rel} → {r.get('merge_target') or r.get('disposition', '')[:60]}")
    print("\nExpand candidates:")
    for r in expand_candidates(Path(args.results))[-5:]:
        print(f"  {r['node']['rel_path']} rel={r.get('relevance_score')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Vault today-relevance probe harness.")
    parser.add_argument("--vault", default=str(VAULT_DIR))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--relevance-floor", type=float, default=0.4,
        help="Min relevance_score (0–1) to warrant a standalone node (default 0.4)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--types", nargs="+", default=["concepts", "threads", "entities", "timeline"])

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--types", nargs="+", default=["concepts", "threads", "entities", "timeline"])
    p_probe.add_argument("--max-nodes", type=int, default=10)
    p_probe.add_argument("--all", action="store_true")
    p_probe.add_argument("--skip-existing", action="store_true")
    p_probe.add_argument("--concurrency", type=int, default=1)
    p_probe.add_argument("--min-chars", type=int, default=200)
    p_probe.add_argument(
        "--mode",
        choices=["tiered", "inline", "agent"],
        default="tiered",
        help="tiered=local context + one-shot LLM, escalate only expand/fail (default); "
             "inline=never escalate; agent=full vault search every time (slow)",
    )
    p_probe.add_argument(
        "--relevance-floor", type=float, default=0.4,
        help="Min relevance_score (0–1) for standalone node (default 0.4)",
    )
    p_probe.add_argument("--improve", action="store_true", help="Run expand pass after probe")
    p_probe.add_argument("--reorganize", action="store_true", help="Run reorganize pass after probe")
    p_probe.add_argument("--max-failures", type=int, default=15)

    p_imp = sub.add_parser("improve", help="Expand standalone-worthy nodes")
    p_imp.add_argument("--max-failures", type=int, default=15)
    p_imp.add_argument("--dry-run", action="store_true")

    p_reorg = sub.add_parser("reorganize", help="Merge/demote below-floor nodes")
    p_reorg.add_argument("--max-candidates", type=int, default=20)
    p_reorg.add_argument("--dry-run", action="store_true")

    sub.add_parser("summary")

    args = parser.parse_args()
    # Subcommand-specific --relevance-floor overrides the global default.
    if getattr(args, "relevance_floor", None) is None:
        args.relevance_floor = 0.4
    if args.command == "audit":
        return cmd_audit(Path(args.vault), tuple(args.types))
    if args.command == "probe":
        return cmd_probe(args)
    if args.command == "improve":
        return cmd_improve(args)
    if args.command == "reorganize":
        return cmd_reorganize(args)
    if args.command == "summary":
        return cmd_summary(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
