"""Export resolved episodic memory to Obsidian run notes (YAML + markdown)."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from harness.memory_schema import EpisodicRecord, ToolCallRecord
from harness.memory_store import JsonlMemoryStore

_SLUG_INVALID = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_filename(job_id: str) -> str:
    base = job_id.strip().replace(" ", "_")
    base = _SLUG_INVALID.sub("-", base).strip("-") or "run"
    return f"{base}.md"


def _horizon_days(ep: EpisodicRecord) -> int:
    return max(0, (ep.resolution_date - ep.cutoff_date).days)


def _checks_line(fired: list[str]) -> str:
    if not fired:
        return "_none_"
    return ", ".join(f"`{c}`" for c in fired)


def _evidence_lines(tool_calls: list[ToolCallRecord]) -> list[str]:
    by_ns: dict[str, list[ToolCallRecord]] = defaultdict(list)
    for t in tool_calls:
        by_ns[t.tool_name].append(t)
    lines: list[str] = []
    for name, calls in sorted(by_ns.items()):
        art = sum(c.evidence_count for c in calls)
        lines.append(f"- **{name}**: {len(calls)} call(s), ~{art} evidence items")
    return lines


def _forecast_direction(p: float) -> str:
    if p > 0.5 + 1e-6:
        return "forecast leaned yes"
    if p < 0.5 - 1e-6:
        return "forecast leaned no"
    return "forecast near 0.5"


def build_run_note(
    ep: EpisodicRecord,
    *,
    market_label: str,
    analogues_surfaced: list[str] | None = None,
) -> str:
    fired = list(ep.blind_spot_checks_fired)
    hd = _horizon_days(ep)
    brier = ep.brier_score
    approach_fm: Any = fired if len(fired) != 1 else fired[0]

    front: dict[str, Any] = {
        "run_id": ep.job_id,
        "question": ep.question,
        "market": market_label,
        "category": ep.market_family,
        "horizon_days": hd,
        "approach": approach_fm,
        "analogues_surfaced": list(analogues_surfaced or []),
        "p_yes": round(float(ep.final_p_yes), 6),
        "brier": float(brier) if brier is not None else None,
        "date": ep.resolution_date.isoformat(),
    }
    fm_yaml = yaml.safe_dump(
        front,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    ).rstrip() + "\n"

    ci = ep.confidence_interval
    ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None else "n/a"

    evidence_lines = _evidence_lines(ep.tool_calls)
    if not evidence_lines:
        evidence_lines = ["- _No tool evidence recorded._"]

    links_checks = " ".join(f"[[{c}]]" for c in fired) or "_none_"
    cat_link = f"[[{ep.market_family} forecasting]]" if ep.market_family else "_none_"

    brier_txt = f"{brier:.3f}" if brier is not None else "n/a"
    miss_txt = ", ".join(ep.misses) if ep.misses else "—"

    body = f"""# Run {ep.job_id}

## Question
{ep.question}

## Context
- **Cutoff**: {ep.cutoff_date.isoformat()} → **Resolution**: {ep.resolution_date.isoformat()} ({hd} day horizon)
- **Category**: {ep.market_family}
- **Checks fired**: {_checks_line(fired)}

## Evidence Gathered
{chr(10).join(evidence_lines)}

## Reasoning
{ep.notes.strip()}

## Forecast
- **p_yes**: {ep.final_p_yes:.3f}
- **Confidence interval**: {ci_txt}

## Outcome
- **Brier**: {brier_txt}
- **Miss tags / resolver notes**: {miss_txt}
- **Direction**: {_forecast_direction(float(ep.final_p_yes))}

## Links
- Related to: {cat_link}
- Used approach: {links_checks}
"""
    return f"---\n{fm_yaml}---\n\n{body.strip()}\n"


def export_resolved_episodes(
    memory_dir: Path,
    vault_dir: Path,
    *,
    runs_subdir: str = "runs",
    since: date | None = None,
    market_label: str = "polymarket",
) -> list[Path]:
    """Write one note per resolved episode (``brier_score`` set) into ``vault_dir/runs``."""

    store = JsonlMemoryStore(memory_dir.expanduser().resolve())
    out_dir = vault_dir.expanduser().resolve() / runs_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for ep in store.read_all_episodes():
        if ep.brier_score is None:
            continue
        if since is not None and ep.resolution_date < since:
            continue
        text = build_run_note(ep, market_label=market_label)
        path = out_dir / _safe_filename(ep.job_id)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    default_vault = Path.home() / "vaults" / "harness-journal"
    parser = argparse.ArgumentParser(description="Export resolved episodes from JSONL memory to Obsidian notes.")
    parser.add_argument("--vault-dir", type=Path, default=default_vault)
    parser.add_argument("--memory-dir", type=Path, default=Path(".harness_memory"))
    parser.add_argument("--runs-subdir", type=str, default="runs")
    parser.add_argument("--since", type=date.fromisoformat, default=None)
    parser.add_argument("--market", type=str, default="polymarket")
    args = parser.parse_args(argv)

    paths = export_resolved_episodes(
        args.memory_dir,
        args.vault_dir,
        runs_subdir=args.runs_subdir,
        since=args.since,
        market_label=args.market,
    )
    print(f"wrote {len(paths)} note(s) under {args.vault_dir.resolve() / args.runs_subdir}")
    return 0


__all__ = ["build_run_note", "export_resolved_episodes", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
