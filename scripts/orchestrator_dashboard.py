#!/usr/bin/env python3
"""Orchestrator Dashboard — live terminal dashboard for the multi-agent system.

Shows:
  - Agent roster health (active / stub / deprecated)
  - Recent orchestration runs with p_yes, agents used, vault edits
  - Agent performance tracking (Brier scores per agent type)
  - Roster gaps discovered
  - Live --watch mode (polls every N seconds for new runs)

Usage:
  python -m scripts.orchestrator_dashboard          # snapshot
  python -m scripts.orchestrator_dashboard --watch   # live refresh every 10s
  python -m scripts.orchestrator_dashboard --watch --interval 30
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.config import VAULT_DIR

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "orchestrator" / "log.jsonl"
AGENT_ROLES_DIR = VAULT_DIR / "agent-roles"
RUNS_DIR = VAULT_DIR / "runs"

try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from datetime import datetime, timezone
    import subprocess
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_agent_roster() -> list[dict[str, Any]]:
    """Scan agent-roles/ directory and parse frontmatter."""
    roster: list[dict[str, Any]] = []
    if not AGENT_ROLES_DIR.exists():
        return roster
    for f in sorted(AGENT_ROLES_DIR.glob("*.md")):
        if f.name.startswith("_"):
            continue  # skip meta docs
        text = f.read_text(encoding="utf-8", errors="replace")
        name = f.stem
        status = "active"
        domain: list[str] = []
        region: list[str] = []
        kind = "unknown"
        # Crude frontmatter parsing
        in_fm = False
        for line in text.split("\n"):
            if line.strip() == "---" and not in_fm:
                in_fm = True
                continue
            if line.strip() == "---" and in_fm:
                break
            if in_fm:
                if line.startswith("status:"):
                    status = line.split(":", 1)[1].strip()
                elif line.startswith("domain:"):
                    raw = line.split(":", 1)[1].strip()
                    domain = [d.strip().strip("[]\"'") for d in raw.strip("[]").split(",") if d.strip()]
                elif line.startswith("region:"):
                    raw = line.split(":", 1)[1].strip()
                    region = [r.strip().strip("[]\"'") for r in raw.strip("[]").split(",") if r.strip()]
                elif line.startswith("kind:"):
                    kind = line.split(":", 1)[1].strip()
        roster.append({
            "name": name,
            "status": status,
            "kind": kind,
            "domain": domain,
            "region": region,
            "path": str(f.relative_to(ROOT)),
        })
    return roster


def load_upcoming_runs() -> list[dict[str, Any]]:
    """Read hermes cron list to find scheduled polymarket-scanner runs."""
    upcoming: list[dict[str, Any]] = []
    try:
        result = subprocess.run(
            ["hermes", "cron", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return upcoming
        # Parse the multi-line format: each job is a block starting with UUID
        lines = result.stdout.split("\n")
        current: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            # Detect job start: UUID + [active|paused]
            if len(stripped) > 10 and "[" in stripped and "]" in stripped:
                parts = stripped.split()
                if current and current.get("name") in ("polymarket-scanner", "daily-hypothesis"):
                    upcoming.append(current)
                current = {"job_id": parts[0].strip() if parts else "?"}
            elif "Name:" in stripped:
                current["name"] = stripped.split(":", 1)[1].strip()
            elif "Schedule:" in stripped:
                current["schedule"] = stripped.split(":", 1)[1].strip()
            elif "Next run:" in stripped:
                current["next_run"] = stripped.split(":", 1)[1].strip()
            elif "Last run:" in stripped:
                # Last run line: "2026-05-18T17:16:01.590296+01:00  ok"
                status_part = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                current["last_status"] = status_part.split()[-1] if status_part.split() else "?"
        # Don't forget the last job
        if current and current.get("name") in ("polymarket-scanner", "daily-hypothesis"):
            upcoming.append(current)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return upcoming


def load_orchestration_log(limit: int = 50) -> list[dict[str, Any]]:
    """Load recent orchestration runs from JSONL log."""
    entries: list[dict[str, Any]] = []
    if not LOG_PATH.exists():
        return entries
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-limit:]


def compute_agent_stats(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-agent performance stats from orchestration log."""
    stats: dict[str, dict[str, Any]] = {}
    for entry in entries:
        agents = entry.get("agents_used", [])
        for agent in agents:
            if agent not in stats:
                stats[agent] = {"runs": 0, "briers": [], "gaps_found": 0}
            stats[agent]["runs"] += 1
            brier = entry.get("brier")
            if brier is not None and isinstance(brier, (int, float)):
                stats[agent]["briers"].append(brier)
        gaps = entry.get("roster_gaps_identified", [])
        if gaps:
            for agent in agents:
                stats[agent]["gaps_found"] += len(gaps)
    return stats


def compute_global_brier() -> float | None:
    """Read global mean Brier from harness.runs."""
    try:
        from harness.runs import mean_brier
        return mean_brier(str(VAULT_DIR))
    except Exception:
        return None


def compute_global_accuracy() -> dict[str, Any]:
    """Compute accuracy from graph-vault/forecasts/ entries.
    
    Returns dict with total, correct, accuracy, and category breakdown.
    """
    forecasts_dir = VAULT_DIR / "forecasts"
    total = 0
    correct = 0
    by_cat: dict[str, dict[str, int]] = {}
    if not forecasts_dir.exists():
        return {"total": 0, "correct": 0, "accuracy": None, "by_cat": {}}
    
    for f in sorted(forecasts_dir.glob("*.md")):
        if f.name.startswith("_") or "starmer" in f.name:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            parts = text.split("---")
            if len(parts) < 3:
                continue
            import yaml
            fm = yaml.safe_load(parts[1])
            if not isinstance(fm, dict):
                continue
            pred = str(fm.get("prediction", "")).upper().strip()
            actual = str(fm.get("actual", "")).upper().strip()
            if pred not in ("YES", "NO") or actual not in ("YES", "NO"):
                continue
            total += 1
            if pred == actual:
                correct += 1
            # Category breakdown
            cat = str(fm.get("category", "unknown")).strip() or "unknown"
            if cat not in by_cat:
                by_cat[cat] = {"total": 0, "correct": 0}
            by_cat[cat]["total"] += 1
            if pred == actual:
                by_cat[cat]["correct"] += 1
        except Exception:
            continue
    
    accuracy = correct / total if total > 0 else None
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_cat": by_cat,
    }


# ---------------------------------------------------------------------------
# Dashboard Rendering
# ---------------------------------------------------------------------------


def build_dashboard(
    roster: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    upcoming: list[dict[str, Any]] | None = None,
) -> Layout:
    if not HAS_RICH:
        return _plain_dashboard(roster, entries, stats)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="roster", ratio=2),
        Layout(name="main", ratio=3),
    )
    layout["main"].split_column(
        Layout(name="recent_runs", ratio=3),
        Layout(name="upcoming_runs", ratio=2),
        Layout(name="stats_panel", ratio=1),
    )

    # Header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global_brier = compute_global_brier()
    accuracy_data = compute_global_accuracy()
    acc = accuracy_data.get("accuracy")
    acc_str = f"Acc={acc:.0%}" if acc is not None else ""
    brier_str = f"Brier={global_brier:.4f}" if global_brier is not None else acc_str
    header_text = Text(
        f"╔══ Psychohistory Orchestrator Dashboard ══╗\n"
        f"║  {now}  {brier_str:>32}  ║\n"
        f"╚══ {len(entries)} runs logged, {len(roster)} agent roles, {len(upcoming or [])} scheduled ══╝",
        style="bold cyan",
    )
    layout["header"].update(Panel(header_text, box=box.SIMPLE))

    # Agent Roster Panel
    roster_table = Table(box=box.SIMPLE, title="Agent Roster", title_style="bold")
    roster_table.add_column("Agent", style="cyan")
    roster_table.add_column("Kind", style="dim")
    roster_table.add_column("Status")
    roster_table.add_column("Domains", style="magenta", no_wrap=False)

    for agent in roster:
        status_style = {
            "active": "green",
            "stub": "yellow",
            "deprecated": "red",
            "fading": "dim",
        }.get(agent["status"], "white")
        domains = ", ".join(agent["domain"][:4])
        if len(agent["domain"]) > 4:
            domains += f" +{len(agent['domain'])-4}"
        roster_table.add_row(
            agent["name"],
            agent["kind"],
            Text(agent["status"], style=status_style),
            domains,
        )
    layout["roster"].update(Panel(roster_table, box=box.SIMPLE))

    # Recent Runs Table
    runs_table = Table(box=box.SIMPLE, title="Recent Orchestrations", title_style="bold")
    runs_table.add_column("Time", style="dim", width=11)
    runs_table.add_column("Question", no_wrap=False, width=35)
    runs_table.add_column("p_yes", justify="right", width=6)
    runs_table.add_column("Agents", width=20)
    runs_table.add_column("Edits")

    for entry in reversed(entries[-12:]):
        ts = entry.get("_logged_at", "")[11:19] if entry.get("_logged_at") else "?"
        q = entry.get("question", "?")[:34]
        py = entry.get("p_yes")
        py_str = f"{py:.0%}" if isinstance(py, (int, float)) else "?"
        agents = ", ".join(entry.get("agents_used", []))[:19]
        edits = "📝" if entry.get("vault_edits_summary") else ""
        runs_table.add_row(ts, q, py_str, agents, edits)
    layout["recent_runs"].update(Panel(runs_table, box=box.SIMPLE))

    # Upcoming Runs Panel
    if upcoming:
        upcoming_table = Table(box=box.SIMPLE, title="Scheduled Runs", title_style="bold")
        upcoming_table.add_column("Job", style="cyan", width=18)
        upcoming_table.add_column("Schedule", width=11)
        upcoming_table.add_column("Next Run", width=18)
        upcoming_table.add_column("Status", width=10)
        for job in upcoming:
            upcoming_table.add_row(
                job.get("name", "?")[:18],
                job.get("schedule", "?")[:10],
                job.get("next_run", "?")[:18],
                job.get("last_status", "?"),
            )
        layout["upcoming_runs"].update(Panel(upcoming_table, box=box.SIMPLE))
    else:
        layout["upcoming_runs"].update(Panel(Text("No scheduled runs found", style="dim"), box=box.SIMPLE))

    # Agent Performance Stats
    stats_table = Table(box=box.SIMPLE, title="Agent Performance", title_style="bold")
    stats_table.add_column("Agent", style="cyan")
    stats_table.add_column("Runs", justify="right", width=5)
    stats_table.add_column("Avg Brier", justify="right", width=10)
    stats_table.add_column("Gaps Found", justify="right", width=10)

    sorted_agents = sorted(stats.items(), key=lambda x: x[1]["runs"], reverse=True)
    for name, s in sorted_agents[:10]:
        avg_b = sum(s["briers"]) / len(s["briers"]) if s["briers"] else None
        brier_str = f"{avg_b:.4f}" if avg_b is not None else "—"
        stats_table.add_row(name[:28], str(s["runs"]), brier_str, str(s["gaps_found"]))
    if not sorted_agents:
        stats_table.add_row("(no data yet)", "0", "—", "0")
    layout["stats_panel"].update(_build_accuracy_panel(accuracy_data))

    return layout


def _build_accuracy_panel(acc_data: dict[str, Any]) -> Panel:
    """Build the global accuracy panel."""
    from rich.table import Table
    acc = acc_data.get("accuracy")
    total = acc_data.get("total", 0)
    correct = acc_data.get("correct", 0)
    by_cat = acc_data.get("by_cat", {})
    
    table = Table(box=box.SIMPLE, title="Historical Performance", title_style="bold")
    table.add_column("Metric", style="cyan", width=16)
    table.add_column("Value", justify="right", width=10)
    
    table.add_row("Total Forecasts", str(total))
    table.add_row("Correct", str(correct))
    if acc is not None:
        table.add_row("Accuracy", f"{acc:.0%}")
    
    if by_cat:
        table.add_section()
        for cat, data in sorted(by_cat.items(), key=lambda x: x[1]["total"], reverse=True)[:4]:
            ca = data["correct"] / data["total"] if data["total"] > 0 else 0
            table.add_row(
                f"  {cat[:12]}",
                f"{data['correct']}/{data['total']} ({ca:.0%})",
            )
    
    return Panel(table, box=box.SIMPLE)


def _plain_dashboard(
    roster: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    upcoming: list[dict[str, Any]] | None = None,
) -> None:
    """Fallback: markdown-style plain text output when Rich is not available."""
    print("=" * 72)
    global_brier = compute_global_brier()
    accuracy_data = compute_global_accuracy()
    acc = accuracy_data.get("accuracy")
    acc_str = f"  Acc={acc:.0%} ({accuracy_data['correct']}/{accuracy_data['total']})" if acc is not None else ""
    brier_str = f"  Brier={global_brier:.4f}" if global_brier is not None else acc_str
    print(f"  Orchestrator Dashboard — {datetime.now().isoformat()[:19]}{brier_str}")
    print(f"  {len(entries)} runs  ·  {len(roster)} agent roles  ·  {len(upcoming or [])} scheduled")
    print("=" * 72)
    print()
    print("  Agent Roster:")
    print(f"  {'Name':<30} {'Status':<10} {'Kind':<12} Domains")
    print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*30}")
    for a in roster:
        domains = ", ".join(a["domain"][:3])
        print(f"  {a['name']:<30} {a['status']:<10} {a['kind']:<12} {domains}")
    print()
    print("  Recent Orchestrations:")
    for entry in reversed(entries[-8:]):
        ts = entry.get("_logged_at", "")[11:19] if entry.get("_logged_at") else "?"
        q = entry.get("question", "?")[:50]
        py = entry.get("p_yes")
        py_str = f"{py:.0%}" if isinstance(py, (int, float)) else "?"
        agents = ", ".join(entry.get("agents_used", [])) or "?"
        print(f"  {ts}  {py_str}  {q}")
        print(f"       agents: {agents}")
    print()
    print("  Agent Performance:")
    for name, s in sorted(stats.items(), key=lambda x: x[1]["runs"], reverse=True)[:8]:
        avg_b = sum(s["briers"]) / len(s["briers"]) if s["briers"] else None
        brier_str = f"brier={avg_b:.4f}" if avg_b is not None else "no data"
        print(f"  {name:<28} {s['runs']} runs, {brier_str}, {s['gaps_found']} gaps found")

    # Historical accuracy
    acc_data = compute_global_accuracy()
    acc = acc_data.get("accuracy")
    if acc is not None:
        print()
        print(f"  Historical: {acc_data['correct']}/{acc_data['total']} = {acc:.0%} accuracy")
        by_cat = acc_data.get("by_cat", {})
        for cat, data in sorted(by_cat.items(), key=lambda x: x[1]["total"], reverse=True)[:5]:
            ca = data["correct"] / data["total"] if data["total"] > 0 else 0
            print(f"    {cat:<20} {data['correct']}/{data['total']} ({ca:.0%})")

    return None  # type: ignore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator Dashboard")
    parser.add_argument("--watch", "-w", action="store_true", help="Live refresh mode")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Refresh interval in seconds (default: 10)")
    args = parser.parse_args()

    if args.watch and HAS_RICH:
        _run_live(args.interval)
    elif args.watch and not HAS_RICH:
        print("Rich not installed. Install with: pip install rich")
        print("Falling back to snapshot mode.")
        _run_snapshot()
    else:
        _run_snapshot()


def _run_snapshot() -> None:
    roster = load_agent_roster()
    entries = load_orchestration_log()
    stats = compute_agent_stats(entries)
    upcoming = load_upcoming_runs()
    layout = build_dashboard(roster, entries, stats, upcoming)
    if HAS_RICH and layout is not None:
        console = Console()
        console.print(layout)
    # _plain_dashboard already prints, no-op fallback


def _run_live(interval: int) -> None:
    console = Console()

    def _generate() -> Layout:
        roster = load_agent_roster()
        entries = load_orchestration_log()
        stats = compute_agent_stats(entries)
        upcoming = load_upcoming_runs()
        return build_dashboard(roster, entries, stats, upcoming)

    try:
        with Live(_generate(), refresh_per_second=1 / interval, screen=True) as live:
            while True:
                live.update(_generate())
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("[dim]Dashboard closed.[/dim]")


if __name__ == "__main__":
    main()
