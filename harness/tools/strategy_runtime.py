"""Load vault files for synthesis context — strategy, approaches, policy.

Simplified: no Dataview execution here. The synthesis agent runs Dataview
queries itself via the dataview_query CLI tool. We just concatenate the
markdown files so the agent can read the strategy and approach notes.
"""

from __future__ import annotations

from pathlib import Path

STRATEGY_FILENAME = "_strategy.md"
APPROACHES_DIR = "approaches"


def load_strategy_markdown(vault_dir: Path) -> str:
    path = vault_dir.expanduser().resolve() / STRATEGY_FILENAME
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def load_approach_notes(vault_dir: Path) -> str:
    """Read all approach notes from vault/approaches/ and return concatenated text."""
    root = vault_dir.expanduser().resolve()
    approaches_dir = root / APPROACHES_DIR
    if not approaches_dir.is_dir():
        return ""

    blocks: list[str] = []
    for path in sorted(approaches_dir.glob("*.md")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        blocks.append(f"## {path.stem}\n\n{content}")

    return "\n\n".join(blocks) if blocks else ""


def build_vault_synthesis_bundle(
    vault_dir: Path,
    *,
    category: str,
    horizon_days: int,
) -> str:
    """Read strategy + approach notes and return them as a plain-text bundle.

    The synthesis agent is expected to:
    1. Read this bundle for protocol and procedural knowledge.
    2. Use the Dataview CLI tool to query past runs by category/horizon.
    3. Read individual run files from vault/runs/ for detailed context.
    """
    root = vault_dir.expanduser().resolve()
    strategy = load_strategy_markdown(root)
    approaches = load_approach_notes(root)

    blocks: list[str] = []
    if strategy.strip():
        blocks.append("# Forecasting Strategy\n")
        blocks.append(strategy.strip())
    if approaches.strip():
        blocks.append("\n---\n# Approach Notes\n")
        blocks.append(approaches.strip())

    if not blocks:
        return ""

    context_md = "\n\n".join(blocks)

    # Append a quick-reference footer telling the agent what tools are available
    footer = f"""
---
## Available tools for this forecast

- **Dataview queries**: Run `python -m harness.tools.dataview_query --category "{category}" --horizon {horizon_days}` to query past runs.
- **Vault runs**: Read individual run files from `{root / 'runs'}`.
- **Approach notes**: Already included above. Reference them explicitly in your reasoning.
- **Policy**: Read `{root.parent / '.harness' / 'policy.md'}` for the machine policy body.
"""
    return context_md + footer


__all__ = [
    "STRATEGY_FILENAME",
    "build_vault_synthesis_bundle",
    "load_approach_notes",
    "load_strategy_markdown",
]
