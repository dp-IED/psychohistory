"""Policy config loaded from graph-vault markdown with optional YAML frontmatter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Canonical vault root — all harness scripts read/write here.
VAULT_DIR = Path(__file__).resolve().parent.parent / "graph-vault"
DEFAULT_POLICY_PATH = VAULT_DIR / "_forecast_instructions.md"


@dataclass
class PolicyConfig:
    shrinkage: float = 0.04
    max_steps: int = 1
    convergence_epsilon: float = 0.015
    blind_spots: list[str] = field(default_factory=list)
    body: str = ""  # full markdown body (heuristics + wikilinks)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def load_policy(path: str | Path) -> PolicyConfig:
    """Load policy from a graph-vault markdown file (YAML frontmatter + body)."""
    p = Path(path).resolve()
    if not p.exists():
        return PolicyConfig()

    text = p.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)

    cfg = PolicyConfig()

    if m:
        try:
            data: dict[str, Any] = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            data = {}

        if "shrinkage" in data:
            cfg.shrinkage = float(data["shrinkage"])
        if "max_steps" in data:
            cfg.max_steps = int(data["max_steps"])
        if "convergence_epsilon" in data:
            cfg.convergence_epsilon = float(data["convergence_epsilon"])
        if "blind_spots" in data and isinstance(data["blind_spots"], list):
            cfg.blind_spots = [str(s) for s in data["blind_spots"]]

        # Body = everything after frontmatter
        cfg.body = text[m.end():].strip()
    else:
        cfg.body = text.strip()

    return cfg


def save_policy(cfg: PolicyConfig, path: str | Path) -> None:
    """Write policy config + body to a graph-vault markdown file."""
    frontmatter = {
        "shrinkage": cfg.shrinkage,
        "max_steps": cfg.max_steps,
        "convergence_epsilon": cfg.convergence_epsilon,
        "blind_spots": cfg.blind_spots,
    }
    yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()
    content = f"---\n{yaml_str}\n---\n\n{cfg.body.strip()}\n"
    Path(path).resolve().write_text(content, encoding="utf-8")
