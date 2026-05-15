"""Load and save forecasting policy from `.harness/policy.md` (YAML frontmatter + markdown body)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MAX_STEPS = 4
DEFAULT_CONVERGENCE_EPSILON = 0.01


@dataclass
class PolicyConfig:
    blind_spot_checks: list[str]
    max_steps: int = DEFAULT_MAX_STEPS
    convergence_epsilon: float = DEFAULT_CONVERGENCE_EPSILON
    shrinkage: float | None = None
    body: str = ""
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.convergence_epsilon < 0:
            raise ValueError("convergence_epsilon must be >= 0")
        if self.shrinkage is not None and not (0.0 <= self.shrinkage <= 1.0):
            raise ValueError("shrinkage must be in [0, 1] when set")


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    text = raw.lstrip("\ufeff")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    closing: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing = idx
            break
    if closing is None:
        return {}, text

    fm_block = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1 :])
    body = body.lstrip("\n")
    loaded = yaml.safe_load(fm_block)
    if loaded is None:
        return {}, body
    if not isinstance(loaded, dict):
        raise ValueError("YAML frontmatter must map to a dict")
    return loaded, body


def load_policy(path: Path = Path(".harness/policy.md")) -> PolicyConfig:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return PolicyConfig(blind_spot_checks=[], body="")

    data, body = _split_frontmatter(resolved.read_text(encoding="utf-8"))
    raw = dict(data)

    blind_spot_checks = data.get("blind_spot_checks")
    if blind_spot_checks is None:
        blind_spot_checks = []
    if not isinstance(blind_spot_checks, list) or not all(isinstance(x, str) for x in blind_spot_checks):
        raise ValueError("blind_spot_checks must be a list of strings when present")

    max_steps = int(data.get("max_steps", DEFAULT_MAX_STEPS))
    convergence_epsilon = float(data.get("convergence_epsilon", DEFAULT_CONVERGENCE_EPSILON))
    shrinkage_raw = data.get("shrinkage", None)
    shrinkage: float | None
    if shrinkage_raw is None or (
        isinstance(shrinkage_raw, str) and shrinkage_raw.lower() in ("null", "none", "~")
    ):
        shrinkage = None
    else:
        shrinkage = float(shrinkage_raw)

    return PolicyConfig(
        blind_spot_checks=list(blind_spot_checks),
        max_steps=max_steps,
        convergence_epsilon=convergence_epsilon,
        shrinkage=shrinkage,
        body=body,
        raw_frontmatter=raw,
    )


def save_policy(config: PolicyConfig, path: Path = Path(".harness/policy.md")) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    fm_payload = {
        "blind_spot_checks": config.blind_spot_checks,
        "max_steps": config.max_steps,
        "convergence_epsilon": config.convergence_epsilon,
        "shrinkage": config.shrinkage,
    }
    dumped = yaml.safe_dump(
        fm_payload,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip() + "\n"

    text = f"---\n{dumped}---\n"
    if config.body.strip():
        text += "\n" + config.body.lstrip("\n")
    resolved.write_text(text, encoding="utf-8")


__all__ = [
    "DEFAULT_MAX_STEPS",
    "PolicyConfig",
    "load_policy",
    "save_policy",
]
