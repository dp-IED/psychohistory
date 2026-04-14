"""Paths and defaults for the local autoresearch harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Assume CWD is repository root; override with PSYCHOHISTORY_ROOT env in runner."""

    return Path.cwd()


@dataclass
class HarnessPaths:
    root: Path
    probe_dir: Path
    experiments_dir: Path
    log_jsonl: Path
    best_dir: Path
    candidates_dir: Path
    optimization_prompt: Path
    agent_system_prompt: Path

    @staticmethod
    def default(root: Path | None = None) -> HarnessPaths:
        r = root or repo_root()
        exp = r / "autoresearch" / "experiments"
        return HarnessPaths(
            root=r,
            probe_dir=r / "probes",
            experiments_dir=exp,
            log_jsonl=exp / "log.jsonl",
            best_dir=exp / "best",
            candidates_dir=exp / "candidates",
            optimization_prompt=r / "autoresearch" / "prompts" / "optimization_target.md",
            agent_system_prompt=r / "autoresearch" / "prompts" / "agent_system.md",
        )
