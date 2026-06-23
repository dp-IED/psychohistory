"""Vault-aware calibration model.

Unlike ReasoningCalibration (which treats mechanisms as opaque strings),
this model reads the vault to understand:

1. Mechanism relationships — pools stats across related concepts
2. Domain context — knows which Polymarket tags are the right baseline
3. YES/NO polarity — knows if a mechanism predicts YES or NO

Architecture:
  vault concept files (YAML frontmatter)
    → mechanism graph (related_concepts wikilinks)
      → hierarchical pooling (domain → cluster → mechanism)
        → calibrated posterior per mechanism
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harness.reasoning_trail import (
    ReasoningTrail, parse_all_runs, load_gold_resolutions,
)


@dataclass
class MechanismMeta:
    """Metadata extracted from a vault concept file."""
    name: str
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)  # normalized concept names
    polarity: str = ""  # "yes" if predicts YES, "no" if predicts NO, "" if neutral
    filepath: str = ""


@dataclass
class VaultDiagnostic:
    """Per-mechanism diagnostic with domain and pooling context."""
    name: str
    domain: str
    hit_rate: float
    ci_low: float
    ci_high: float
    n_resolved: int
    n_yes: int
    tag_baseline: float
    value_added: float
    pooled_from: list[str] = field(default_factory=list)
    related_mechanisms: list[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.n_resolved >= 3


class VaultCalibration:
    """Calibration model that reads the vault to understand mechanisms."""

    def __init__(self, vault_dir: str | Path):
        self.vault_dir = Path(vault_dir)
        # Mechanism metadata from vault
        self._meta: dict[str, MechanismMeta] = {}
        # Stats: {name: {yes, no, p_yes_sum, brier_sum}}
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"yes": 0, "no": 0, "p_yes_sum": 0.0, "brier_sum": 0.0}
        )
        # Domain → Polymarket tag baseline mapping
        self._domain_baseline: dict[str, float] = {}
        self._n_total = 0

        # Load vault metadata
        self._load_vault_metadata()

    # ── vault reading ──────────────────────────────────────────────────

    def _load_vault_metadata(self) -> None:
        """Scan concept files for mechanism metadata."""
        concepts_dir = self.vault_dir / "domains"
        if not concepts_dir.exists():
            return

        for md_file in concepts_dir.rglob("_concept.md"):
            self._parse_concept_file(md_file)
        for md_file in concepts_dir.rglob("*.md"):
            if md_file.name.startswith("_concept") or md_file.name == "_index.md":
                continue
            # Only parse if it looks like a concept (has type: concept frontmatter)
            text = md_file.read_text(encoding="utf-8")
            if "type: concept" in text[:200]:
                self._parse_concept_file(md_file)

    def _parse_concept_file(self, filepath: Path) -> None:
        """Extract mechanism metadata from a concept file."""
        text = filepath.read_text(encoding="utf-8")
        fm = self._parse_yaml_frontmatter(text)
        if not fm:
            return

        name = fm.get("slug", "") or fm.get("title", "")
        if not name:
            name = filepath.parent.name if filepath.name == "_concept.md" else filepath.stem
        name = name.lower().replace(" ", "-")[:60]

        # Parse related concepts from wikilinks
        related_raw = fm.get("related_concepts") or []
        if isinstance(related_raw, str):
            related_raw = [related_raw]
        related = []
        for r in related_raw:
            # Extract concept name from [[path/to/concept]]
            m = re.search(r'\[\[(?:.*?/)?([^/\]]+?)(?:/_concept)?\]\]', str(r))
            if m:
                related.append(m.group(1).lower().replace(" ", "-"))
            else:
                related.append(str(r).lower().replace(" ", "-")[:60])

        # Determine polarity from content
        polarity = self._detect_polarity(text)

        self._meta[name] = MechanismMeta(
            name=name,
            domain=fm.get("domain", ""),
            tags=fm.get("tags", []),
            related=related,
            polarity=polarity,
            filepath=str(filepath),
        )

    @staticmethod
    def _parse_yaml_frontmatter(text: str) -> dict | None:
        match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if not match:
            return None
        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None

    @staticmethod
    def _detect_polarity(text: str) -> str:
        """Heuristic: does this concept predict YES or NO?"""
        text_lower = text.lower()
        yes_signals = ["near-certain", "lock-in", "ratification becomes",
                       "functionally impossible", "structurally dominant",
                       "p(yes) > 0.90", "high-confidence yes"]
        no_signals = ["structurally cannot", "ceiling below", "near-zero",
                      "p(yes) < 0.10", "functionally impossible to win",
                      "structurally capped", "third-party ceiling"]

        yes_count = sum(1 for s in yes_signals if s in text_lower)
        no_count = sum(1 for s in no_signals if s in text_lower)

        if yes_count > no_count:
            return "yes"
        elif no_count > yes_count:
            return "no"
        return ""

    # ── training ───────────────────────────────────────────────────────

    def update(self, trail: ReasoningTrail) -> None:
        """Ingest one forecast with its reasoning trail."""
        self._n_total += 1
        if trail.resolution is None:
            return

        for mech_raw in trail.mechanisms:
            name = self._normalize_mechanism(mech_raw)
            if trail.resolution:
                self._stats[name]["yes"] += 1
            else:
                self._stats[name]["no"] += 1
            self._stats[name]["p_yes_sum"] += trail.p_yes

    def load_trails(self, resolutions: dict[str, bool] | None = None) -> int:
        """Parse and train on all run notes."""
        trails = parse_all_runs(self.vault_dir, resolutions=resolutions)
        resolved = [t for t in trails if t.resolution is not None]
        for t in resolved:
            self.update(t)
        return len(resolved)

    def set_domain_baselines(self, baselines: dict[str, float]) -> None:
        """Set Polymarket tag baselines per domain."""
        self._domain_baseline = baselines

    # ── mechanism graph ────────────────────────────────────────────────

    def get_mechanism_cluster(self, name: str, max_depth: int = 1) -> list[str]:
        """Return mechanisms related to this one (bounded depth)."""
        visited = set()
        cluster = []
        queue = [(name, 0)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            cluster.append(current)

            meta = self._meta.get(current)
            if meta and depth < max_depth:
                for related in meta.related:
                    if related not in visited:
                        queue.append((related, depth + 1))

        return cluster

    # ── query ──────────────────────────────────────────────────────────

    def query(self, mechanisms: list[str]) -> dict:
        """Calibrate using mechanisms with vault-aware pooling."""
        if not mechanisms:
            return {"mean": 0.5, "mechanisms_used": [], "diagnostics": {}}

        diagnostics: dict[str, VaultDiagnostic] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for mech_raw in mechanisms:
            name = self._normalize_mechanism(mech_raw)
            meta = self._meta.get(name)

            # Pool stats from this mechanism + related ones
            cluster = self.get_mechanism_cluster(name) if meta else [name]
            pooled_yes = sum(self._stats.get(c, {}).get("yes", 0) for c in cluster)
            pooled_no = sum(self._stats.get(c, {}).get("no", 0) for c in cluster)
            n = pooled_yes + pooled_no

            alpha = 1 + pooled_yes
            beta = 1 + pooled_no
            hit_rate = alpha / (alpha + beta) if (alpha + beta) > 2 else 0.5

            ci_low = _beta_ppf(alpha, beta, 0.05)
            ci_high = _beta_ppf(alpha, beta, 0.95)

            domain = meta.domain if meta else ""
            tag_baseline = self._domain_baseline.get(domain, 0.5)

            diagnostics[name] = VaultDiagnostic(
                name=name,
                domain=domain,
                hit_rate=round(hit_rate, 3),
                ci_low=round(ci_low, 3),
                ci_high=round(ci_high, 3),
                n_resolved=n,
                n_yes=pooled_yes,
                tag_baseline=round(tag_baseline, 3),
                value_added=round(hit_rate - tag_baseline, 3),
                pooled_from=[c for c in cluster if c != name][:5],
                related_mechanisms=meta.related[:5] if meta else [],
            )

            w = alpha + beta
            weighted_sum += hit_rate * w
            weight_sum += w

        pooled_mean = weighted_sum / weight_sum if weight_sum > 0 else 0.5

        return {
            "mean": round(pooled_mean, 4),
            "mechanisms_used": list(diagnostics.keys()),
            "diagnostics": {
                name: {
                    "domain": d.domain,
                    "hit_rate": d.hit_rate,
                    "ci": [d.ci_low, d.ci_high],
                    "n": d.n_resolved,
                    "value_added": d.value_added,
                    "pooled_from": d.pooled_from,
                    "reliable": d.reliable,
                }
                for name, d in diagnostics.items()
            },
        }

    # ── diagnostics ────────────────────────────────────────────────────

    def diagnostics(self) -> str:
        """Human-readable report with vault-aware context."""
        lines = [
            "=== Vault-Aware Calibration Diagnostics ===",
            f"Total forecasts: {self._n_total}",
            f"Mechanisms with vault metadata: {len(self._meta)}",
            f"Mechanisms with stats: {len(self._stats)}",
            "",
            f"{'Mechanism':<45} {'Domain':<12} {'N':>4} {'Hit':>6} {'CI':>16} {'Δbase':>7} {'Pooled'}",
            "-" * 100,
        ]

        # Sort by resolved count
        sorted_mechs = sorted(
            self._stats.items(),
            key=lambda x: x[1]["yes"] + x[1]["no"],
            reverse=True,
        )

        for name, s in sorted_mechs:
            n = s["yes"] + s["no"]
            if n == 0:
                continue

            meta = self._meta.get(name)
            domain = meta.domain if meta else "?"
            cluster = self.get_mechanism_cluster(name)
            pooled_n = sum(
                self._stats.get(c, {}).get("yes", 0) +
                self._stats.get(c, {}).get("no", 0)
                for c in cluster
            )
            pooled_yes = sum(self._stats.get(c, {}).get("yes", 0) for c in cluster)

            alpha = 1 + pooled_yes
            beta = 1 + (pooled_n - pooled_yes)
            hit = alpha / (alpha + beta) if (alpha + beta) > 2 else 0.5
            lo = _beta_ppf(alpha, beta, 0.05)
            hi = _beta_ppf(alpha, beta, 0.95)

            baseline = self._domain_baseline.get(domain, 0.5)
            delta = hit - baseline

            reliable = "✓" if pooled_n >= 3 else "?"
            pool_info = f"[+{len(cluster)-1}]" if len(cluster) > 1 else ""

            lines.append(
                f"  {name:<43} {domain:<12} {pooled_n:>4} {hit:.3f}  "
                f"[{lo:.3f}, {hi:.3f}]  {delta:+.3f}  {reliable} {pool_info}"
            )

        return "\n".join(lines)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_mechanism(raw: str) -> str:
        name = raw.split(':')[0].strip()
        name = name.split('(')[0].strip()
        return name.lower().replace(' ', '-')[:60]


def _beta_ppf(alpha: float, beta: float, q: float) -> float:
    import math
    if q <= 0: return 0.0
    if q >= 1: return 1.0
    mean = alpha / (alpha + beta)
    x = mean
    for _ in range(20):
        cdf = _reg_beta_cdf(x, alpha, beta)
        pdf_val = math.exp(
            math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
            + (alpha - 1) * math.log(max(x, 1e-15))
            + (beta - 1) * math.log(max(1 - x, 1e-15))
        )
        if pdf_val < 1e-15: break
        dx = (cdf - q) / pdf_val
        x -= dx
        if abs(dx) < 1e-8: break
        x = max(0.0, min(1.0, x))
    return x


def _reg_beta_cdf(x: float, a: float, b: float) -> float:
    import math
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _reg_beta_cdf(1 - x, b, a)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - log_beta) / a
    f, c2, d = 1.0, 1.0, 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, 200):
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c2 = 1.0 + num / c2
        if abs(c2) < 1e-30: c2 = 1e-30
        d = 1.0 / d
        f *= d * c2
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c2 = 1.0 + num / c2
        if abs(c2) < 1e-30: c2 = 1e-30
        d = 1.0 / d
        delta = d * c2
        f *= delta
        if abs(delta - 1.0) < 1e-12: break
    return min(1.0, max(0.0, front * (f - 1.0)))
