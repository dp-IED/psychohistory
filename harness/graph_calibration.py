"""Graph-aware calibration model.

Uses a weighted MechanismGraph to pool statistics across related
mechanisms.  Unlike previous models that treat mechanisms as
independent strings or use unweighted transitive closure, this model:

1. Reads the vault to build a weighted mechanism graph
2. Pools statistics proportionally to edge weights
3. Co-occurring mechanisms that share a domain AND are wikilinked
   get the strongest pooling (w > 1.0)
4. Domain-only connections pool weakly (w = 0.3)
5. No artificial hop limit — natural weights prevent collapse

This is the measurement layer for: "does the agent's reasoning
actually add value over the Polymarket base rate?"
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from harness.mechanism_graph import MechanismGraph, _normalize_mechanism
from harness.reasoning_trail import (
    ReasoningTrail, parse_all_runs, load_gold_resolutions,
)
from harness.tag_calibration import TagCalibration


@dataclass
class GraphDiagnostic:
    """Per-mechanism diagnostic with evidence from the graph."""
    name: str
    hit_rate: float
    ci_low: float
    ci_high: float
    n_direct: int           # observations for THIS mechanism
    n_pooled: int           # observations including weighted neighbors
    tag_baseline: float     # Polymarket tag base rate for this domain
    value_added: float      # hit_rate - baseline
    top_neighbors: list[tuple[str, float]] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.n_pooled >= 5


class GraphCalibration:
    """Calibration model using weighted mechanism graph for pooling."""

    def __init__(self, graph: MechanismGraph,
                 tag_cal: TagCalibration | None = None):
        self.graph = graph
        self.tag_cal = tag_cal

        # Direct stats: {name: {yes, no, p_yes_sum}}
        self._direct: dict[str, dict] = defaultdict(
            lambda: {"yes": 0, "no": 0, "p_yes_sum": 0.0}
        )
        self._n_total = 0

    def update(self, trail: ReasoningTrail) -> None:
        """Ingest one forecast."""
        self._n_total += 1
        if trail.resolution is None:
            return

        for mech_raw in trail.mechanisms:
            name = _normalize_mechanism(mech_raw)
            if trail.resolution:
                self._direct[name]["yes"] += 1
            else:
                self._direct[name]["no"] += 1
            self._direct[name]["p_yes_sum"] += trail.p_yes

    def load_trails(self, vault_dir: str | Path,
                    resolutions: dict[str, bool] | None = None) -> int:
        trails = parse_all_runs(vault_dir, resolutions=resolutions)
        resolved = [t for t in trails if t.resolution is not None]
        for t in resolved:
            self.update(t)
        return len(resolved)

    # ── weighted pooling ───────────────────────────────────────────────

    def _pooled_stats(self, name: str) -> tuple[int, int]:
        """Return (pooled_yes, pooled_no) using weighted neighbor contributions."""
        direct_yes = self._direct.get(name, {}).get("yes", 0)
        direct_no = self._direct.get(name, {}).get("no", 0)

        pooled_yes = direct_yes
        pooled_no = direct_no

        neighbors = self.graph.get_neighbors(name, min_weight=0.2)
        for neighbor, weight in neighbors.items():
            if neighbor == name:
                continue
            neigh_stats = self._direct.get(neighbor, {})
            # Contribution = weight × neighbor's stats
            # Cap at 1.0 to prevent double-counting through strong edges
            contrib = min(weight, 1.0)
            pooled_yes += int(neigh_stats.get("yes", 0) * contrib)
            pooled_no += int(neigh_stats.get("no", 0) * contrib)

        return pooled_yes, pooled_no

    def _tag_baseline(self, name: str) -> float:
        """Get Polymarket tag baseline for this mechanism's domain tags."""
        if self.tag_cal is None:
            return 0.5

        meta = self.graph.get_metadata(name)
        tags = meta.get("tags", [])
        if not tags:
            return 0.5

        result = self.tag_cal.query(tags)
        return result.mean

    # ── query ──────────────────────────────────────────────────────────

    def query(self, mechanisms: list[str]) -> dict:
        """Calibrate using graph-weighted pooling."""
        if not mechanisms:
            return {"mean": 0.5, "mechanisms_used": [], "diagnostics": {}}

        diagnostics: dict[str, GraphDiagnostic] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for mech_raw in mechanisms:
            name = _normalize_mechanism(mech_raw)
            direct = self._direct.get(name, {})
            direct_n = direct.get("yes", 0) + direct.get("no", 0)

            pooled_yes, pooled_no = self._pooled_stats(name)
            pooled_n = pooled_yes + pooled_no

            alpha = 1 + pooled_yes
            beta = 1 + pooled_no
            hit_rate = alpha / (alpha + beta) if (alpha + beta) > 2 else 0.5

            ci_low = _beta_ppf(alpha, beta, 0.05)
            ci_high = _beta_ppf(alpha, beta, 0.95)

            baseline = self._tag_baseline(name)
            neighbors = self.graph.get_neighbors(name, min_weight=0.3)
            top = sorted(neighbors.items(), key=lambda x: -x[1])[:5]

            diagnostics[name] = GraphDiagnostic(
                name=name,
                hit_rate=round(hit_rate, 3),
                ci_low=round(ci_low, 3),
                ci_high=round(ci_high, 3),
                n_direct=direct_n,
                n_pooled=pooled_n,
                tag_baseline=round(baseline, 3),
                value_added=round(hit_rate - baseline, 3),
                top_neighbors=[(n, round(w, 2)) for n, w in top],
            )

            w = alpha + beta
            weighted_sum += hit_rate * w
            weight_sum += w

        return {
            "mean": round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.5,
            "mechanisms_used": list(diagnostics.keys()),
            "diagnostics": {
                name: {
                    "hit_rate": d.hit_rate,
                    "ci": [d.ci_low, d.ci_high],
                    "n_direct": d.n_direct,
                    "n_pooled": d.n_pooled,
                    "value_added": d.value_added,
                    "reliable": d.reliable,
                    "top_neighbors": d.top_neighbors,
                }
                for name, d in diagnostics.items()
            },
        }

    def diagnostics(self) -> str:
        """Human-readable report."""
        lines = [
            "=== Graph-Aware Calibration Diagnostics ===",
            f"Total forecasts: {self._n_total}",
            f"Mechanisms with stats: {len(self._direct)}",
            "",
            f"{'Mechanism':<45} {'Dir':>4} {'Pool':>5} {'Hit':>6} {'CI':>16} {'Δbase':>7}",
            "-" * 90,
        ]

        sorted_mechs = sorted(
            self._direct.items(),
            key=lambda x: x[1]["yes"] + x[1]["no"],
            reverse=True,
        )

        for name, s in sorted_mechs:
            direct_n = s["yes"] + s["no"]
            if direct_n == 0:
                continue

            py, pn = self._pooled_stats(name)
            pooled_n = py + pn
            alpha = 1 + py
            beta = 1 + pn
            hit = alpha / (alpha + beta) if (alpha + beta) > 2 else 0.5
            lo = _beta_ppf(alpha, beta, 0.05)
            hi = _beta_ppf(alpha, beta, 0.95)
            baseline = self._tag_baseline(name)

            lines.append(
                f"  {name:<43} {direct_n:>4} {pooled_n:>5} {hit:.3f}  "
                f"[{lo:.3f}, {hi:.3f}]  {hit - baseline:+.3f}"
            )

        return "\n".join(lines)


def _beta_ppf(alpha: float, beta: float, q: float) -> float:
    if q <= 0: return 0.0
    if q >= 1: return 1.0
    import math
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
