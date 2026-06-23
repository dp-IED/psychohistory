"""Calibration model trained on agent reasoning features.

Unlike TagCalibration (which trains on Polymarket's tags), this model
trains on the agent's own reasoning trail: which mechanisms it invoked,
which vault threads it consulted, which concepts it referenced.

This enables answering: "does the agent's analysis actually add value
over the base rate from Polymarket tags?"
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from harness.reasoning_trail import ReasoningTrail, parse_all_runs


@dataclass
class MechanismDiagnostic:
    """Per-mechanism calibration stats."""
    name: str
    hit_rate: float        # posterior mean
    ci_low: float
    ci_high: float
    n_resolved: int
    n_yes: int
    avg_p_yes: float       # what the agent predicted when invoking this
    avg_brier: float | None
    value_added: float     # hit_rate minus the tag-based base rate

    @property
    def reliable(self) -> bool:
        return self.n_resolved >= 3


@dataclass
class ReasoningCalibrationResult:
    mean: float
    mechanisms_used: list[str]
    per_mechanism: dict[str, MechanismDiagnostic] = field(default_factory=dict)
    tag_baseline: float = 0.0


class ReasoningCalibration:
    """Calibration model trained on agent reasoning features.

    For each mechanism m that the agent invokes:
      hit_rate(m) = P(resolution=YES | mechanism m was invoked)

    A forecast's calibrated probability is the weighted average of
    mechanism hit rates for the mechanisms the agent used.
    """

    def __init__(self):
        # Per-mechanism stats: {name: {yes: N, no: N, p_yes_sum: float, brier_sum: float}}
        self._mech: dict[str, dict] = defaultdict(
            lambda: {"yes": 0, "no": 0, "p_yes_sum": 0.0, "brier_sum": 0.0, "invoked": 0}
        )
        self._n_total = 0
        self._tag_baselines: dict[str, float] = {}  # mechanism → tag base rate

    def update(self, trail: ReasoningTrail,
               tag_baseline: float | None = None) -> None:
        """Ingest one forecast with its reasoning trail."""
        self._n_total += 1
        resolved = trail.resolution
        if resolved is None:
            return  # skip unresolved

        for mech_raw in trail.mechanisms:
            name = _normalize_mechanism(mech_raw)
            self._mech[name]["invoked"] += 1

            if resolved:
                self._mech[name]["yes"] += 1
            else:
                self._mech[name]["no"] += 1

            self._mech[name]["p_yes_sum"] += trail.p_yes
            if trail.brier is not None:
                self._mech[name]["brier_sum"] += trail.brier

            # Record tag baseline for this mechanism
            if tag_baseline is not None and name not in self._tag_baselines:
                self._tag_baselines[name] = tag_baseline

    def load_trails(self, vault_dir: str | Path,
                    resolutions: dict[str, bool] | None = None) -> int:
        """Parse all run notes and train on resolved ones."""
        trails = parse_all_runs(vault_dir, resolutions=resolutions)
        resolved = [t for t in trails if t.resolution is not None]
        for t in resolved:
            self.update(t)
        return len(resolved)

    def query(self, mechanisms: list[str]) -> ReasoningCalibrationResult:
        """Calibrate using the mechanisms the agent invoked."""
        if not mechanisms:
            return ReasoningCalibrationResult(
                mean=0.5, mechanisms_used=[],
            )

        per_mech: dict[str, MechanismDiagnostic] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for mech_raw in mechanisms:
            name = _normalize_mechanism(mech_raw)
            stats = self._mech.get(name, {})
            n = stats.get("yes", 0) + stats.get("no", 0)

            # Beta posterior with prior 1,1
            alpha = 1 + stats.get("yes", 0)
            beta = 1 + stats.get("no", 0)
            hit_rate = alpha / (alpha + beta)

            ci_low = _beta_ppf(alpha, beta, 0.05)
            ci_high = _beta_ppf(alpha, beta, 0.95)

            avg_p = stats.get("p_yes_sum", 0) / max(n, 1)
            avg_brier = stats.get("brier_sum", 0) / max(n, 1) if n > 0 else None

            tag_base = self._tag_baselines.get(name, 0.5)
            value_added = hit_rate - tag_base

            per_mech[name] = MechanismDiagnostic(
                name=name,
                hit_rate=round(hit_rate, 3),
                ci_low=round(ci_low, 3),
                ci_high=round(ci_high, 3),
                n_resolved=n,
                n_yes=stats.get("yes", 0),
                avg_p_yes=round(avg_p, 3),
                avg_brier=round(avg_brier, 4) if avg_brier else None,
                value_added=round(value_added, 3),
            )

            # Weight by precision (α+β)
            w = alpha + beta
            weighted_sum += hit_rate * w
            weight_sum += w

        pooled = weighted_sum / weight_sum if weight_sum > 0 else 0.5

        return ReasoningCalibrationResult(
            mean=round(pooled, 4),
            mechanisms_used=[_normalize_mechanism(m) for m in mechanisms],
            per_mechanism=per_mech,
            tag_baseline=sum(
                d.value_added for d in per_mech.values()
            ) / max(len(per_mech), 1),
        )

    def diagnostics(self) -> str:
        """Human-readable diagnostic report."""
        lines = [
            "=== Agent Reasoning Diagnostic Report ===",
            f"Total forecasts tracked: {self._n_total}",
            f"Mechanisms tracked: {len(self._mech)}",
            "",
            f"{'Mechanism':<45} {'N':>4} {'Hit':>6} {'CI':>16} {'Agent_p':>8} {'Δbase':>7}",
            "-" * 92,
        ]

        sorted_mechs = sorted(
            self._mech.items(),
            key=lambda x: x[1]["yes"] + x[1]["no"],
            reverse=True,
        )

        for name, s in sorted_mechs:
            n = s["yes"] + s["no"]
            if n == 0:
                continue
            alpha = 1 + s["yes"]
            beta = 1 + s["no"]
            hit = alpha / (alpha + beta)
            lo = _beta_ppf(alpha, beta, 0.05)
            hi = _beta_ppf(alpha, beta, 0.95)
            avg_p = s["p_yes_sum"] / n if n > 0 else 0
            tag_base = self._tag_baselines.get(name, 0.5)
            delta = hit - tag_base
            reliable = "✓" if n >= 3 else "?"

            lines.append(
                f"  {name:<43} {n:>4} {hit:.3f}  "
                f"[{lo:.3f}, {hi:.3f}]  {avg_p:.3f}  "
                f"{delta:+.3f} {reliable}"
            )

        return "\n".join(lines)


def _normalize_mechanism(raw: str) -> str:
    """Normalize a mechanism name from the trail."""
    # "Public-framework-announcement-commitment: superpower patron's..."
    name = raw.split(':')[0].strip()
    name = name.split('(')[0].strip()
    name = name.lower().replace(' ', '-')[:60]
    return name


def _beta_ppf(alpha: float, beta: float, q: float) -> float:
    """Quantile of Beta(α, β)."""
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
    f, c, d = 1.0, 1.0, 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, 200):
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        f *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < 1e-12: break
    return min(1.0, max(0.0, front * (f - 1.0)))
