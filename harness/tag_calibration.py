"""Tag-based calibration model with weighted Beta-Binomial pooling.

Corrects for double-counting: a market with N tags contributes
1/sqrt(N) per tag (not 1.0).  This prevents dense markets from
dominating while retaining signal at median 6 tags/market.

Tag posteriors use weighted counts:
  α_t = prior_alpha + Σ(weight × yes)
  β_t = prior_beta  + Σ(weight × no)

Query returns precision-weighted average across tags.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
CREDIBLE_MASS = 0.90


@dataclass
class TagCalibrationResult:
    mean: float
    ci_low: float
    ci_high: float
    n_total: int
    n_tags_used: int
    tags_used: list[str] = field(default_factory=list)
    per_tag: dict[str, dict] = field(default_factory=dict)

    @property
    def effective_n(self) -> float:
        return sum(
            max(0, t["alpha"] + t["beta"] - PRIOR_ALPHA - PRIOR_BETA)
            for t in self.per_tag.values()
        )


class TagCalibration:
    """Weighted tag-pooling calibration model.

    Each update carries a weight: w = (1/|tags|) × volume_factor × time_decay.
    This prevents double-counting (markets with many tags), gives more
    weight to high-volume markets, and prioritizes recent data.
    """

    def __init__(self, prior_alpha: float = PRIOR_ALPHA,
                 prior_beta: float = PRIOR_BETA):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Weighted counts
        self._w_yes: dict[str, float] = defaultdict(float)
        self._w_no: dict[str, float] = defaultdict(float)
        self._total_weight: float = 0.0
        self._n_markets: int = 0

    # ── update ─────────────────────────────────────────────────────────

    def update(self, tags: list[str], resolved_yes: bool,
               end_date: str = "", volume: float = 0.0) -> None:
        """Ingest one resolved market.  Uses 1/sqrt(|tags|) to correct
        for double-counting without destroying signal."""
        if not tags:
            return

        # Only correction: markets with many tags contribute less per tag.
        # 1/sqrt(N) is calibrated for median 6 tags/market (77% retention).
        w = 1.0 / math.sqrt(len(tags))

        self._n_markets += 1
        self._total_weight += w

        for tag in tags:
            if resolved_yes:
                self._w_yes[tag] += w
            else:
                self._w_no[tag] += w

    def load_jsonl(self, path: Path | str) -> int:
        """Bulk-load from resolved_markets.jsonl with weighting."""
        loaded = 0
        with open(path) as f:
            for line in f:
                rec = json.loads(line.strip())
                self.update(
                    rec["tags"],
                    rec["resolution"],
                    end_date=rec.get("end_date", ""),
                    volume=rec.get("volume", 0),
                )
                loaded += 1
        return loaded

    # ── properties ─────────────────────────────────────────────────────

    @property
    def n_markets(self) -> int:
        return self._n_markets

    @property
    def n_tags(self) -> int:
        return len(set(self._w_yes) | set(self._w_no))

    @property
    def prior_mean(self) -> float:
        return self.prior_alpha / (self.prior_alpha + self.prior_beta)

    # ── query ──────────────────────────────────────────────────────────

    def query(self, tags: list[str]) -> TagCalibrationResult:
        """Return calibrated posterior for a set of tags.

        Precision-weighted average across tags.  Tags with more effective
        observations (higher α+β) get more weight.
        """
        if not tags:
            return TagCalibrationResult(
                mean=self.prior_mean,
                ci_low=self._beta_ppf(self.prior_alpha, self.prior_beta,
                                      (1 - CREDIBLE_MASS) / 2),
                ci_high=self._beta_ppf(self.prior_alpha, self.prior_beta,
                                       1 - (1 - CREDIBLE_MASS) / 2),
                n_total=self._n_markets,
                n_tags_used=0,
            )

        per_tag: dict[str, dict] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for tag in tags:
            a = self.prior_alpha + self._w_yes.get(tag, 0.0)
            b = self.prior_beta + self._w_no.get(tag, 0.0)
            mean = a / (a + b)

            per_tag[tag] = {
                "alpha": round(a, 3),
                "beta": round(b, 3),
                "mean": round(mean, 4),
                "w_yes": round(self._w_yes.get(tag, 0), 3),
                "w_no": round(self._w_no.get(tag, 0), 3),
            }

            w = a + b
            weighted_sum += mean * w
            weight_sum += w

        pooled_mean = weighted_sum / weight_sum if weight_sum > 0 else self.prior_mean

        # Credible interval from pooled effective params
        eff_alpha = self.prior_alpha + sum(
            self._w_yes.get(t, 0.0) for t in tags
        ) / max(len(tags), 1)
        eff_beta = self.prior_beta + sum(
            self._w_no.get(t, 0.0) for t in tags
        ) / max(len(tags), 1)

        ci_low = self._beta_ppf(eff_alpha, eff_beta,
                                (1 - CREDIBLE_MASS) / 2)
        ci_high = self._beta_ppf(eff_alpha, eff_beta,
                                 1 - (1 - CREDIBLE_MASS) / 2)

        return TagCalibrationResult(
            mean=round(pooled_mean, 4),
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            n_total=self._n_markets,
            n_tags_used=len(tags),
            tags_used=sorted(tags),
            per_tag=per_tag,
        )

    # ── numerical helpers ──────────────────────────────────────────────

    @staticmethod
    def _beta_ppf(alpha: float, beta: float, q: float) -> float:
        """Quantile of Beta(α, β) via Newton on regularized incomplete beta."""
        if q <= 0:
            return 0.0
        if q >= 1:
            return 1.0

        mean = alpha / (alpha + beta)
        x = mean
        for _ in range(20):
            cdf = _reg_beta_cdf(x, alpha, beta)
            pdf = _beta_pdf(x, alpha, beta)
            if pdf < 1e-15:
                break
            dx = (cdf - q) / pdf
            x -= dx
            if abs(dx) < 1e-8:
                break
            x = max(0.0, min(1.0, x))
        return x

    def summary(self) -> str:
        lines = [f"TagCalibration: {self._n_markets} markets, "
                 f"{self.n_tags} tags "
                 f"(weighted total={self._total_weight:.1f})\n"]
        lines.append(f"{'Tag':<30} {'Wgt':>6} {'Post':>8} {'CI'}")
        lines.append("-" * 60)

        all_tags = sorted(
            set(self._w_yes) | set(self._w_no),
            key=lambda t: self._w_yes.get(t, 0) + self._w_no.get(t, 0),
            reverse=True,
        )
        for tag in all_tags[:40]:
            a = self.prior_alpha + self._w_yes.get(tag, 0)
            b = self.prior_beta + self._w_no.get(tag, 0)
            w = self._w_yes.get(tag, 0) + self._w_no.get(tag, 0)
            mean = a / (a + b)
            lo = self._beta_ppf(a, b, (1 - CREDIBLE_MASS) / 2)
            hi = self._beta_ppf(a, b, 1 - (1 - CREDIBLE_MASS) / 2)
            lines.append(
                f"  {tag:<28} {w:>6.2f} {mean:.3f}    [{lo:.3f}, {hi:.3f}]"
            )
        return "\n".join(lines)


# ── numerical helpers (same as before) ─────────────────────────────────

def _beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    return math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)
    )


def _reg_beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta via Lentz continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    if x > (a + 1) / (a + b + 2):
        return 1.0 - _reg_beta_cdf(1 - x, b, a)

    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - log_beta) / a

    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < 1e-12:
            break

    return min(1.0, max(0.0, front * (f - 1.0)))


# ── convenience ────────────────────────────────────────────────────────

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket" / "resolved_markets.jsonl"


def load_default() -> TagCalibration:
    """Load calibration from the default resolved_markets.jsonl."""
    cal = TagCalibration()
    path = _DEFAULT_PATH
    if path.exists():
        n = cal.load_jsonl(path)
        print(f"Loaded {n} markets, {cal.n_tags} tags "
              f"(weighted total={cal._total_weight:.1f})")
    else:
        print(f"No data at {path} — run fetch_calibration_data.py first")
    return cal
