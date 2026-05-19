"""PIT calibration probes: graph forecasts vs Polymarket YES price at cutoff."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from ingest.polymarket_price_at import yes_price_at_cutoff_with_retry

ProbeKind = Literal["graph", "market_anchor", "vault_stance"]
ProbeDomain = Literal["geopolitics", "meta", "institutions", "economics", "culture", "other"]

DEFAULT_MARKET_CALIBRATION_BAND = 0.05

_GEO_SLUG_HINTS = (
    "israel", "iran", "hamas", "gaza", "russia", "ukraine", "nato", "ceasefire",
    "war", "strike", "hezbollah", "houthi", "syria", "lebanon", "taiwan", "china",
    "congress", "senate", "election", "trump", "biden", "shutdown", "invasion",
)
_EXCLUDE_SLUG_HINTS = (
    "fed", "fomc", "bps", "interest-rate", "bitcoin", "btc", "eth", "solana", "xrp",
    "doge", "tennis", "nba", "nfl", "ufc", "mlb", "grammy", "billboard", "spotify",
    "netflix", "musk-post", "tweet", "assists", "o-u", "halftime", "completed-match",
    "price-of-", "above-", "below-", "dip-to",
    "chamber-of-dep", "hold-the-most-seats", "win-the-most-seats",
)


@dataclass(frozen=True)
class MarketProbeSpec:
    probe_id: str
    cutoff: date
    question: str
    kind: ProbeKind = "graph"
    domain: ProbeDomain = "geopolitics"
    polymarket_slug: str | None = None
    clob_yes_token_id: str | None = None
    market_yes_at_cutoff: float | None = None
    vault_target_p_yes: float | None = None
    resolution: bool | None = None
    vault_anchors: tuple[str, ...] = ()
    graph_question: str | None = None
    notes: str = ""

    def effective_question(self) -> str:
        return (self.graph_question or self.question).strip()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cutoff"] = self.cutoff.isoformat()
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MarketProbeSpec:
        cutoff_raw = raw.get("cutoff")
        if isinstance(cutoff_raw, str):
            cutoff = date.fromisoformat(cutoff_raw[:10])
        else:
            raise ValueError(f"probe {raw.get('probe_id')}: missing cutoff")
        anchors = raw.get("vault_anchors") or ()
        return cls(
            probe_id=str(raw["probe_id"]),
            cutoff=cutoff,
            question=str(raw.get("question") or ""),
            kind=raw.get("kind", "graph"),
            domain=raw.get("domain", "geopolitics"),
            polymarket_slug=raw.get("polymarket_slug"),
            clob_yes_token_id=raw.get("clob_yes_token_id"),
            market_yes_at_cutoff=raw.get("market_yes_at_cutoff"),
            vault_target_p_yes=raw.get("vault_target_p_yes"),
            resolution=raw.get("resolution"),
            vault_anchors=tuple(anchors),
            graph_question=raw.get("graph_question"),
            notes=str(raw.get("notes") or ""),
        )


@dataclass
class MarketProbeResult:
    spec: MarketProbeSpec
    p_yes: float
    reasoning: str
    market_yes_at_cutoff: float | None
    market_abs_error: float | None
    market_brier: float | None
    vault_abs_error: float | None
    resolution_brier: float | None
    within_band: bool | None
    librarian_conjuncture: str = ""
    librarian_uncertainties: list[str] = field(default_factory=list)
    librarian_excluded: list[str] = field(default_factory=list)
    librarian_sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.spec.probe_id,
            "cutoff": self.spec.cutoff.isoformat(),
            "kind": self.spec.kind,
            "domain": self.spec.domain,
            "question": self.spec.effective_question(),
            "p_yes": self.p_yes,
            "market_yes_at_cutoff": self.market_yes_at_cutoff,
            "vault_target_p_yes": self.spec.vault_target_p_yes,
            "market_abs_error": self.market_abs_error,
            "market_brier": self.market_brier,
            "vault_abs_error": self.vault_abs_error,
            "resolution_brier": self.resolution_brier,
            "within_band": self.within_band,
            "resolution": self.spec.resolution,
            "polymarket_slug": self.spec.polymarket_slug,
            "librarian_conjuncture": self.librarian_conjuncture,
            "librarian_uncertainties": self.librarian_uncertainties,
            "librarian_excluded": self.librarian_excluded,
            "librarian_sources": self.librarian_sources,
            "reasoning": self.reasoning,
            "errors": self.errors,
        }


def load_catalog(path: Path) -> list[MarketProbeSpec]:
    if not path.is_file():
        return []
    specs: list[MarketProbeSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            specs.append(MarketProbeSpec.from_dict(json.loads(line)))
    return specs


def write_catalog(path: Path, specs: list[MarketProbeSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(s.to_dict(), ensure_ascii=False) for s in specs) + "\n",
        encoding="utf-8",
    )


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def completed_probe_ids(path: Path) -> set[str]:
    return {r["probe_id"] for r in load_results(path) if r.get("probe_id")}


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _is_geopolitics_slug(slug: str, question: str) -> bool:
    blob = f"{slug} {question}".lower()
    if any(x in blob for x in _EXCLUDE_SLUG_HINTS):
        return False
    return any(x in blob for x in _GEO_SLUG_HINTS)


def _quarter_end_before(dt: datetime, n: int = 1) -> date:
    """Last day of the quarter ``n`` quarters before ``dt`` (UTC)."""
    q = (dt.month - 1) // 3 + 1
    year, month = dt.year, dt.month
    for _ in range(n):
        q -= 1
        if q < 1:
            q = 4
            year -= 1
    if q == 1:
        return date(year, 3, 31)
    if q == 2:
        return date(year, 6, 30)
    if q == 3:
        return date(year, 9, 30)
    return date(year, 12, 31)


def market_calibration_cutoff(record: dict[str, object]) -> date | None:
    """Pick a cutoff while the market is open and CLOB history likely exists."""
    start = _parse_iso_datetime(
        record.get("start_date") or record.get("startDate") or record.get("created_at") or record.get("createdAt")
    )
    end = _parse_iso_datetime(record.get("end_date") or record.get("endDate"))
    if start is None:
        return None
    # Mid-life sample: ~30 days after open, or one quarter forward, capped before close.
    candidate = (start + timedelta(days=30)).date()
    if end is not None:
        last = (end - timedelta(days=7)).date()
        if candidate > last:
            candidate = last
    if candidate < start.date():
        candidate = start.date()
    return candidate


def _parse_iso_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def graph_question_from_market(question: str) -> str:
    """Shorter graph-native framing; resolution criteria stay in PM slug anchor."""
    q = question.strip()
    q = re.sub(r"\s+", " ", q)
    return (
        f"{q}\n\n"
        "Forecast using only the PIT knowledge graph at cutoff. "
        "Your probability should match what a well-calibrated Polymarket YES price "
        "would be at this date — not a post-hoc resolution guess."
    )


def seed_from_gold_dataset(
    gold_path: Path,
    *,
    max_probes: int = 12,
) -> list[MarketProbeSpec]:
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    specs: list[MarketProbeSpec] = []
    for case in cases:
        if len(specs) >= max_probes:
            break
        record = case.get("record") or {}
        slug = str(record.get("slug") or "")
        question = str(record.get("question") or "").strip()
        if not slug or not question:
            continue
        if not _is_geopolitics_slug(slug, question):
            continue
        cutoff = market_calibration_cutoff(record)
        if cutoff is None:
            continue
        tokens = record.get("clob_token_ids") or record.get("clobTokenIds")
        token_id = None
        if isinstance(tokens, list) and tokens:
            token_id = str(tokens[0])
        case_id = str(case.get("case_id") or slug)
        specs.append(
            MarketProbeSpec(
                probe_id=f"gold-{case_id}",
                cutoff=cutoff,
                question=question,
                graph_question=graph_question_from_market(question),
                kind="market_anchor",
                domain="geopolitics",
                polymarket_slug=slug,
                clob_yes_token_id=token_id,
                notes=f"Seeded from {case_id}; cutoff=mid-market sample for CLOB alignment",
            )
        )
    return specs


def resolve_market_price(spec: MarketProbeSpec) -> tuple[float | None, list[str]]:
    errors: list[str] = []
    if spec.market_yes_at_cutoff is not None:
        p = float(spec.market_yes_at_cutoff)
        if 0.0 <= p <= 1.0:
            return p, errors
        errors.append("market_yes_at_cutoff out of range")
    if spec.polymarket_slug is None and spec.clob_yes_token_id is None:
        return None, errors
    price, _tid = yes_price_at_cutoff_with_retry(
        slug=spec.polymarket_slug,
        token_id=spec.clob_yes_token_id,
        cutoff=spec.cutoff,
    )
    if price is None:
        errors.append("no CLOB price at or before cutoff")
    return price, errors


def score_probe(
    p_yes: float,
    spec: MarketProbeSpec,
    market_yes: float | None,
    *,
    band: float = DEFAULT_MARKET_CALIBRATION_BAND,
) -> MarketProbeResult:
    market_abs = abs(p_yes - market_yes) if market_yes is not None else None
    market_brier = (p_yes - market_yes) ** 2 if market_yes is not None else None
    vault_abs = (
        abs(p_yes - spec.vault_target_p_yes)
        if spec.vault_target_p_yes is not None
        else None
    )
    resolution_brier = None
    if spec.resolution is not None:
        resolution_brier = (p_yes - (1.0 if spec.resolution else 0.0)) ** 2
    within = market_abs <= band if market_abs is not None else None
    return MarketProbeResult(
        spec=spec,
        p_yes=p_yes,
        reasoning="",
        market_yes_at_cutoff=market_yes,
        market_abs_error=market_abs,
        market_brier=market_brier,
        vault_abs_error=vault_abs,
        resolution_brier=resolution_brier,
        within_band=within,
    )


def build_forecast_prompt(
    spec: MarketProbeSpec,
    market_yes: float | None,
    *,
    band: float = DEFAULT_MARKET_CALIBRATION_BAND,
) -> str:
    lines = [
        spec.effective_question(),
        "",
        f"Cutoff (strict PIT): {spec.cutoff.isoformat()}",
        "Use graph-vault only. Do not use web search.",
        "",
        "=== POLYMARKET CALIBRATION MODE (Rule 9) ===",
        "Read graph-vault/_forecast_instructions.md Rule 9 before forecasting.",
    ]
    if market_yes is not None:
        lines += [
            f"Polymarket YES at cutoff: {market_yes:.4f}",
            f"Target: p_yes within ±{band:.2f} of {market_yes:.4f} unless vault shows strong "
            "pre-cutoff evidence the market was structurally mispriced.",
            "Do NOT set p_yes≈1.0 merely because a later vault paragraph describes an outcome — "
            "the market price encodes trader information at cutoff.",
        ]
    elif spec.vault_target_p_yes is not None:
        lines.append(
            f"No Polymarket contract; vault conjuncture stance ≈ {spec.vault_target_p_yes:.2f}."
        )
    if spec.vault_anchors:
        lines.append("Priority vault paths: " + ", ".join(spec.vault_anchors))
    return "\n".join(lines)


def run_market_probe(
    spec: MarketProbeSpec,
    *,
    vault_dir: Path,
    band: float = DEFAULT_MARKET_CALIBRATION_BAND,
    use_pit_librarian: bool = True,
) -> MarketProbeResult:
    """Two-step calibration: PIT librarian brief, then forecaster (both logged for reflect)."""
    from harness.orchestrator import run_structured
    from harness.pit_research import run_pit_research

    market_yes, price_errors = resolve_market_price(spec)
    question = build_forecast_prompt(spec, market_yes, band=band)

    brief_block = ""
    lib_conjuncture = ""
    lib_uncertainties: list[str] = []
    lib_excluded: list[str] = []
    lib_sources: list[str] = []
    pit_tmp = None

    if use_pit_librarian:
        try:
            brief, pit_tmp = run_pit_research(
                spec.effective_question(),
                spec.cutoff,
                vault_dir=vault_dir,
                market_yes_at_cutoff=market_yes,
                use_snapshot=True,
            )
            brief_block = brief.to_prompt_block()
            lib_conjuncture = brief.conjuncture
            lib_uncertainties = list(brief.uncertainties)
            lib_excluded = list(brief.excluded_as_post_cutoff)
            lib_sources = list(brief.sources)
        except Exception as exc:
            price_errors = price_errors + [f"librarian: {exc}"]
        finally:
            if pit_tmp is not None:
                pit_tmp.cleanup()

    try:
        p_yes, reasoning = run_structured(
            question,
            cutoff=spec.cutoff,
            vault_dir=vault_dir,
            enforce_pit=True,
            graph_only=True,
            use_pit_librarian=False,
            pit_brief_block=brief_block,
            category=spec.domain,
            question_id=spec.probe_id,
            resolution=spec.resolution,
        )
    except Exception as exc:
        result = score_probe(0.5, spec, market_yes, band=band)
        result.errors = price_errors + [f"forecaster: {exc}"]
        result.librarian_conjuncture = lib_conjuncture
        result.librarian_uncertainties = lib_uncertainties
        result.librarian_excluded = lib_excluded
        result.librarian_sources = lib_sources
        return result

    result = score_probe(p_yes, spec, market_yes, band=band)
    result.reasoning = reasoning
    result.errors = price_errors
    result.librarian_conjuncture = lib_conjuncture
    result.librarian_uncertainties = lib_uncertainties
    result.librarian_excluded = lib_excluded
    result.librarian_sources = lib_sources
    if result.market_abs_error is not None:
        result.within_band = result.market_abs_error <= band
    return result


def _classify_calibration_miss(row: dict[str, Any], *, band: float) -> str:
    """Diagnostic label for reflection — no post-cutoff outcome narrative."""
    p = float(row.get("p_yes", 0.5))
    mkt = row.get("market_yes_at_cutoff")
    if mkt is None:
        return "no_market_anchor"
    mkt_f = float(mkt)
    if p > mkt_f + band and p >= 0.85:
        return "forecaster_too_high_vs_market (likely treated outcome as certain or vault post-hoc narrative)"
    if p > mkt_f + band:
        return "forecaster_above_market (check vault bullets after cutoff; add pit_body_cutoff / trim)"
    if p < mkt_f - band:
        return "forecaster_below_market (missing trigger-density or PM underweighted)"
    return "outside_band_other"


def format_market_calibration_feedback(
    results_path: Path,
    *,
    band: float = DEFAULT_MARKET_CALIBRATION_BAND,
    max_rows: int = 15,
    for_reflect: bool = False,
) -> str:
    """Summarize librarian + forecaster calibration misses for pit_reflect.

    When for_reflect=True, omit forecaster_reasoning and librarian_conjuncture text
    that often states post-cutoff outcomes — reflection must fix PIT boundaries and
    market-alignment discipline, not transcribe what happened.
    """
    rows = load_results(results_path)
    if not rows:
        return "(no market calibration results yet)"
    lines: list[str] = [
        "Reflect on BOTH stages: (1) pit-research-librarian retrieval/leakage, "
        "(2) forecaster p_yes vs Polymarket at cutoff.",
        "Fix vault conjuncture/PIT snapshot — not only forecaster wording.",
        "",
    ]
    if for_reflect:
        lines += [
            "=== REFLECTION ANTI-LEAKAGE (mandatory) ===",
            "Do NOT fix misses by documenting terminal outcomes in threads/concepts/timeline.",
            "Do NOT add 'X happened on date Y' to vault docs when the miss is miscalibration vs PM at cutoff.",
            "Allowed fixes: pit_body_cutoff, truncate post-cutoff bullets, separate PIT conjuncture files,",
            "Rule 9 wording, librarian retrieval rules, mechanism gaps — align structure with PM at T.",
            "",
        ]
    summary = summarize_results(rows, band=band)
    pt = int(round(band * 100))
    lines.append(
        f"Market calibration: {summary.get('n_with_market', 0)} with PM price, "
        f"mean MAE={summary.get('mean_market_abs_error', 0):.3f}, "
        f"within ±{pt}pt={summary.get('pct_within_band', 0):.1f}%"
    )
    misses = [
        r for r in rows
        if r.get("market_abs_error") is not None and r["market_abs_error"] > band
    ]
    misses.sort(key=lambda r: float(r.get("market_abs_error") or 0), reverse=True)
    for r in misses[:max_rows]:
        q = (r.get("question") or "")[:120].replace("\n", " ")
        lines.append(
            f"\n  MISS {r['probe_id'][:50]} cutoff={r['cutoff']} "
            f"forecaster_p={r['p_yes']:.3f} market={r.get('market_yes_at_cutoff')} "
            f"mae={r['market_abs_error']:.3f}"
        )
        if q:
            lines.append(f"    question: {q}")
        if for_reflect:
            lines.append(f"    diagnosis: {_classify_calibration_miss(r, band=band)}")
            unc = r.get("librarian_uncertainties") or []
            if unc:
                lines.append(f"    librarian_uncertainties: {'; '.join(str(u) for u in unc[:4])}")
            excl = r.get("librarian_excluded") or []
            if excl:
                lines.append(f"    librarian_excluded: {'; '.join(str(x) for x in excl[:3])}")
            src = r.get("librarian_sources") or []
            if src:
                lines.append(f"    librarian_sources: {', '.join(str(s) for s in src[:6])}")
        else:
            lib = (r.get("librarian_conjuncture") or "")[:300]
            if lib:
                lines.append(f"    librarian_conjuncture: {lib}")
            unc = r.get("librarian_uncertainties") or []
            if unc:
                lines.append(f"    librarian_uncertainties: {'; '.join(str(u) for u in unc[:4])}")
            excl = r.get("librarian_excluded") or []
            if excl:
                lines.append(f"    librarian_excluded: {'; '.join(str(x) for x in excl[:3])}")
            lines.append(f"    forecaster_reasoning: {(r.get('reasoning') or '')[:250]}")
    vault_misses = [
        r for r in rows
        if r.get("vault_abs_error") is not None and r["vault_abs_error"] > band
    ]
    for r in vault_misses[:5]:
        lines.append(
            f"\n  VAULT_MISS {r['probe_id']} forecaster_p={r['p_yes']:.3f} "
            f"vault_target={r.get('vault_target_p_yes')} err={r['vault_abs_error']:.3f}"
        )
        if not for_reflect:
            lib = (r.get("librarian_conjuncture") or "")[:200]
            if lib:
                lines.append(f"    librarian_conjuncture: {lib}")
    return "\n".join(lines)


def summarize_results(
    rows: list[dict[str, Any]],
    *,
    band: float = DEFAULT_MARKET_CALIBRATION_BAND,
) -> dict[str, Any]:
    market_rows = [r for r in rows if r.get("market_yes_at_cutoff") is not None]
    vault_rows = [r for r in rows if r.get("vault_target_p_yes") is not None]
    res_rows = [r for r in rows if r.get("resolution_brier") is not None]

    def _mean(key: str, subset: list[dict[str, Any]]) -> float | None:
        vals = [float(r[key]) for r in subset if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    within = [r for r in market_rows if r.get("within_band")]
    return {
        "n_total": len(rows),
        "n_with_market": len(market_rows),
        "n_with_vault_target": len(vault_rows),
        "mean_market_abs_error": _mean("market_abs_error", market_rows),
        "mean_market_brier": _mean("market_brier", market_rows),
        "calibration_band": band,
        "pct_within_band": (len(within) / len(market_rows) * 100) if market_rows else None,
        "mean_vault_abs_error": _mean("vault_abs_error", vault_rows),
        "mean_resolution_brier": _mean("resolution_brier", res_rows),
    }


DEFAULT_CATALOG = Path("data/pit_market_probes/catalog.jsonl")
DEFAULT_RESULTS = Path("data/pit_market_probes/results.jsonl")
