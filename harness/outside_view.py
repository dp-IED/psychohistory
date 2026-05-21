"""Outside-view anchoring: base rates from resolved cases + Polymarket prices.

Provides the Stage 1 "anchor first, adjust later" step of the cognitive
forecasting pipeline. Deterministic Python — no LLM, sub-second.

Now supports all output types: binary (p_yes), numeric (value + CI),
categorical (distribution over choices), discrete (ordered distribution).

Usage:
    from harness.outside_view import get_outside_view_anchor

    anchor = get_outside_view_anchor(
        question="Will X happen before Y?",
        vault_dir="graph-vault",
    )
    # anchor.output_type → "binary" | "numeric" | "categorical" | "discrete"
    # anchor.binary → BaseRateResult (if binary)
    # anchor.numeric → NumericAnchor (if numeric)
    # anchor.categorical → CategoricalAnchor (if categorical)
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OutputType(str, Enum):
    BINARY = "binary"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"


# ── Domain / event type classification ───────────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "geopolitics": [
        "war", "conflict", "invasion", "military", "nuclear", "missile",
        "nato", "ukraine", "russia", "china", "iran", "israel", "gaza",
        "hamas", "ceasefire", "truce", "peace", "sanction", "taiwan",
        "hezbollah", "houthi", "syria", "iraq", "afghanistan", "opec",
    ],
    "politics": [
        "election", "president", "vote", "parliament", "congress",
        "senate", "minister", "party", "government", "impeach",
        "biden", "trump", "nominee", "candidate", "ballot",
        "democratic", "republican", "supreme court", "governor",
        "mayor", "referendum", "duterte", "pope", "bill", "legislation",
        "senate", "remove", "representative", "ambassador",
    ],
    "economics": [
        "gdp", "inflation", "cpi", "rate", "market", "price",
        "tariff", "trade", "recession", "fed", "stock", "bond",
        "currency", "debt", "interest rate", "fomc", "federal reserve",
        "etf", "bitcoin", "ethereum", "sec", "crypto", "treasury",
        "building permits", "permits",
    ],
    "technology": [
        "tiktok", "ai", "openai", "tesla", "musk",
        "regulation", "antitrust", "privacy", "data", "ban",
    ],
    "health": [
        "outbreak", "case", "disease", "virus", "infection",
        "epidemic", "hospital", "who", "cdc", "vaccine", "pandemic",
        "hantavirus", "covid", "ebola", "health",
    ],
    "culture": [
        "theatrical", "play", "tony", "oscar", "grammy", "album",
        "song", "film", "movie", "box office", "fifa", "world cup",
        "olympic", "sport", "game", "tournament",
    ],
}

EVENT_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("ceasefire", r"ceasefire|truce|peace\s+(deal|agreement|treaty)|halt\s+in\s+military"),
    ("election", r"election|presidential\s+race|win\s+the\s+most\s+seats|vote|ballot|referendum|senate.*remove|win.*election"),
    ("resignation", r"drop\s+out|resign|step\s+down|withdraw\s+from\s+(presidential\s+)?race"),
    ("rate_decision", r"fed(?:eral\s+reserve)?\s+(?:cuts?|raises?|decreases?|increases?)|interest\s+rate|fomc|bps"),
    ("regulatory_approval", r"sec\s+(?:approves?|rejects?|delays?)|etf\s+(?:begins?|trading|approved|rejected)|ban(?:ned)?\s+in\s+the\s+us|signed\s+into\s+law"),
    ("court_ruling", r"supreme\s+court|court\s+(?:ruling|decision|case)|trial|sentenced?|convict"),
    ("government_action", r"government\s+shutdown|debt\s+ceiling|budget|spending\s+bill"),
    ("military_strike", r"strike|attack|invasion|bomb(?:ing)?|military\s+(?:action|operation)"),
    ("appointment", r"nominat(?:ion|ed|ee)|appoint|confirm(?:ation|ed)|vp\s+(?:nominee|pick|selection)|high\s+representative|in\s+place|visit.*country"),
    ("numeric_indicator", r"score|percentage|how\s+many|what\s+will\s+be\s+the|index|rank"),
    ("macro_release", r"cpi|inflation|unemployment|gdp|jobs\s+report|economic\s+data"),
    ("health_outbreak", r"outbreak|case|disease|virus|infection|hantavirus|linked\s+to"),
    ("cultural_award", r"best\s+play|tony|award|win\s+.*award|grammy|oscar"),
    ("sports_outcome", r"fifa|world\s+cup|knockout\s+stage|weakest\s+team|advance\s+to"),
    ("other", r"."),
]

# Output type detection patterns
NUMERIC_PATTERNS = [
    r"what\s+will\s+be\s+the",
    r"how\s+many",
    r"what\s+percentage",
    r"score\s+for",
    r"score\s+in\s+20",
    r"index\s+score",
    r"price\s+of",
    r"rank\s+of",
    r"relative\s+rank",
]

CATEGORICAL_PATTERNS = [
    r"who\s+will\s+win",
    r"which\s+(?:theatrical|production|play|film|movie)",
    r"which\s+(?:candidate|party)",
    r"who\s+will\s+be",
]


STOP_WORDS = frozenset({
    "will", "there", "before", "after", "during", "this", "that",
    "with", "from", "into", "have", "been", "being", "market",
    "resolves", "resolution", "criteria", "shall", "question",
    "whether", "price", "above", "below", "yes", "no", "the",
    "and", "for", "of", "to", "in", "on", "by", "at", "a", "an",
    "is", "are", "was", "were", "be", "or", "not", "it", "its",
    "what", "who", "when", "where", "how", "which", "their",
    "they", "has", "had", "does", "did", "can", "could", "would",
    "should", "may", "might", "must", "first", "also", "other",
})


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class CaseMatch:
    """A single resolved case matched to the current question."""
    case_id: str
    question: str
    similarity: float
    resolution_bool: bool | None
    resolution_label: str
    event_type: str
    domain: str
    time_horizon_days: int | None
    reference_class_base_rate: float
    reference_class_size: int
    slug: str = ""


@dataclass
class BaseRateResult:
    """Base rate for binary questions."""
    event_type: str
    domain: str
    base_rate: float
    total_cases: int
    resolved_yes: int
    resolved_no: int
    matches: list[CaseMatch] = field(default_factory=list)
    confidence: str = "medium"


@dataclass
class NumericAnchor:
    """Reference class distribution for numeric questions.

    Currently no reference-class data (gold set is binary-only).
    Provides event_type/domain classification for downstream reasoning.
    """
    event_type: str
    domain: str
    matches: list[CaseMatch] = field(default_factory=list)
    note: str = "No numeric reference class data available — gold set is binary-only."


@dataclass
class CategoricalAnchor:
    """Reference class frequency distribution for categorical questions.

    Currently no reference-class data (gold set is binary-only).
    Provides event_type/domain classification and similar-case retrieval.
    """
    event_type: str
    domain: str
    matches: list[CaseMatch] = field(default_factory=list)
    note: str = "No categorical reference class data available — gold set is binary-only."


@dataclass
class PolymarketAnchor:
    """Live Polymarket price for an equivalent market."""
    price: float
    volume: float
    slug: str
    question: str
    liquidity: float = 0.0


@dataclass
class OutsideViewAnchor:
    """Complete outside-view anchor — polymorphic based on output_type."""
    output_type: OutputType
    event_type: str
    domain: str
    binary: BaseRateResult | None = None
    numeric: NumericAnchor | None = None
    categorical: CategoricalAnchor | None = None
    polymarket: PolymarketAnchor | None = None
    anchoring_strategy: str = ""


# ── Tokenization utilities ───────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {t for t in tokens if t not in STOP_WORDS}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


# ── Output type detection ────────────────────────────────────────────


def detect_output_type(question: str) -> OutputType:
    """Detect the output type from the question text."""
    q = question.lower()

    for pat in NUMERIC_PATTERNS:
        if re.search(pat, q):
            # "relative rank" is discrete, not pure numeric
            if "relative rank" in q or "rank of the" in q:
                return OutputType.DISCRETE
            return OutputType.NUMERIC

    for pat in CATEGORICAL_PATTERNS:
        if re.search(pat, q):
            return OutputType.CATEGORICAL

    # Check for "Who will win" style questions explicitly
    if re.search(r"who\s+will\s+(win|be)", q):
        return OutputType.CATEGORICAL

    # "Which X will Y" is categorical
    if re.search(r"which\s+\w+\s+will", q):
        return OutputType.CATEGORICAL

    return OutputType.BINARY


# ── Classification ───────────────────────────────────────────────────


def classify_question(question: str) -> tuple[str, str]:
    """Classify a question into (event_type, domain)."""
    q = question.lower()

    # Domain
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in q)
    domain = max(scores, key=lambda d: scores[d])
    if scores[domain] == 0:
        domain = "politics"

    # Event type
    for event_type, pattern in EVENT_TYPE_PATTERNS:
        if re.search(pattern, q):
            return event_type, domain

    return "other", domain


# ── Parsing case files ───────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        fm[key] = val
    return fm


def load_case_library(vault_dir: Path) -> list[dict[str, Any]]:
    cases_dir = vault_dir / "cases"
    if not cases_dir.exists():
        return []

    cases: list[dict[str, Any]] = []
    for case_file in sorted(cases_dir.glob("*.md")):
        try:
            text = case_file.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)

            body_match = re.search(r"^# (.+)$", text, re.MULTILINE)
            question = body_match.group(1).strip() if body_match else fm.get("case_id", "?")

            resolution_bool = None
            rb = fm.get("resolution_bool", "").lower()
            if rb == "true":
                resolution_bool = True
            elif rb == "false":
                resolution_bool = False

            time_horizon = None
            th = fm.get("time_horizon_days", "")
            if th and th != "null":
                try:
                    time_horizon = int(th)
                except ValueError:
                    pass

            cases.append({
                "case_id": fm.get("case_id", case_file.stem),
                "question": question,
                "event_type": fm.get("event_type", "other"),
                "domain": fm.get("domain", "politics"),
                "resolution_bool": resolution_bool,
                "resolution_label": fm.get("resolution", "?"),
                "time_horizon_days": time_horizon,
                "reference_class_base_rate": float(fm.get("reference_class_base_rate", "0.5")),
                "reference_class_size": int(fm.get("reference_class_size", "0")),
                "slug": fm.get("slug", ""),
                "_text": text,
            })
        except Exception:
            continue

    return cases


# ── Binary base rate computation ─────────────────────────────────────


def compute_base_rate(
    question: str,
    vault_dir: str | Path,
    *,
    max_matches: int = 5,
) -> BaseRateResult:
    vault = Path(vault_dir)
    event_type, domain = classify_question(question)
    cases = load_case_library(vault)

    if not cases:
        return BaseRateResult(
            event_type=event_type, domain=domain,
            base_rate=0.5, total_cases=0, resolved_yes=0, resolved_no=0,
            matches=[], confidence="low",
        )

    ref_cases = [c for c in cases if c["event_type"] == event_type and c["domain"] == domain]
    if not ref_cases:
        ref_cases = [c for c in cases if c["domain"] == domain]
    if not ref_cases:
        ref_cases = cases

    yes_count = sum(1 for c in ref_cases if c["resolution_bool"] is True)
    no_count = sum(1 for c in ref_cases if c["resolution_bool"] is False)
    total = yes_count + no_count
    base_rate = yes_count / total if total > 0 else 0.5

    confidence = "high" if total >= 5 else "medium" if total >= 3 else "low"

    # Similarity scoring
    q_tokens = _tokenize(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in cases:
        c_tokens = _tokenize(c["question"] + " " + c.get("_text", "")[:2000])
        sim = _jaccard_similarity(q_tokens, c_tokens)
        if c["event_type"] == event_type and c["domain"] == domain:
            sim += 0.15
        elif c["domain"] == domain:
            sim += 0.05
        if c.get("slug"):
            slug_tokens = _tokenize(c["slug"].replace("-", " "))
            sim += _jaccard_similarity(q_tokens, slug_tokens) * 0.1
        scored.append((sim, c))

    scored.sort(key=lambda x: -x[0])

    matches: list[CaseMatch] = []
    for sim, c in scored[:max_matches]:
        matches.append(CaseMatch(
            case_id=c["case_id"],
            question=c["question"],
            similarity=round(sim, 3),
            resolution_bool=c["resolution_bool"],
            resolution_label=c["resolution_label"],
            event_type=c["event_type"],
            domain=c["domain"],
            time_horizon_days=c.get("time_horizon_days"),
            reference_class_base_rate=c.get("reference_class_base_rate", base_rate),
            reference_class_size=c.get("reference_class_size", total),
            slug=c.get("slug", ""),
        ))

    return BaseRateResult(
        event_type=event_type, domain=domain,
        base_rate=base_rate, total_cases=total,
        resolved_yes=yes_count, resolved_no=no_count,
        matches=matches, confidence=confidence,
    )


# ── Numeric / Categorical anchor ──────────────────────────────────────


def compute_numeric_anchor(
    question: str,
    vault_dir: str | Path,
    *,
    max_matches: int = 5,
) -> NumericAnchor:
    """Build a numeric anchor with event_type/domain classification and similar cases.

    Gold set is binary-only, so no reference class distribution data yet.
    Returns classification + best-match similar binary cases for context.
    """
    vault = Path(vault_dir)
    event_type, domain = classify_question(question)
    cases = load_case_library(vault)
    matches = _find_similar_cases(question, cases, event_type, domain, max_matches)
    return NumericAnchor(event_type=event_type, domain=domain, matches=matches)


def compute_categorical_anchor(
    question: str,
    vault_dir: str | Path,
    *,
    max_matches: int = 5,
) -> CategoricalAnchor:
    """Build a categorical anchor with event_type/domain classification and similar cases."""
    vault = Path(vault_dir)
    event_type, domain = classify_question(question)
    cases = load_case_library(vault)
    matches = _find_similar_cases(question, cases, event_type, domain, max_matches)
    return CategoricalAnchor(event_type=event_type, domain=domain, matches=matches)


def _find_similar_cases(
    question: str,
    cases: list[dict[str, Any]],
    event_type: str,
    domain: str,
    max_matches: int,
) -> list[CaseMatch]:
    """Shared similarity search for all output types."""
    q_tokens = _tokenize(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in cases:
        c_tokens = _tokenize(c["question"] + " " + c.get("_text", "")[:2000])
        sim = _jaccard_similarity(q_tokens, c_tokens)
        if c["event_type"] == event_type and c["domain"] == domain:
            sim += 0.15
        elif c["domain"] == domain:
            sim += 0.05
        if c.get("slug"):
            slug_tokens = _tokenize(c["slug"].replace("-", " "))
            sim += _jaccard_similarity(q_tokens, slug_tokens) * 0.1
        scored.append((sim, c))
    scored.sort(key=lambda x: -x[0])

    matches: list[CaseMatch] = []
    for sim, c in scored[:max_matches]:
        matches.append(CaseMatch(
            case_id=c["case_id"],
            question=c["question"],
            similarity=round(sim, 3),
            resolution_bool=c["resolution_bool"],
            resolution_label=c["resolution_label"],
            event_type=c["event_type"],
            domain=c["domain"],
            time_horizon_days=c.get("time_horizon_days"),
            reference_class_base_rate=c.get("reference_class_base_rate", 0.5),
            reference_class_size=c.get("reference_class_size", 0),
            slug=c.get("slug", ""),
        ))
    return matches


# ── Polymarket anchor ────────────────────────────────────────────────


def query_polymarket_anchor(question: str, *, timeout: int = 10) -> PolymarketAnchor | None:
    try:
        encoded = urllib.parse.quote(question[:200])
        url = f"https://gamma-api.polymarket.com/public-search?q={encoded}&limit=5"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; psychohistory-outside-view/0.2)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None

    if not isinstance(data, list) or not data:
        return None

    for item in data:
        markets = item.get("markets", [])
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(outcomes, list) or len(outcomes) < 2:
                continue

            prices_str = market.get("outcomePrices", "[]")
            if isinstance(prices_str, str):
                try:
                    prices = [float(p) for p in json.loads(prices_str)]
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
            elif isinstance(prices_str, list):
                prices = [float(p) for p in prices_str]
            else:
                continue

            if not prices:
                continue

            price = prices[0]
            volume = float(market.get("volumeNum", market.get("volume", 0)) or 0)
            liquidity = float(market.get("liquidityNum", market.get("liquidity", 0)) or 0)

            return PolymarketAnchor(
                price=price,
                volume=volume,
                slug=market.get("slug", ""),
                question=market.get("question", "")[:200],
                liquidity=liquidity,
            )

    return None


# ── Main entry point ─────────────────────────────────────────────────


def get_outside_view_anchor(
    question: str,
    vault_dir: str | Path,
    *,
    query_polymarket: bool = True,
) -> OutsideViewAnchor:
    """Get the complete outside-view anchor for any question type.

    Detects output_type, computes the appropriate anchor, and optionally
    queries Polymarket for a live calibration signal.
    """
    output_type = detect_output_type(question)
    event_type, domain = classify_question(question)
    pm = query_polymarket_anchor(question) if query_polymarket else None

    if output_type == OutputType.BINARY:
        base = compute_base_rate(question, vault_dir)
        strategy = _binary_strategy(base, pm)
        return OutsideViewAnchor(
            output_type=output_type,
            event_type=event_type,
            domain=domain,
            binary=base,
            polymarket=pm,
            anchoring_strategy=strategy,
        )

    elif output_type == OutputType.NUMERIC:
        num = compute_numeric_anchor(question, vault_dir)
        strategy = (
            f"numeric({event_type}/{domain}) — no reference class distribution available. "
            f"Use causal/analogical/narrative reasoning to estimate value + confidence interval."
        )
        return OutsideViewAnchor(
            output_type=output_type,
            event_type=event_type,
            domain=domain,
            numeric=num,
            polymarket=pm,
            anchoring_strategy=strategy,
        )

    else:  # categorical or discrete
        cat = compute_categorical_anchor(question, vault_dir)
        strategy = (
            f"categorical({event_type}/{domain}) — no reference class distribution available. "
            f"Use causal/analogical/narrative reasoning to estimate probability distribution over choices."
        )
        return OutsideViewAnchor(
            output_type=output_type,
            event_type=event_type,
            domain=domain,
            categorical=cat,
            polymarket=pm,
            anchoring_strategy=strategy,
        )


def _binary_strategy(base: BaseRateResult, pm: PolymarketAnchor | None) -> str:
    if pm is not None and pm.volume > 50_000:
        prior = 0.55 * base.base_rate + 0.45 * pm.price
        return (
            f"base_rate({base.event_type}/{base.domain})={base.base_rate:.1%} "
            f"(N={base.total_cases}) × polymarket={pm.price:.1%} (${pm.volume:,.0f}) "
            f"→ prior≈{prior:.1%}"
        )
    elif pm is not None:
        prior = 0.75 * base.base_rate + 0.25 * pm.price
        return (
            f"base_rate({base.event_type}/{base.domain})={base.base_rate:.1%} (N={base.total_cases}) "
            f"+ thin polymarket={pm.price:.1%} → prior≈{prior:.1%}"
        )
    elif base.total_cases >= 3:
        return (
            f"base_rate({base.event_type}/{base.domain})={base.base_rate:.1%} "
            f"(N={base.total_cases}) — no polymarket anchor"
        )
    else:
        return (
            f"base_rate({base.event_type}/{base.domain})={base.base_rate:.1%} "
            f"(N={base.total_cases}, LOW confidence) — use causal/analogical reasoning"
        )


# ── Prompt formatting ────────────────────────────────────────────────


def format_anchor_for_prompt(anchor: OutsideViewAnchor) -> str:
    """Format an OutsideViewAnchor as a prompt block for injection into agent prompts."""
    lines = [
        "=== OUTSIDE-VIEW ANCHOR ===",
        f"Output type: {anchor.output_type.value}",
        f"Event type: {anchor.event_type} | Domain: {anchor.domain}",
        f"Strategy: {anchor.anchoring_strategy}",
        "",
    ]

    if anchor.polymarket:
        pm = anchor.polymarket
        lines += [
            f"Polymarket equivalent: {pm.question[:120]}",
            f"  YES/price: {pm.price:.1%} | Volume: ${pm.volume:,.0f}",
            "",
        ]

    # Binary anchoring
    if anchor.binary:
        br = anchor.binary
        lines += [
            f"Reference class: {br.total_cases} resolved binary cases "
            f"({br.resolved_yes} YES, {br.resolved_no} NO)",
            f"Base rate: {br.base_rate:.1%} (confidence: {br.confidence})",
            "",
            "=== BINARY ANCHORING INSTRUCTION ===",
            "Start from the base rate above. Adjust ONLY with PIT-constrained evidence.",
            "Document any deviation from base rate with specific justification.",
            "",
        ]

    # Numeric anchoring
    if anchor.numeric:
        lines += [
            f"Reference class: {len(anchor.numeric.matches)} similar resolved cases found.",
            anchor.numeric.note,
            "",
            "=== NUMERIC ANCHORING INSTRUCTION ===",
            "Estimate a central value AND a confidence interval (ci_low, ci_high).",
            "Use causal modeling to identify structural bounds on the value.",
            "Use analogical reasoning to find similar historical numeric outcomes.",
            "Use narrative reasoning to generate scenarios at different value levels.",
            "No reference class distribution exists — your reasoning IS the calibration.",
            "",
        ]

    # Categorical anchoring
    if anchor.categorical:
        lines += [
            f"Reference class: {len(anchor.categorical.matches)} similar resolved cases found.",
            anchor.categorical.note,
            "",
            "=== CATEGORICAL ANCHORING INSTRUCTION ===",
            "Produce a probability distribution over all possible choices/outcomes.",
            "All probabilities must sum to 1.0.",
            "Use causal modeling to identify which choices are structurally possible.",
            "Use analogical reasoning to find similar historical categorical outcomes.",
            "Use narrative reasoning to generate scenarios for each leading choice.",
            "No reference class distribution exists — your reasoning IS the calibration.",
            "",
        ]

    # Similar cases (shared across types)
    matches = (
        anchor.binary.matches if anchor.binary
        else anchor.numeric.matches if anchor.numeric
        else anchor.categorical.matches if anchor.categorical
        else []
    )
    if matches:
        lines += [f"Most similar resolved cases ({len(matches)}):"]
        for m in matches:
            res_icon = "✅" if m.resolution_bool else "❌" if m.resolution_bool is not None else "⬜"
            lines.append(
                f"  {res_icon} [{m.similarity:.2f}] {m.question[:100]} "
                f"({m.event_type}/{m.domain})"
            )
        lines.append("")

    return "\n".join(lines)
