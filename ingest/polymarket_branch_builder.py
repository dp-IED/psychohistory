"""Deterministic resolved Polymarket event -> branch portfolio/graph builder.

This module is intentionally a contract/baseline builder, not a forecasting model.
It follows the pivot note's immediate scope: binary resolved Polymarket questions,
strict separation of terminal resolution labels from PIT evidence, and bounded
branch portfolios that a later agent/GNN loop can stress-test.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from evals.graph_artifact_contract import (
    ArtifactEdge,
    ArtifactNode,
    ArtifactProvenance,
    ArtifactTargetRecord,
    GraphArtifactV1,
)
from ingest.polymarket_resolved import ResolvedMarketRecord
from schemas.polymarket_agentic import (
    Branch,
    BranchType,
    Direction,
    ElementRole,
    EvidenceRef,
    EvidenceTier,
    HypothesisSide,
    MarketFamily,
    MarketFrame,
    OutcomeHypothesis,
    POLYMARKET_V1_POLICIES,
    PortfolioElement,
    Prerequisite,
    SubgraphPortfolio,
    validate_portfolio_against_policy,
)

CUTOFF_POLICY = "terminal resolution is label only; PIT evidence must be attached downstream"


@dataclass(frozen=True)
class GoldBranchCase:
    case_id: str
    record: ResolvedMarketRecord
    expected_family: str
    expected_required_branches: tuple[str, ...]
    expected_target_value: float
    expected_min_prerequisites: int
    expected_node_types: tuple[str, ...]
    full_text: str = ""
    gold_orientation: dict[str, object] | None = None


@dataclass(frozen=True)
class GoldEvaluationReport:
    case_count: int
    family_accuracy: float
    branch_recall: float
    target_accuracy: float
    policy_pass_rate: float
    content_coverage: float
    branch_content_recall: float
    expressiveness_score: float
    failures: list[str]


STOPWORDS = {
    "will", "there", "before", "after", "during", "this", "that", "with", "from", "into",
    "have", "been", "being", "market", "resolves", "resolution", "criteria", "shall",
    "question", "whether", "price", "above", "below", "yes", "no", "the", "and", "for",
    "of", "to", "in", "on", "by", "at", "from", "into", "following", "between",
}


def _content_terms(text: str, *, limit: int = 18) -> list[str]:
    import re

    counts: dict[str, int] = {}
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9'\-]{3,}", text):
        token = raw.lower().strip("'-")
        if len(token) < 4 or token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    return sorted(counts, key=lambda key: (-counts[key], len(key), key))[:limit]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def infer_market_family(record: ResolvedMarketRecord) -> MarketFamily:
    """Infer the v1 family that determines branch policy.

    The rules mirror the .md guidance: start narrow with binary Polymarket
    questions and keep the mapping interpretable enough to audit against a gold
    set before moving to learned/agentic retrieval.
    """

    text = f"{record.question} {record.description} {record.category or ''} {record.slug}".lower()
    # Institutional markers are more specific than country/conflict terms. Check
    # them first so e.g. an "Iran sanctions vote in Senate" is routed as a
    # formal process, not a generic Iran negotiation event.
    if _contains_any(
        text,
        (
            "inaugurat",
            "congress",
            "senate",
            "parliament",
            "court",
            "supreme court",
            "bill ",
            "veto",
            "impeach",
            "confirmation",
            "ballot",
            "sec",
            " ban ",
            "trial",
            "shutdown",
            "sentenced",
            "sentencing",
            "deputies",
        ),
    ):
        return MarketFamily.INSTITUTIONAL_PROCESS
    if _contains_any(
        text,
        (
            "ceasefire",
            "truce",
            "hostage",
            "deal",
            "agreement",
            "treaty",
            "negotiation",
            "peace",
            "war",
            "strike",
            "summit",
            "nato",
            "iran",
            "ukraine",
            "gaza",
            "russia",
            "taiwan",
            "hamas",
            "israel",
        ),
    ):
        return MarketFamily.EVENT_NEGOTIATION
    if _contains_any(
        text,
        (
            "election",
            "nomination",
            "presidential race",
        ),
    ):
        return MarketFamily.INSTITUTIONAL_PROCESS
    if _contains_any(
        text,
        (
            "fed",
            "fomc",
            "rate cut",
            "cut rates",
            "rates ",
            "interest rate",
            "target range",
            "cpi",
            "inflation",
            "jobs report",
            "unemployment",
            "gdp",
            "treasury",
            "recession",
            "policy print",
            "central bank",
            "opec",
        ),
    ):
        return MarketFamily.MACRO_POLICY_PRINT
    return MarketFamily.EVENT_NEGOTIATION


def market_frame_from_resolved_record(
    record: ResolvedMarketRecord,
    *,
    family: MarketFamily | None = None,
) -> MarketFrame:
    return MarketFrame(
        market_id=record.id,
        question=record.question,
        family=family or infer_market_family(record),
        resolution_criteria=record.description,
        close_time=record.closed_time or record.end_date,
        resolution_time=record.closed_time,
        category=record.category,
        url=record.url,
        outcomes=tuple(record.outcomes),
        resolved_outcome=record.resolved_outcome,
    )


def _evidence(record: ResolvedMarketRecord) -> EvidenceRef:
    return EvidenceRef(
        ref_id=f"market:{record.id}",
        tier=EvidenceTier.MARKET_METADATA,
        uri=record.gamma_url or record.url,
        as_of_time=record.updated_at or record.closed_time,
        note="Resolved market metadata used for framing/labeling, not PIT forecast evidence.",
    )


def _element(
    prefix: str,
    slug: str,
    label: str,
    role: ElementRole,
    direction: Direction,
    rationale: str,
    evidence_ref: str,
) -> PortfolioElement:
    safe = label.lower().replace("/", " ").replace("-", " ")
    token = "_".join(part for part in safe.split()[:5])
    return PortfolioElement(
        element_id=f"{prefix}:{slug}:{token}",
        label=label,
        role=role,
        direction=direction,
        rationale=rationale,
        evidence_refs=(evidence_ref,),
    )


def _with_content_specs(
    base: dict[BranchType, tuple[tuple[str, ElementRole, Direction, str], ...]],
    context_text: str,
) -> dict[BranchType, tuple[tuple[str, ElementRole, Direction, str], ...]]:
    """Return family scaffold specs without assigning content-derived semantics.

    V1 deliberately avoids keyword/index-based enrichment: a generic but typed
    scaffold is safer than assigning SIGNAL/DRIVER/CONSTRAINT roles to extracted
    market terms by position. The downstream agent loop owns market-specific
    expansion and should fill concrete PIT-backed factors into these branches.
    """

    _ = context_text  # Reserved for the agent-filled v2 enrichment interface.
    return base


def _family_branch_specs(family: MarketFamily, context_text: str = "") -> dict[BranchType, tuple[tuple[str, ElementRole, Direction, str], ...]]:
    if family == MarketFamily.MACRO_POLICY_PRINT:
        return _with_content_specs({
            BranchType.LOCAL: (
                ("latest indicator path supports threshold", ElementRole.SIGNAL, Direction.FOR, "Local data can make the market criterion reachable."),
                ("official communication resists threshold", ElementRole.CONSTRAINT, Direction.AGAINST, "Policy guidance or release details can block the path."),
                ("scheduled data-release gate", ElementRole.INSTITUTIONAL_GATE, Direction.MIXED, "Release timing controls what can be known before close."),
                ("cutoff-safe indicator freshness check", ElementRole.SIGNAL, Direction.MIXED, "The branch must distinguish latest PIT-observable data from post-cutoff revisions."),
                ("threshold definition or bracket mapping", ElementRole.CONSTRAINT, Direction.AGAINST, "Resolution definitions can make a headline macro move insufficient for the market threshold."),
            ),
            BranchType.ANALOGUE: (
                ("historical regime analogue", ElementRole.DRIVER, Direction.MIXED, "Comparable macro regimes define plausible threshold behavior."),
                ("methodology or revision analogue", ElementRole.CONSTRAINT, Direction.AGAINST, "Revisions can make a naive indicator branch fragile."),
                ("prior central-bank communication analogue", ElementRole.SIGNAL, Direction.MIXED, "Past statement-to-action patterns seed a comparable policy-communication branch."),
                ("regime-break mismatch", ElementRole.CONSTRAINT, Direction.AGAINST, "Analogues must log when inflation, labor, or liquidity regimes differ from the current setup."),
            ),
            BranchType.DISRUPTOR: (
                ("exogenous financial stress shock", ElementRole.SPOILER, Direction.AGAINST, "Banking or liquidity shocks can force a policy path outside the local indicator baseline."),
                ("geopolitical price shock", ElementRole.SPOILER, Direction.MIXED, "Energy or conflict shocks can reverse the expected policy-print interpretation."),
                ("surprise data revision", ElementRole.SIGNAL, Direction.MIXED, "Late revisions can change the information set that policymakers and markets react to."),
                ("emergency policy communication", ElementRole.CONSTRAINT, Direction.AGAINST, "Unscheduled guidance can invalidate a planned-meeting-only scaffold."),
            ),
        }, context_text)
    if family == MarketFamily.INSTITUTIONAL_PROCESS:
        return _with_content_specs({
            BranchType.LOCAL: (
                ("formal process step completed", ElementRole.INSTITUTIONAL_GATE, Direction.FOR, "The institutional path requires explicit procedural completion."),
                ("procedural blocker or legal challenge", ElementRole.CONSTRAINT, Direction.AGAINST, "Formal challenges can delay or prevent resolution."),
                ("deadline pressure", ElementRole.SIGNAL, Direction.MIXED, "Deadlines shape strategic behavior and observability."),
                ("actor authority verified", ElementRole.INSTITUTIONAL_GATE, Direction.FOR, "The relevant actor or body must actually control the formal step named by the market."),
                ("current status uncertainty", ElementRole.SIGNAL, Direction.MIXED, "The scaffold leaves room for the agent to attach the latest PIT status rather than assuming completion."),
            ),
            BranchType.DISRUPTOR: (
                ("agenda disruption", ElementRole.SPOILER, Direction.AGAINST, "Remote shocks can dominate a locally plausible process."),
                ("coalition fracture risk", ElementRole.SPOILER, Direction.AGAINST, "Institutional coalitions can fail late."),
                ("late procedural reinterpretation", ElementRole.CONSTRAINT, Direction.MIXED, "Rules, courts, or administrators can alter how a process counts for resolution."),
                ("external scandal or crisis signal", ElementRole.SIGNAL, Direction.MIXED, "Non-local shocks can change incentives even when formal gates remain available."),
            ),
        }, context_text)
    return _with_content_specs({
        BranchType.LOCAL: (
            ("direct parties maintain bargaining channel", ElementRole.DRIVER, Direction.FOR, "The Yes path needs direct or mediated bargaining to remain open."),
            ("spoiler actor can break talks", ElementRole.SPOILER, Direction.AGAINST, "A local spoiler can invalidate a simple deal path."),
            ("timing constraint before resolution", ElementRole.CONSTRAINT, Direction.MIXED, "The deadline controls reachability."),
            ("credible public signal of bargaining status", ElementRole.SIGNAL, Direction.MIXED, "The branch needs PIT-observable signals of process status without treating them as final evidence."),
            ("implementation or definition gate", ElementRole.INSTITUTIONAL_GATE, Direction.AGAINST, "Resolution definitions can require an official/qualifying act rather than informal progress."),
        ),
        BranchType.ANALOGUE: (
            ("prior comparable negotiation", ElementRole.SIGNAL, Direction.MIXED, "Analogues test whether the mechanism has worked before."),
            ("failed-deal analogue", ElementRole.CONSTRAINT, Direction.AGAINST, "Negative analogues prevent one-sided optimism."),
            ("successful mechanism analogue", ElementRole.DRIVER, Direction.MIXED, "Positive analogues seed a mechanism path the agent can verify against PIT evidence."),
            ("analogue scope mismatch", ElementRole.CONSTRAINT, Direction.AGAINST, "Differences in actors, deadline, or enforcement prevent copying a historical case naively."),
        ),
        BranchType.DISRUPTOR: (
            ("unrelated escalation channel", ElementRole.SPOILER, Direction.AGAINST, "A remote escalation can flip the forecast despite local progress."),
            ("sponsor or domestic constraint", ElementRole.CONSTRAINT, Direction.AGAINST, "Backers and domestic audiences can veto compromise."),
            ("third-party intervention signal", ElementRole.SIGNAL, Direction.MIXED, "External mediation or pressure can change bargaining incentives outside the local baseline."),
            ("remote de-escalation surprise", ElementRole.DRIVER, Direction.MIXED, "Disruptors can also unexpectedly open a path that the local branch alone would miss."),
        ),
    }, context_text)


def _prerequisites(family: MarketFamily, slug: str, side: HypothesisSide, evidence_ref: str) -> tuple[Prerequisite, ...]:
    # These are scaffold-floor prerequisites. ``status`` and ``importance`` are
    # placeholders for the downstream agent/GNN loop, not factual assessments of
    # the live market state.
    if family == MarketFamily.MACRO_POLICY_PRINT:
        labels = (
            "cutoff-safe latest indicator coverage exists",
            "release calendar precedes market close",
            "methodology/revision risk is logged",
        )
    elif family == MarketFamily.INSTITUTIONAL_PROCESS:
        labels = (
            "formal process calendar remains valid",
            "institutional actor authority is established",
            "blocking litigation or procedural veto is assessed",
            "deadline path is still reachable before close",
        )
    else:
        labels = (
            "direct parties remain capable of agreement",
            "mediator or bargaining channel is active",
            "spoiler escalation risk is explicitly checked",
            "deadline path is still reachable before close",
        )
    return tuple(
        Prerequisite(
            prerequisite_id=f"prereq:{slug}:{side.value.lower()}:{index}",
            description=label,
            status=Direction.MIXED if index == 1 else Direction.UNKNOWN,
            importance=0.7 if index == 1 else 0.55,
            evidence_refs=(evidence_ref,),
        )
        for index, label in enumerate(labels, start=1)
    )


def _branch_for_spec(
    *,
    record: ResolvedMarketRecord,
    side: HypothesisSide,
    branch_type: BranchType,
    specs: tuple[tuple[str, ElementRole, Direction, str], ...],
    evidence_ref: str,
) -> Branch:
    elements = tuple(
        _element(
            f"element:{side.value.lower()}:{branch_type.value}",
            record.slug,
            label,
            role,
            direction,
            rationale,
            evidence_ref,
        )
        for label, role, direction, rationale in specs
    )
    return Branch(
        branch_id=f"branch:{record.id}:{side.value.lower()}:{branch_type.value}",
        branch_type=branch_type,
        seed_elements=elements,
        rationale=(
            f"Bounded {branch_type.value} branch for the {side.value}-world of '{record.question}'. "
            "Generated from market text only as a planning scaffold; PIT evidence is attached downstream."
        ),
        evidence_refs=(evidence_ref,),
        # NOTE: expansion_budget is a planning hint for the agent loop; this
        # deterministic scaffold builder stamps it but does not enforce expansion.
        expansion_budget=POLYMARKET_V1_POLICIES[infer_market_family(record)].max_nodes_by_branch.get(branch_type, 10),
    )


def build_portfolios_from_resolved_record(
    record: ResolvedMarketRecord,
    *,
    as_of_time: str,
) -> tuple[SubgraphPortfolio, SubgraphPortfolio]:
    """Create Yes/No branch portfolios for a resolved binary market.

    Both sides are generated even though the outcome is known. The resolution is
    carried only as a label on ``MarketFrame``/target artifacts so the same
    branch contract can be used for pre-resolution forecast runs.
    """

    family = infer_market_family(record)
    frame = market_frame_from_resolved_record(record, family=family)
    evidence = _evidence(record)
    specs = _family_branch_specs(family, f"{record.question}\n{record.description}")
    portfolios: list[SubgraphPortfolio] = []
    for side in (HypothesisSide.YES, HypothesisSide.NO):
        hypothesis = OutcomeHypothesis(
            hypothesis_id=f"hypothesis:{record.id}:{side.value.lower()}",
            market_frame=frame,
            side=side,
            summary=f"{side.value}-world for resolved Polymarket question: {record.question}",
            assumptions=(
                "Use resolved outcome only as benchmark label.",
                "Treat branch elements as bounded retrieval seeds, not final evidence.",
            ),
            evidence_refs=(evidence,),
        )
        branches = tuple(
            _branch_for_spec(
                record=record,
                side=side,
                branch_type=branch_type,
                specs=branch_specs,
                evidence_ref=evidence.ref_id,
            )
            for branch_type, branch_specs in specs.items()
        )
        portfolios.append(
            SubgraphPortfolio(
                portfolio_id=f"portfolio:{record.id}:{side.value.lower()}",
                hypothesis=hypothesis,
                as_of_time=as_of_time,
                branches=branches,
                prerequisites=_prerequisites(family, record.slug, side, evidence.ref_id),
            )
        )
    return (portfolios[0], portfolios[1])


def _node(node_id: str, node_type: str, label: str, *, sources: list[str], **attributes: object) -> ArtifactNode:
    return ArtifactNode(
        id=node_id,
        type=node_type,
        label=label,
        provenance=ArtifactProvenance(sources=sources),
        attributes=dict(attributes),
    )


def build_graph_artifact_from_record(record: ResolvedMarketRecord, *, as_of_time: str) -> GraphArtifactV1:
    portfolios = build_portfolios_from_resolved_record(record, as_of_time=as_of_time)
    sources = [record.gamma_url or record.url]
    nodes: list[ArtifactNode] = [
        _node(
            f"market:{record.id}",
            "market",
            record.question,
            sources=sources,
            slug=record.slug,
            family=infer_market_family(record).value,
            category=record.category,
            cutoff_policy=CUTOFF_POLICY,
        )
    ]
    edges: list[ArtifactEdge] = []

    for portfolio in portfolios:
        outcome_node_id = f"outcome:{record.id}:{portfolio.hypothesis.side.value.lower()}"
        nodes.append(
            _node(
                outcome_node_id,
                "outcome_hypothesis",
                portfolio.hypothesis.summary,
                sources=sources,
                side=portfolio.hypothesis.side.value,
                portfolio_id=portfolio.portfolio_id,
            )
        )
        edges.append(ArtifactEdge(source=f"market:{record.id}", target=outcome_node_id, type="HAS_OUTCOME"))
        for branch in portfolio.branches:
            branch_node_id = branch.branch_id
            nodes.append(
                _node(
                    branch_node_id,
                    "branch",
                    branch.branch_type.value,
                    sources=sources,
                    branch_type=branch.branch_type.value,
                    expansion_budget=branch.expansion_budget,
                    rationale=branch.rationale,
                )
            )
            edges.append(ArtifactEdge(source=outcome_node_id, target=branch_node_id, type="HAS_BRANCH"))
            for element in branch.seed_elements:
                nodes.append(
                    _node(
                        element.element_id,
                        "portfolio_element",
                        element.label,
                        sources=sources,
                        role=element.role.value,
                        direction=element.direction.value,
                        rationale=element.rationale,
                    )
                )
                edges.append(ArtifactEdge(source=branch_node_id, target=element.element_id, type="SEEDS_ELEMENT"))
        for prerequisite in portfolio.prerequisites:
            nodes.append(
                _node(
                    prerequisite.prerequisite_id,
                    "prerequisite",
                    prerequisite.description,
                    sources=sources,
                    status=prerequisite.status.value,
                    importance=prerequisite.importance,
                )
            )
            edges.append(ArtifactEdge(source=prerequisite.prerequisite_id, target=outcome_node_id, type="GATES_OUTCOME"))

    resolved_yes = 1.0 if record.resolved_outcome.lower() == "yes" else 0.0
    return GraphArtifactV1(
        probe_id=f"polymarket:{record.id}",
        schema_version="polymarket_agentic_v1",
        nodes=nodes,
        edges=edges,
        target_table=[
            ArtifactTargetRecord(
                target_id=f"target:{record.id}:resolved_yes",
                name="resolved_yes",
                value=resolved_yes,
                split="gold",
                node_ids=[f"market:{record.id}"],
                metadata={
                    "resolved_outcome": record.resolved_outcome,
                    "terminal_outcome_prices": record.terminal_outcome_prices,
                    "cutoff_policy": CUTOFF_POLICY,
                },
            )
        ],
        metadata={
            "source": "polymarket_gamma_resolved_metadata",
            "cutoff_policy": CUTOFF_POLICY,
            "as_of_time": as_of_time,
            "portfolio_ids": [portfolio.portfolio_id for portfolio in portfolios],
        },
    )


def load_gold_branch_cases(path: str | Path) -> list[GoldBranchCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[GoldBranchCase] = []
    for raw in payload.get("cases", []):
        record = ResolvedMarketRecord(**raw["record"])
        cases.append(
            GoldBranchCase(
                case_id=str(raw["case_id"]),
                record=record,
                expected_family=str(raw["expected_family"]),
                expected_required_branches=tuple(raw["expected_required_branches"]),
                expected_target_value=float(raw["expected_target_value"]),
                expected_min_prerequisites=int(raw["expected_min_prerequisites"]),
                expected_node_types=tuple(raw["expected_node_types"]),
                full_text=str(raw.get("full_text", "")),
                gold_orientation=dict(raw.get("gold_orientation") or {}),
            )
        )
    return cases


def _semantic_content_score(case: GoldBranchCase, portfolio: SubgraphPortfolio) -> tuple[float, float, float, list[str]]:
    """Score semantic orientation without using bag-of-words overlap.

    The gold orientation declares branch-level semantic obligations as roles,
    directions, and mechanism families. This rewards whether the generated graph
    can express the right kinds of causal/evidential structure for the market,
    rather than whether it copied Wikipedia/market vocabulary.
    """

    failures: list[str] = []
    branch_by_type = {branch.branch_type.value: branch for branch in portfolio.branches}
    requirements = (case.gold_orientation or {}).get("semantic_requirements", {})
    if not isinstance(requirements, dict) or not requirements:
        return (1.0, 1.0, 1.0, failures)

    coverage_scores: list[float] = []
    branch_scores: list[float] = []
    for branch_name, raw_req in requirements.items():
        if not isinstance(raw_req, dict):
            continue
        branch = branch_by_type.get(str(branch_name))
        if branch is None:
            coverage_scores.append(0.0)
            branch_scores.append(0.0)
            failures.append(f"{case.case_id}: missing semantic branch {branch_name}")
            continue
        roles = {element.role.value for element in branch.seed_elements}
        directions = {element.direction.value for element in branch.seed_elements}
        req_roles = set(raw_req.get("roles", []))
        req_directions = {str(direction).lower() for direction in raw_req.get("directions", [])}
        role_score = len(roles & req_roles) / (len(req_roles) or 1)
        direction_score = len(directions & req_directions) / (len(req_directions) or 1)
        min_elements = int(raw_req.get("min_elements", 1))
        element_score = min(1.0, len(branch.seed_elements) / max(min_elements, 1))
        coverage_scores.append((role_score + direction_score + element_score) / 3.0)
        branch_scores.append(1.0 if role_score >= 0.67 and direction_score >= 0.67 and element_score >= 1.0 else 0.0)
        if branch_scores[-1] < 1.0:
            failures.append(
                f"{case.case_id}: branch {branch_name} semantic obligation under-covered "
                f"(roles={sorted(roles)}, directions={sorted(directions)}, elements={len(branch.seed_elements)})"
            )

    elements = [element for branch in portfolio.branches for element in branch.seed_elements]
    distinct_roles = {element.role.value for element in elements}
    distinct_directions = {element.direction.value for element in elements}
    rationale_chars = sum(len(element.rationale) for element in elements) + sum(len(branch.rationale) for branch in portfolio.branches)
    orientation = case.gold_orientation or {}
    min_elements = int(orientation.get("expressiveness_min_elements", 1))
    min_roles = int(orientation.get("expressiveness_min_distinct_roles", 1))
    min_rationale = int(orientation.get("expressiveness_min_rationale_chars", 1))
    expressiveness = (
        min(1.0, len(elements) / max(min_elements, 1))
        + min(1.0, len(distinct_roles) / max(min_roles, 1))
        + min(1.0, len(distinct_directions) / 3)
        + min(1.0, rationale_chars / max(min_rationale, 1))
    ) / 4.0
    return (
        sum(coverage_scores) / (len(coverage_scores) or 1),
        sum(branch_scores) / (len(branch_scores) or 1),
        expressiveness,
        failures,
    )


def evaluate_branch_builder_against_gold(path: str | Path, *, as_of_time: str = "gold-cutoff") -> GoldEvaluationReport:
    cases = load_gold_branch_cases(path)
    failures: list[str] = []
    family_hits = 0
    branch_hits = 0
    target_hits = 0
    policy_hits = 0
    content_scores: list[float] = []
    branch_content_scores: list[float] = []
    expressiveness_scores: list[float] = []
    for case in cases:
        family = infer_market_family(case.record).value
        if family == case.expected_family:
            family_hits += 1
        else:
            failures.append(f"{case.case_id}: expected family {case.expected_family}, got {family}")

        portfolios = build_portfolios_from_resolved_record(case.record, as_of_time=as_of_time)
        branch_ok = True
        policy_ok = True
        case_semantic_coverages: list[float] = []
        case_branch_content_scores: list[float] = []
        case_expressiveness_scores: list[float] = []
        for portfolio in portfolios:
            side = portfolio.hypothesis.side.value
            branches = {branch.branch_type.value for branch in portfolio.branches}
            if branches < set(case.expected_required_branches):
                branch_ok = False
                failures.append(
                    f"{case.case_id} {side}-world: missing branches {set(case.expected_required_branches) - branches}"
                )

            issues = validate_portfolio_against_policy(portfolio)
            if issues:
                policy_ok = False
                failures.append(f"{case.case_id} {side}-world: policy issues {issues}")

            semantic_coverage, semantic_branch_recall, expressiveness, semantic_failures = _semantic_content_score(case, portfolio)
            case_semantic_coverages.append(semantic_coverage)
            case_branch_content_scores.append(semantic_branch_recall)
            case_expressiveness_scores.append(expressiveness)
            failures.extend(f"{case.case_id} {side}-world: {failure.split(': ', 1)[-1]}" for failure in semantic_failures)

        if branch_ok:
            branch_hits += 1
        if policy_ok:
            policy_hits += 1
        content_scores.append(sum(case_semantic_coverages) / (len(case_semantic_coverages) or 1))
        branch_content_scores.append(sum(case_branch_content_scores) / (len(case_branch_content_scores) or 1))
        expressiveness_scores.append(sum(case_expressiveness_scores) / (len(case_expressiveness_scores) or 1))

        artifact = build_graph_artifact_from_record(case.record, as_of_time=as_of_time)
        target = next(item for item in artifact.target_table if item.name == "resolved_yes")
        if float(target.value) == case.expected_target_value:
            target_hits += 1
        else:
            failures.append(f"{case.case_id}: expected target {case.expected_target_value}, got {target.value}")
    n = len(cases) or 1
    return GoldEvaluationReport(
        case_count=len(cases),
        family_accuracy=family_hits / n,
        branch_recall=branch_hits / n,
        target_accuracy=target_hits / n,
        policy_pass_rate=policy_hits / n,
        content_coverage=sum(content_scores) / (len(content_scores) or 1),
        branch_content_recall=sum(branch_content_scores) / (len(branch_content_scores) or 1),
        expressiveness_score=sum(expressiveness_scores) / (len(expressiveness_scores) or 1),
        failures=failures,
    )


def write_graph_artifacts_jsonl(
    records: list[ResolvedMarketRecord],
    path: str | Path,
    *,
    as_of_time: str,
) -> int:
    """Materialize one graph artifact per resolved market as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            artifact = build_graph_artifact_from_record(record, as_of_time=as_of_time)
            handle.write(artifact.model_dump_json(exclude_none=True) + "\n")
    return len(records)


def load_resolved_records_json(path: str | Path) -> list[ResolvedMarketRecord]:
    """Load records from ``fetch_polymarket_resolved.py`` JSON output."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ResolvedMarketRecord(**item) for item in payload.get("records", [])]


__all__ = [
    "CUTOFF_POLICY",
    "GoldBranchCase",
    "GoldEvaluationReport",
    "build_graph_artifact_from_record",
    "build_portfolios_from_resolved_record",
    "evaluate_branch_builder_against_gold",
    "infer_market_family",
    "load_gold_branch_cases",
    "load_resolved_records_json",
    "market_frame_from_resolved_record",
    "write_graph_artifacts_jsonl",
]
