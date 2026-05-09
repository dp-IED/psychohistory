"""Agentic Polymarket portfolio contracts.

These contracts encode the v1 pivot from a single deterministic graph-builder
pipeline to an agent-led research harness where an LLM proposes bounded outcome
worlds and a graph/GNN layer stress-tests reachability, missingness, and branch
usefulness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MarketFamily(StrEnum):
    """V1 market families with tractable graph structure."""

    INSTITUTIONAL_PROCESS = "institutional_process"
    EVENT_NEGOTIATION = "event_negotiation"
    MACRO_POLICY_PRINT = "macro_policy_print"


class HypothesisSide(StrEnum):
    YES = "YES"
    NO = "NO"


class BranchType(StrEnum):
    LOCAL = "local"
    ANALOGUE = "analogue"
    DISRUPTOR = "disruptor"
    COUNTERWORLD = "counterworld"


class ElementRole(StrEnum):
    DRIVER = "driver"
    CONSTRAINT = "constraint"
    PREREQUISITE = "prerequisite"
    SPOILER = "spoiler"
    SIGNAL = "signal"
    INSTITUTIONAL_GATE = "institutional_gate"


class Direction(StrEnum):
    FOR = "for"
    AGAINST = "against"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EvidenceTier(StrEnum):
    MARKET_METADATA = "market_metadata"
    PIT_WEB = "pit_web"
    WAREHOUSE = "warehouse"
    ANALOGUE = "analogue"
    AGENT_HYPOTHESIS = "agent_hypothesis"


@dataclass(frozen=True)
class EvidenceRef:
    """Small provenance pointer; full source bodies stay outside the contract."""

    ref_id: str
    tier: EvidenceTier
    uri: str
    as_of_time: str | None = None
    note: str = ""


@dataclass(frozen=True)
class MarketFrame:
    """Normalized Polymarket question frame used by the harness planner."""

    market_id: str
    question: str
    family: MarketFamily
    resolution_criteria: str
    close_time: str | None = None
    resolution_time: str | None = None
    category: str | None = None
    url: str | None = None
    outcomes: tuple[str, ...] = ("Yes", "No")
    resolved_outcome: str | None = None

    def is_binary_yes_no(self) -> bool:
        return tuple(o.lower() for o in self.outcomes) == ("yes", "no")


@dataclass(frozen=True)
class PortfolioElement:
    """Forecast-relevant node/factor admitted into a branch or prerequisite set."""

    element_id: str
    label: str
    role: ElementRole
    direction: Direction
    rationale: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Branch:
    """Bounded branch in a candidate outcome-world portfolio."""

    branch_id: str
    branch_type: BranchType
    seed_elements: tuple[PortfolioElement, ...]
    rationale: str
    evidence_refs: tuple[str, ...] = ()
    expansion_budget: int = 10

    def directions_present(self) -> set[Direction]:
        return {element.direction for element in self.seed_elements}


@dataclass(frozen=True)
class Prerequisite:
    """Outcome-specific gate that must hold for a hypothesis to remain reachable."""

    prerequisite_id: str
    description: str
    status: Direction
    linked_element_ids: tuple[str, ...] = ()
    importance: float = 0.5
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Prerequisite.importance must be in [0, 1]")


@dataclass(frozen=True)
class OutcomeHypothesis:
    """Agent-authored candidate Yes-world or No-world."""

    hypothesis_id: str
    market_frame: MarketFrame
    side: HypothesisSide
    summary: str
    assumptions: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class SubgraphPortfolio:
    """Portfolio handed from the agent/planner to graph expansion and GNN scoring."""

    portfolio_id: str
    hypothesis: OutcomeHypothesis
    as_of_time: str
    branches: tuple[Branch, ...]
    prerequisites: tuple[Prerequisite, ...] = ()

    def branch_types_present(self) -> set[BranchType]:
        return {branch.branch_type for branch in self.branches}


@dataclass(frozen=True)
class RequirementStressTest:
    """GNN/graph-layer stress-test output over a portfolio."""

    portfolio_id: str
    p_yes: float
    uncertainty: float
    branch_contributions: dict[str, float] = field(
        default_factory=dict,
        metadata={
            "key_contract": "branch_type.value",
            "description": (
                "Keys are BranchType.value strings (e.g. 'local', 'disruptor'); "
                "values are marginal forecast contribution scores in [0, 1]."
            ),
        },
    )
    missingness_risk: float = 0.0
    surfaced_prerequisites: tuple[Prerequisite, ...] = ()
    fragile_element_ids: tuple[str, ...] = ()
    branch_disagreement_score: float = 0.0
    macro_correction: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("p_yes", self.p_yes),
            ("uncertainty", self.uncertainty),
            ("missingness_risk", self.missingness_risk),
            ("branch_disagreement_score", self.branch_disagreement_score),
            *(
                (f"branch_contributions[{branch_type}]", contribution)
                for branch_type, contribution in self.branch_contributions.items()
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"RequirementStressTest.{name} must be in [0, 1]")


@dataclass(frozen=True)
class ConstructionPolicy:
    """Family-specific branch and admissibility policy for portfolio construction."""

    family: MarketFamily
    required_branches: tuple[BranchType, ...]
    required_node_categories: tuple[str, ...]
    allowed_relation_types: tuple[str, ...]
    min_prerequisites: int
    max_nodes_by_branch: dict[BranchType, int]
    blind_spot_checks: tuple[str, ...]
    stop_rules: tuple[str, ...]


POLYMARKET_V1_POLICIES: dict[MarketFamily, ConstructionPolicy] = {
    MarketFamily.INSTITUTIONAL_PROCESS: ConstructionPolicy(
        family=MarketFamily.INSTITUTIONAL_PROCESS,
        required_branches=(BranchType.LOCAL, BranchType.DISRUPTOR),
        required_node_categories=(
            "actor",
            "institution",
            "formal_process_step",
            "deadline",
            "current_status",
            "coalition_or_fracture_risk",
        ),
        allowed_relation_types=(
            "controls",
            "votes_on",
            "blocks",
            "enables",
            "scheduled_before",
            "opposes",
            "supports",
            "delays",
        ),
        min_prerequisites=4,
        max_nodes_by_branch={
            BranchType.LOCAL: 25,
            BranchType.DISRUPTOR: 12,
            BranchType.COUNTERWORLD: 15,
        },
        blind_spot_checks=(
            "external_agenda_disruption",
            "coalition_fracture",
            "legal_or_scandal_branch",
        ),
        stop_rules=(
            "mandatory_branch_checklists_pass",
            "missingness_risk_below_threshold_or_logged",
            "budget_cap_reached",
        ),
    ),
    MarketFamily.EVENT_NEGOTIATION: ConstructionPolicy(
        family=MarketFamily.EVENT_NEGOTIATION,
        required_branches=(BranchType.LOCAL, BranchType.ANALOGUE, BranchType.DISRUPTOR),
        required_node_categories=(
            "direct_party",
            "mediator",
            "bargaining_status",
            "timing_constraint",
            "spoiler_actor",
            "domestic_constraint",
        ),
        allowed_relation_types=(
            "negotiates_with",
            "mediates",
            "spoils",
            "pressures",
            "constrains",
            "escalates",
            "deescalates",
        ),
        min_prerequisites=4,
        max_nodes_by_branch={
            BranchType.LOCAL: 25,
            BranchType.ANALOGUE: 15,
            BranchType.DISRUPTOR: 12,
            BranchType.COUNTERWORLD: 15,
        },
        blind_spot_checks=(
            "spoiler_actor",
            "unrelated_escalatory_channel",
            "sponsor_or_domestic_constraint",
        ),
        stop_rules=(
            "at_least_three_mechanism_analogues_or_unavailable_logged",
            "branch_disagreement_below_threshold_or_logged",
            "budget_cap_reached",
        ),
    ),
    MarketFamily.MACRO_POLICY_PRINT: ConstructionPolicy(
        family=MarketFamily.MACRO_POLICY_PRINT,
        required_branches=(BranchType.LOCAL, BranchType.ANALOGUE, BranchType.DISRUPTOR),
        required_node_categories=(
            "indicator",
            "official_signal",
            "threshold",
            "data_release_calendar",
            "historical_regime",
        ),
        allowed_relation_types=(
            "drives",
            "lags",
            "signals",
            "revises",
            "correlates_with",
            "threshold_for",
        ),
        min_prerequisites=3,
        max_nodes_by_branch={
            BranchType.LOCAL: 20,
            BranchType.ANALOGUE: 15,
            BranchType.DISRUPTOR: 8,
        },
        blind_spot_checks=(
            "release_methodology_change",
            "exogenous_policy_shock",
            "liquidity_or_market_stress",
        ),
        stop_rules=(
            "latest_pit_indicator_coverage_present",
            "analogue_regime_differences_logged",
            "budget_cap_reached",
        ),
    ),
}


def validate_portfolio_against_policy(portfolio: SubgraphPortfolio) -> list[str]:
    """Return validation issues for a portfolio; empty means policy-compliant."""

    policy = POLYMARKET_V1_POLICIES[portfolio.hypothesis.market_frame.family]
    issues: list[str] = []
    present = portfolio.branch_types_present()
    for branch_type in policy.required_branches:
        if branch_type not in present:
            issues.append(f"missing required branch: {branch_type.value}")
    if len(portfolio.prerequisites) < policy.min_prerequisites:
        issues.append(
            f"expected at least {policy.min_prerequisites} prerequisites, got {len(portfolio.prerequisites)}"
        )
    for branch in portfolio.branches:
        max_nodes = policy.max_nodes_by_branch.get(branch.branch_type)
        if max_nodes is not None and len(branch.seed_elements) > max_nodes:
            issues.append(
                f"{branch.branch_id} has {len(branch.seed_elements)} elements; max for "
                f"{branch.branch_type.value} is {max_nodes}"
            )
        if branch.branch_type == BranchType.LOCAL:
            directions = branch.directions_present()
            if Direction.FOR not in directions or Direction.AGAINST not in directions:
                issues.append(
                    f"{branch.branch_id} local branch must include both for and against elements"
                )
    return issues


__all__ = [
    "Branch",
    "BranchType",
    "ConstructionPolicy",
    "Direction",
    "ElementRole",
    "EvidenceRef",
    "EvidenceTier",
    "HypothesisSide",
    "MarketFamily",
    "MarketFrame",
    "OutcomeHypothesis",
    "POLYMARKET_V1_POLICIES",
    "PortfolioElement",
    "Prerequisite",
    "RequirementStressTest",
    "SubgraphPortfolio",
    "validate_portfolio_against_policy",
]
