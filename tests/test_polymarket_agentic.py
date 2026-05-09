from schemas.polymarket_agentic import (
    Branch,
    BranchType,
    Direction,
    ElementRole,
    HypothesisSide,
    MarketFamily,
    MarketFrame,
    OutcomeHypothesis,
    PortfolioElement,
    Prerequisite,
    RequirementStressTest,
    SubgraphPortfolio,
    validate_portfolio_against_policy,
)


def _frame(family: MarketFamily = MarketFamily.EVENT_NEGOTIATION) -> MarketFrame:
    return MarketFrame(
        market_id="m1",
        question="Will there be a ceasefire before August 1?",
        family=family,
        resolution_criteria="Resolves Yes if a ceasefire is announced and active before deadline.",
        outcomes=("Yes", "No"),
    )


def _hypothesis(family: MarketFamily = MarketFamily.EVENT_NEGOTIATION) -> OutcomeHypothesis:
    return OutcomeHypothesis(
        hypothesis_id="h_yes",
        market_frame=_frame(family),
        side=HypothesisSide.YES,
        summary="Yes-world where mediator pressure overcomes spoiler risk.",
    )


def test_market_frame_binary_yes_no() -> None:
    assert _frame().is_binary_yes_no()


def test_local_branch_requires_for_and_against_elements() -> None:
    local = Branch(
        branch_id="local",
        branch_type=BranchType.LOCAL,
        seed_elements=(
            PortfolioElement(
                element_id="mediator_pressure",
                label="Mediator pressure",
                role=ElementRole.DRIVER,
                direction=Direction.FOR,
                rationale="Directly supports ceasefire path.",
            ),
        ),
        rationale="One-sided local branch should fail.",
    )
    disruptor = Branch(
        branch_id="disruptor",
        branch_type=BranchType.DISRUPTOR,
        seed_elements=(),
        rationale="Required disruptor branch present.",
    )
    analogue = Branch(
        branch_id="analogue",
        branch_type=BranchType.ANALOGUE,
        seed_elements=(),
        rationale="Required analogue branch present.",
    )
    portfolio = SubgraphPortfolio(
        portfolio_id="p1",
        hypothesis=_hypothesis(),
        as_of_time="2026-05-09T00:00:00Z",
        branches=(local, disruptor, analogue),
        prerequisites=(
            Prerequisite("pr1", "Talks stay open", Direction.MIXED),
            Prerequisite("pr2", "Mediator stays engaged", Direction.FOR),
            Prerequisite("pr3", "No spoiler attack", Direction.AGAINST),
            Prerequisite("pr4", "Leadership accepts sequencing", Direction.UNKNOWN),
        ),
    )

    assert "local branch must include both for and against elements" in "\n".join(
        validate_portfolio_against_policy(portfolio)
    )


def test_valid_event_negotiation_portfolio_passes_policy() -> None:
    local = Branch(
        branch_id="local",
        branch_type=BranchType.LOCAL,
        seed_elements=(
            PortfolioElement("mediator", "Mediator pressure", ElementRole.DRIVER, Direction.FOR, "Supports deal."),
            PortfolioElement("spoiler", "Spoiler militia", ElementRole.SPOILER, Direction.AGAINST, "Could break talks."),
        ),
        rationale="Contested local world.",
    )
    portfolio = SubgraphPortfolio(
        portfolio_id="p_ok",
        hypothesis=_hypothesis(),
        as_of_time="2026-05-09T00:00:00Z",
        branches=(
            local,
            Branch("analogue", BranchType.ANALOGUE, (), "Prior ceasefire attempts."),
            Branch("disruptor", BranchType.DISRUPTOR, (), "Remote spoiler and escalation risks."),
        ),
        prerequisites=(
            Prerequisite("pr1", "Talks stay open", Direction.MIXED),
            Prerequisite("pr2", "Mediator stays engaged", Direction.FOR),
            Prerequisite("pr3", "No spoiler attack", Direction.AGAINST),
            Prerequisite("pr4", "Leadership accepts sequencing", Direction.UNKNOWN),
        ),
    )

    assert validate_portfolio_against_policy(portfolio) == []


def test_stress_test_probabilities_are_bounded() -> None:
    ok = RequirementStressTest(portfolio_id="p", p_yes=0.62, uncertainty=0.2, missingness_risk=0.4)
    assert ok.p_yes == 0.62

    try:
        RequirementStressTest(portfolio_id="p", p_yes=1.2, uncertainty=0.2)
    except ValueError as exc:
        assert "p_yes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected bounded probability validation")
