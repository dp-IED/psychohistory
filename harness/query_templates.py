from __future__ import annotations

from harness.policy.blind_spot_checks import (
    base_rate_check,
    current_state_check,
    markets_price_check,
    policy_decision_check,
    resolution_criteria_check,
)

# Template registry — maps blind_spot_check names to callables
# that produce WebSearchRequest from a MarketFrame.
# Populated at import time from the canonical blind-spot check module.
TEMPLATE_REGISTRY: dict = {
    "base_rate_check": base_rate_check,
    "current_state_check": current_state_check,
    "resolution_criteria_check": resolution_criteria_check,
    "markets_price_check": markets_price_check,
    "policy_decision_check": policy_decision_check,
}
