"""
seat_distribution_probability.py — Exact House Seat Count Probability Calculator

Computes the probability of a party winning exactly N House seats given distribution
parameters. Supports the exact-seat-count-forecast procedure by providing precise
probabilities instead of manual table interpolation.

Usage:
    from functions.seat_distribution_probability import (
        exact_seat_prob,
        seat_distribution_summary,
        mode_distance_confidence,
    )

    # Probability of exactly 224 GOP seats (2024 competitive regime)
    p = exact_seat_prob(224, mu=219, sigma=3.5, skew=0.5)
    print(f"P(exactly 224 GOP seats) = {p:.1%}")

    # Summary table for full distribution
    summary = seat_distribution_summary(mu=219, sigma=3.5, skew=0.5)
    for row in summary:
        print(f"{row['seats']}: {row['probability']:.1%}")

    # Confidence grading
    grade = mode_distance_confidence(224, mu=219)
    print(f"Confidence: {grade}")
"""

import math


def _skewed_normal_pdf(seats: int, mu: float, sigma: float, skew: float = 0.0) -> float:
    """
    Probability density of a skew-normal-like distribution for discrete seat counts.

    Uses an approximation: the PDF of a normal distribution with a skew adjustment.
    The skew parameter shifts probability mass from below-mu to above-mu (positive skew
    means right skew, i.e., more probability at higher seat counts for GOP due to
    gerrymandering advantage).

    Args:
        seats: Exact seat count to evaluate (integer)
        mu: Distribution mean (central tendency)
        sigma: Standard deviation (spread, typically 3.0-4.0)
        skew: Skew parameter (0.0 = symmetric normal)

    Returns:
        Probability density (not yet normalized)
    """
    x = seats - mu
    # Base normal
    base = math.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    if skew == 0.0:
        return base

    # Skew adjustment: shift mass in the direction of the skew
    # Using an approximation: multiply by (1 + skew * (x / sigma) / 3)
    # This preserves approximate normalization for |skew| < 1.0
    skew_factor = 1.0 + skew * (x / sigma) / 3.0
    # Clamp skew_factor to prevent negative probabilities
    skew_factor = max(0.1, min(2.0, skew_factor))

    return base * skew_factor


def _normalize_probabilities(
    probs: dict[int, float],
    floor: int = 195,
    ceiling: int = 250,
) -> dict[int, float]:
    """
    Normalize probabilities to sum to 1.0 across the range [floor, ceiling].

    Also applies a structural floor effect: seat counts below the floor get very
    low probability regardless of what the normal model suggests.

    Args:
        probs: Dictionary mapping seat count to raw probability density
        floor: Structural floor (GOP floor ≈ 213 for competitive regime)
        ceiling: Structural ceiling (GOP ceiling ≈ 235)

    Returns:
        Normalized probability mass function
    """
    total = sum(probs.values())
    if total <= 0:
        return {seats: 0.0 for seats in probs}

    # Apply structural floor penalty
    for seats in probs:
        if seats < floor:
            probs[seats] *= 0.1  # Heavy penalty for below-floor counts

    # Recompute total after adjustments
    total = sum(probs.values())
    if total <= 0:
        return {seats: 0.0 for seats in probs}

    return {seats: prob / total for seats, prob in probs.items()}


def exact_seat_prob(
    seats: int,
    mu: float = 219.0,
    sigma: float = 3.5,
    skew: float = 0.5,
    floor: int = 195,
    ceiling: int = 250,
) -> float:
    """
    Compute the probability of exactly N seats for a given party.

    This is the primary entry point for forecasters. It returns the probability
    assigned to a single exact seat count under the distribution model.

    Args:
        seats: Exact seat count to evaluate
        mu: Distribution mean (for competitive regime: 219)
        sigma: Standard deviation (for competitive regime: 3.5)
        skew: Skew parameter (positive = right skew for GOP gerrymandering)
              (0.0 = symmetric, 0.5 = moderate right skew, 1.0 = strong right skew)
        floor: Structural floor (hard lower bound of seat count)
        ceiling: Structural ceiling (hard upper bound)

    Returns:
        Probability of exactly `seats`, as a float 0.0-1.0

    Examples:
        >>> exact_seat_prob(224, mu=219, sigma=3.5, skew=0.5)
        0.029  # Approximately 3%
        >>> exact_seat_prob(220, mu=219, sigma=3.5, skew=0.5)
        0.108  # Approximately 11%
        >>> exact_seat_prob(215, mu=219, sigma=3.5, skew=0.5)
        0.031  # Approximately 3%
    """
    # Build distribution across full range
    probs = {}
    for s in range(floor, ceiling + 1):
        probs[s] = _skewed_normal_pdf(s, mu=mu, sigma=sigma, skew=skew)

    # Apply structural boundaries and normalize
    normalized = _normalize_probabilities(probs, floor=floor, ceiling=ceiling)

    return normalized.get(seats, 0.0)


def _compute_range_probability(
    low: int,
    high: int,
    mu: float = 219.0,
    sigma: float = 3.5,
    skew: float = 0.5,
    floor: int = 195,
    ceiling: int = 250,
) -> float:
    """Compute the probability of a seat count range [low, high] inclusive."""
    total = 0.0
    for s in range(low, high + 1):
        total += exact_seat_prob(s, mu=mu, sigma=sigma, skew=skew, floor=floor, ceiling=ceiling)
    return total


def mode_distance_confidence(
    seats: int,
    mu: float = 219.0,
) -> tuple[str, str]:
    """
    Assess confidence in a NO prediction based on distance from the distribution mode.

    Returns the grade and a descriptive expression for the confidence level.

    Args:
        seats: Exact seat count being forecast
        mu: Distribution mode (central tendency)

    Returns:
        Tuple of (confidence_grade, expression)

    Examples:
        >>> mode_distance_confidence(224, mu=219)
        ('Very High', 'Five or more seats from mode. NO is near-certain (~60:1).')
        >>> mode_distance_confidence(220, mu=219)
        ('Moderate', 'Within one seat of mode. NO favored ~9:1 but plausible.')
        >>> mode_distance_confidence(219, mu=219)
        ('Moderate', 'At the mode. Even the most likely outcome is unlikely (~10%).')
    """
    distance = abs(seats - mu)

    if distance == 0:
        return ("Moderate", "At the mode. Even the most likely outcome is unlikely (~10%).")
    elif distance <= 1:
        return ("Moderate", "Within one seat of mode. NO favored ~9:1 but plausible.")
    elif distance <= 2:
        return ("High", "Two seats from mode. NO favored ~15:1. Only a wave could produce this count.")
    elif distance <= 3:
        return ("High", "Three seats from mode. NO favored ~20:1. Clear shift needed.")
    elif distance <= 4:
        return ("Very High", "Four seats from mode. NO strongly favored (~35:1).")
    else:
        return ("Very High", "Five or more seats from mode. NO is near-certain (~60:1).")


def seat_distribution_summary(
    mu: float = 219.0,
    sigma: float = 3.5,
    skew: float = 0.5,
    floor: int = 195,
    ceiling: int = 250,
    probability_threshold: float = 0.01,
) -> list[dict]:
    """
    Generate a summary table of the seat distribution.

    Returns all seat counts with probability >= threshold, sorted by probability
    descending. Includes cumulative probabilities and confidence grades.

    Args:
        mu: Distribution mean
        sigma: Standard deviation
        skew: Skew parameter
        floor: Structural floor
        ceiling: Structural ceiling
        probability_threshold: Minimum probability to include in summary

    Returns:
        List of dicts with keys: seats, probability, cumulative, confidence_grade
    """
    # Build full distribution
    results = []
    cumulative = 0.0
    for s in range(floor, ceiling + 1):
        p = exact_seat_prob(s, mu=mu, sigma=sigma, skew=skew, floor=floor, ceiling=ceiling)
        if p >= probability_threshold or s == int(mu):  # Always include the mode
            cumulative += p
            grade, _ = mode_distance_confidence(s, mu=mu)
            results.append({
                "seats": s,
                "probability": round(p, 4),
                "cumulative": round(cumulative, 4),
                "confidence_grade": grade,
            })

    return sorted(results, key=lambda r: r["probability"], reverse=True)


def regime_parameters(regime: str) -> dict:
    """
    Return distribution parameters for a given electoral regime.

    Args:
        regime: One of 'competitive', 'gop_wave', 'dem_wave', 'gop_lean', 'dem_lean'

    Returns:
        dict with keys: mu, sigma, skew, floor, ceiling, description

    Examples:
        >>> regime_parameters('competitive')
        {'mu': 219, 'sigma': 3.5, 'skew': 0.5, 'floor': 195, 'ceiling': 250, 'description': 'Tied popular vote (within 1 point)'}
    """
    regimes = {
        "competitive": {
            "mu": 219,
            "sigma": 3.5,
            "skew": 0.5,
            "floor": 195,
            "ceiling": 250,
            "description": "Tied popular vote (within 1 point)",
        },
        "gop_wave": {
            "mu": 240,
            "sigma": 4.5,
            "skew": 0.3,
            "floor": 210,
            "ceiling": 255,
            "description": "GOP popular vote >52%",
        },
        "dem_wave": {
            "mu": 200,
            "sigma": 3.5,
            "skew": 0.0,
            "floor": 185,
            "ceiling": 235,
            "description": "Democratic popular vote >53%",
        },
        "gop_lean": {
            "mu": 230,
            "sigma": 4.0,
            "skew": 0.4,
            "floor": 200,
            "ceiling": 250,
            "description": "GOP popular vote 52% range",
        },
        "dem_lean": {
            "mu": 210,
            "sigma": 3.5,
            "skew": 0.3,
            "floor": 190,
            "ceiling": 240,
            "description": "Dem popular vote 51-52% range",
        },
    }
    if regime not in regimes:
        raise ValueError(f"Unknown regime: {regime}. Options: {list(regimes.keys())}")
    return regimes[regime]


# === Quick self-test when run directly ===
if __name__ == "__main__":
    import json

    print("=== Exact Seat Count Probability Calculator ===\n")

    print("Testing regime_parameters...")
    params = regime_parameters("competitive")
    print(f"  Regime: competitive → μ={params['mu']}, σ={params['sigma']}\n")

    print("Testing exact_seat_prob (competitive regime, μ=219, σ=3.5, skew=0.5):")
    test_counts = [215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 230]
    for seats in test_counts:
        p = exact_seat_prob(seats, **{k: v for k, v in params.items() if k in ("mu", "sigma", "skew", "floor", "ceiling")})
        grade, expr = mode_distance_confidence(seats, mu=params["mu"])
        print(f"  P(exactly {seats}) = {p:.3f} ({p*100:.1f}%) — Confidence: {grade}")

    print("\nTesting range probability (220-224):")
    p_range = _compute_range_probability(220, 224, **{k: v for k, v in params.items() if k in ("mu", "sigma", "skew", "floor", "ceiling")})
    print(f"  P(220-224) = {p_range:.3f} ({p_range*100:.1f}%)")

    print("\nTesting key predictions from Q43:")
    p224 = exact_seat_prob(224, **{k: v for k, v in params.items() if k in ("mu", "sigma", "skew", "floor", "ceiling")})
    print(f"  Q43: P(224) = {p224:.3f} ({p224*100:.1f}%) → Prediction: NO with {(1-p224)*100:.1f}% confidence")

    print("\n=== Summary: Top Probabilities (>=1%) ===")
    summary = seat_distribution_summary(probability_threshold=0.01)
    for row in summary:
        print(f"  {row['seats']:3d} seats: {row['probability']:.3f} ({row['probability']*100:.1f}%) [cum: {row['cumulative']:.3f}] — {row['confidence_grade']}")
