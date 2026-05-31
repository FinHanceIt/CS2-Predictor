"""
Domain-level calculations for CS2 betting:
  - Vig removal (Shin method, more accurate than basic normalization for ~2-way markets)
  - EV (expected value)
  - Kelly fraction with safety caps per CLAUDE.md confidence tier
"""
from __future__ import annotations
import math
from dataclasses import dataclass


# ---------- Vig removal ----------

def remove_vig_normalization(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Simple inverse-odds normalization (fast, fine for ~2-way bookmaker books)."""
    p_a_raw = 1.0 / odds_a
    p_b_raw = 1.0 / odds_b
    s = p_a_raw + p_b_raw
    return p_a_raw / s, p_b_raw / s


def remove_vig_shin(odds_a: float, odds_b: float, z_max: float = 0.1) -> tuple[float, float]:
    """
    Shin model for vig removal: assumes a fraction z of the book is from
    informed insiders. More accurate for asymmetric books than naive
    normalization. Searches z in [0, z_max] for the value that makes
    the implied probabilities sum to 1.

    Returns (true_p_a, true_p_b).
    """
    raw_a = 1.0 / odds_a
    raw_b = 1.0 / odds_b
    book_sum = raw_a + raw_b
    if abs(book_sum - 1.0) < 1e-9:
        return raw_a, raw_b  # already fair

    def implied(z):
        # Shin's formula
        def transform(p_raw, sum_):
            disc = z * z + 4 * (1 - z) * p_raw * p_raw / sum_
            return (math.sqrt(max(disc, 0)) - z) / (2 * (1 - z))
        return transform(raw_a, book_sum), transform(raw_b, book_sum)

    # Binary search for z that makes implied probs sum to 1
    lo, hi = 0.0, z_max
    for _ in range(60):
        mid = (lo + hi) / 2
        p_a, p_b = implied(mid)
        if p_a + p_b > 1.0:
            lo = mid
        else:
            hi = mid
    return implied((lo + hi) / 2)


# ---------- EV + Kelly ----------

@dataclass
class BetEvaluation:
    market: str
    side: str
    decimal_odds: float
    prob_estimate: float
    implied_prob_fair: float
    edge: float                   # prob_estimate - implied_prob_fair
    ev: float                     # expected return per unit stake
    kelly_full: float             # full-Kelly stake fraction
    kelly_scaled: float           # tier-adjusted stake fraction
    tier: str                     # A / B / C


def expected_value(prob_estimate: float, decimal_odds: float) -> float:
    """EV per unit stake. Positive = profitable in the long run."""
    return prob_estimate * (decimal_odds - 1) - (1 - prob_estimate)


def kelly_fraction(prob_estimate: float, decimal_odds: float) -> float:
    """Full Kelly stake fraction. Returns 0 if EV<=0."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    p = prob_estimate
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, f)


def classify_tier(
    prob_estimate: float,
    edge: float,
    mrt_confidence: float | None,
    n_maps_a: int | None,
    n_maps_b: int | None,
    days_since_roster_change_a: int | None = None,
    days_since_roster_change_b: int | None = None,
) -> str:
    """
    Per CLAUDE.md:
      A-tier (1/4 Kelly): edge ≥ 5pp, both teams ≥ 20 maps in 60d,
                          MRT conf ≥ 0.70, no roster change in 21d.
      B-tier (1/8 Kelly): edge 2–5pp, both teams ≥ 10 maps in 60d,
                          MRT conf ≥ 0.55.
      C-tier (no stake): otherwise.
    """
    if edge < 0.02:
        return "C"
    # Roster check — only downgrade if explicitly known to be < 21d
    if (days_since_roster_change_a is not None and days_since_roster_change_a < 21) or \
       (days_since_roster_change_b is not None and days_since_roster_change_b < 21):
        return "C"
    sample_a = n_maps_a or 0
    sample_b = n_maps_b or 0
    conf = mrt_confidence if mrt_confidence is not None else 0.0
    if edge >= 0.05 and sample_a >= 20 and sample_b >= 20 and conf >= 0.70:
        return "A"
    if edge >= 0.02 and sample_a >= 10 and sample_b >= 10 and conf >= 0.55:
        return "B"
    return "C"


def evaluate_bet(
    market: str,
    side: str,
    decimal_odds: float,
    prob_estimate: float,
    pair_odds: float | None = None,
    mrt_confidence: float | None = None,
    n_maps_a: int | None = None,
    n_maps_b: int | None = None,
) -> BetEvaluation:
    """
    Build a BetEvaluation for one side of one market.

    pair_odds: the opposing side's odds — used to de-vig. If not provided,
    we'll assume the book is fair (rare, mostly for testing).
    """
    if pair_odds is not None:
        # we have both sides, can devig properly
        if side == "a":
            p_fair_a, _ = remove_vig_shin(decimal_odds, pair_odds)
            implied_fair = p_fair_a
        else:
            _, p_fair_b = remove_vig_shin(pair_odds, decimal_odds)
            implied_fair = p_fair_b
    else:
        implied_fair = 1.0 / decimal_odds

    edge = prob_estimate - implied_fair
    ev = expected_value(prob_estimate, decimal_odds)
    full_k = kelly_fraction(prob_estimate, decimal_odds)
    tier = classify_tier(prob_estimate, edge, mrt_confidence, n_maps_a, n_maps_b)
    # Negative-EV bets are always C-tier regardless of edge (the bookmaker's vig
    # ate the edge — staking still loses money in expectation).
    if ev <= 0:
        tier = "C"

    if tier == "A":
        scaled = full_k * 0.25
    elif tier == "B":
        scaled = full_k * 0.125
    else:
        scaled = 0.0
    # absolute cap: never more than 2% of bankroll per single bet
    scaled = min(scaled, 0.02)

    return BetEvaluation(
        market=market, side=side, decimal_odds=decimal_odds,
        prob_estimate=prob_estimate, implied_prob_fair=implied_fair,
        edge=edge, ev=ev, kelly_full=full_k, kelly_scaled=scaled, tier=tier,
    )
