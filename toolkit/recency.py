"""
Recency weighting helpers.

CLAUDE.md sets λ = 0.015/day (half-life ~46 days). A match N days before the
reference date contributes weight exp(-λ N).

The reference date is the latest match in the team's recent history (not "now"),
so that backtest reproducibility doesn't depend on system time.
"""
from __future__ import annotations
import math
import pandas as pd

DEFAULT_LAMBDA = 0.015  # per day; matches CLAUDE.md
HALF_LIFE_DAYS = math.log(2.0) / DEFAULT_LAMBDA  # ~46.2 days


def weight_for_date(match_date: pd.Timestamp, reference_date: pd.Timestamp,
                    lam: float = DEFAULT_LAMBDA) -> float:
    """Return exp(-λ * days_before_reference). Future matches get weight 1.0."""
    delta = (reference_date - match_date).total_seconds() / 86400.0
    if delta <= 0:
        return 1.0
    return math.exp(-lam * delta)


def weighted_mean(values, weights):
    """Weighted arithmetic mean. Returns nan if all weights are zero."""
    total_w = sum(weights)
    if total_w <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / total_w


def weighted_std(values, weights, mean=None):
    """Weighted std (population). Falls back to 0 on insufficient data."""
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    if mean is None:
        mean = weighted_mean(values, weights)
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_w
    return math.sqrt(max(var, 0.0))
