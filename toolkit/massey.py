"""
Massey rating system for CS2 teams.

Standard Massey: solve M·r = b where
  M[i,i] = total game count for team i
  M[i,j] = -(games between i and j)
  b[i]  = total point margin for team i (here: maps won - maps lost across all series)

With one row replaced by sum(r)=0 for uniqueness, the solution `r` gives each
team a rating such that r_i - r_j ≈ expected map margin per series of i vs j.

Recency weighting: each game contributes weight exp(-λ * days_before_reference)
to both the matrix counts and the margin sums.

Conversion to per-map win prob via logistic:
  P(team A wins a map) = 1 / (1 + exp(-β * (r_A - r_B)))
β is fit so that the average |r_i - r_j| corresponds to a reasonable spread.
A reference β can be derived from backtest, or set heuristically (~1.0).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from toolkit.recency import weight_for_date, DEFAULT_LAMBDA


@dataclass
class MasseyModel:
    """Massey rating output."""
    ratings: dict[int, float]                  # team_id -> rating
    reference_date: pd.Timestamp
    beta: float = 1.0                          # logistic scale
    n_games_per_team: dict[int, float] = field(default_factory=dict)  # weighted

    def rating_diff(self, team_a_id: int, team_b_id: int) -> float:
        ra = self.ratings.get(int(team_a_id), 0.0)
        rb = self.ratings.get(int(team_b_id), 0.0)
        return ra - rb

    def p_map_a(self, team_a_id: int, team_b_id: int) -> float:
        diff = self.rating_diff(team_a_id, team_b_id)
        return 1.0 / (1.0 + math.exp(-self.beta * diff))


def fit_massey(
    matches_df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    lam: float = DEFAULT_LAMBDA,
    days_back: int = 90,
    min_total_games: float = 1.0,
    ridge: float = 1e-3,
) -> MasseyModel:
    """
    Fit a Massey rating system from the matches dataframe.

    Each series contributes its score margin (team1_score - team2_score) as the
    "point margin" for team1's row (and the negative for team2's row).
    Recency-weighted by exp(-lam * days_before_reference).

    `ridge` adds a small diagonal regularization for numerical stability and
    to handle disconnected components in the team graph.
    """
    if reference_date is None:
        reference_date = matches_df["end_date"].max()
    cutoff = reference_date - pd.Timedelta(days=days_back)
    df = matches_df[
        (matches_df["end_date"] >= cutoff) &
        (matches_df["end_date"] <= reference_date)
    ].copy()

    # Collect all team ids
    team_ids = sorted(set(df["team1_id"].dropna().astype(int)) |
                      set(df["team2_id"].dropna().astype(int)))
    idx = {tid: i for i, tid in enumerate(team_ids)}
    n = len(team_ids)
    if n == 0:
        return MasseyModel(ratings={}, reference_date=reference_date)

    M = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for _, row in df.iterrows():
        t1 = int(row["team1_id"]); t2 = int(row["team2_id"])
        s1 = float(row["team1_score"] or 0); s2 = float(row["team2_score"] or 0)
        if s1 + s2 == 0:
            continue
        w = weight_for_date(row["end_date"], reference_date, lam)
        margin = s1 - s2  # team1's perspective

        i, j = idx[t1], idx[t2]
        M[i, i] += w
        M[j, j] += w
        M[i, j] -= w
        M[j, i] -= w
        b[i] += w * margin
        b[j] -= w * margin

    # Per-team weighted game count
    n_games = {team_ids[i]: float(M[i, i]) for i in range(n)}

    # Constraint: replace last row with sum(r)=0 for uniqueness
    M[-1, :] = 1.0
    b[-1] = 0.0

    # Ridge regularization on diagonal (except constraint row)
    for i in range(n - 1):
        M[i, i] += ridge

    try:
        r = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        # singular matrix: fall back to pseudo-inverse
        r = np.linalg.pinv(M) @ b

    ratings = {team_ids[i]: float(r[i]) for i in range(n)}

    # Force teams with too-few games to rating 0 (and we'll filter at predict time)
    for tid in team_ids:
        if n_games[tid] < min_total_games:
            ratings[tid] = 0.0

    return MasseyModel(
        ratings=ratings,
        reference_date=reference_date,
        beta=1.0,
        n_games_per_team=n_games,
    )


def calibrate_beta(model: MasseyModel, matches_df: pd.DataFrame,
                   max_iters: int = 50, lr: float = 0.05) -> float:
    """
    Fit β by minimizing log loss on each map in the training set.
    Each series contributes (s1+s2) Bernoulli observations: team1 has p_map
    based on rating diff, observed s1 wins / s2 losses.
    """
    pairs = []
    for _, row in matches_df.iterrows():
        s1 = int(row["team1_score"] or 0); s2 = int(row["team2_score"] or 0)
        if s1 + s2 == 0: continue
        ra = model.ratings.get(int(row["team1_id"]), 0.0)
        rb = model.ratings.get(int(row["team2_id"]), 0.0)
        pairs.append((ra - rb, s1, s2))
    if not pairs:
        return 1.0
    beta = 1.0
    for _ in range(max_iters):
        grad = 0.0; n_obs = 0
        for diff, s1, s2 in pairs:
            p = 1.0 / (1.0 + math.exp(-beta * diff))
            # dL/dβ = sum over maps of (p - y) * diff
            grad += (p - (s1 / (s1 + s2))) * diff * (s1 + s2)
            n_obs += s1 + s2
        grad /= max(n_obs, 1)
        beta -= lr * grad
        beta = max(0.05, min(beta, 5.0))
    model.beta = float(beta)
    return float(beta)
