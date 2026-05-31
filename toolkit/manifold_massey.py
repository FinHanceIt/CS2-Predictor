"""
Manifold-coupled Massey rating.

Idea: standard Massey solves M·r = b for ratings r given the game-result matrix.
This works well for teams with many games but treats sparsely-played teams as
independent — a team with 3 games can swing wildly because its rating depends
only on those 3 results.

We add a graph-Laplacian regularizer L built from Fisher-Rao distances between
team manifolds:
    (M + λL)·r = b

The matrix L[i,j] = -w_ij for i≠j and L[i,i] = Σ_j w_ij, where
    w_ij = 1 / (1 + d_FR(M_i, M_j)^2)
captures geometric similarity (closer manifold → higher coupling).

Effect: ratings of geometrically-similar teams get pulled toward each other.
A 3-game team with a profile matching Spirit's manifold rates closer to
Spirit, not in isolation.

λ is fit by minimizing OOS log-loss on a backtest.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from toolkit.cs2_mrt import build_team_manifold, fisher_rao_distance_1d, GaussianH2
from toolkit.massey import MasseyModel
from toolkit.recency import weight_for_date, DEFAULT_LAMBDA


def _team_signature(manifold) -> tuple[GaussianH2, GaussianH2]:
    """Two Gaussians per team (offense, defense)."""
    return (manifold.offense, manifold.defense)


def manifold_distance(m_a, m_b) -> float:
    """Symmetric distance between two teams in (offense, defense) space."""
    d_off = fisher_rao_distance_1d(m_a.offense, m_b.offense)
    d_def = fisher_rao_distance_1d(m_a.defense, m_b.defense)
    return math.sqrt(d_off ** 2 + d_def ** 2)


def fit_manifold_massey(
    matches_df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    lam_recency: float = DEFAULT_LAMBDA,
    days_back: int = 60,
    lam_coupling: float = 0.5,
    ridge: float = 1e-3,
    sim_cutoff: float = 3.0,
    min_games_for_manifold: int = 3,
) -> MasseyModel:
    """
    Fit manifold-coupled Massey.

    `lam_coupling`  : strength of the geometric regularizer (0 = vanilla Massey).
    `sim_cutoff`    : pairs of teams with d_FR > sim_cutoff are NOT coupled
                      (improves matrix conditioning, avoids weak global pull).
    `ridge`         : small diagonal regularization for stability.
    """
    if reference_date is None:
        reference_date = matches_df["end_date"].max()
    cutoff = reference_date - pd.Timedelta(days=days_back)
    df = matches_df[
        (matches_df["end_date"] >= cutoff) &
        (matches_df["end_date"] <= reference_date)
    ].copy()

    team_ids = sorted(set(df["team1_id"].dropna().astype(int)) |
                      set(df["team2_id"].dropna().astype(int)))
    idx = {tid: i for i, tid in enumerate(team_ids)}
    n = len(team_ids)
    if n == 0:
        return MasseyModel(ratings={}, reference_date=reference_date)

    # === Massey matrix M and target b ===
    M = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for _, row in df.iterrows():
        t1 = int(row["team1_id"]); t2 = int(row["team2_id"])
        s1 = float(row["team1_score"] or 0); s2 = float(row["team2_score"] or 0)
        if s1 + s2 == 0:
            continue
        w = weight_for_date(row["end_date"], reference_date, lam_recency)
        margin = s1 - s2
        i, j = idx[t1], idx[t2]
        M[i, i] += w; M[j, j] += w
        M[i, j] -= w; M[j, i] -= w
        b[i] += w * margin; b[j] -= w * margin

    n_games_per_team = {team_ids[i]: float(M[i, i]) for i in range(n)}

    # === Build manifolds for coupling ===
    manifolds = {}
    for tid in team_ids:
        m = build_team_manifold(matches_df, tid, days_back=days_back,
                                reference_date=reference_date)
        if m is not None:
            manifolds[tid] = m

    # === Graph Laplacian L from manifold distances ===
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        tid_i = team_ids[i]
        if tid_i not in manifolds:
            continue
        for j in range(i + 1, n):
            tid_j = team_ids[j]
            if tid_j not in manifolds:
                continue
            d = manifold_distance(manifolds[tid_i], manifolds[tid_j])
            if d > sim_cutoff:
                continue
            w = 1.0 / (1.0 + d * d)
            L[i, j] = -w
            L[j, i] = -w
            L[i, i] += w
            L[j, j] += w

    # === Combine and solve (M + λL)·r = b ===
    A = M + lam_coupling * L
    # Replace last row with sum constraint
    A[-1, :] = 1.0
    b_aug = b.copy(); b_aug[-1] = 0.0
    # Ridge regularization
    for i in range(n - 1):
        A[i, i] += ridge
    try:
        r = np.linalg.solve(A, b_aug)
    except np.linalg.LinAlgError:
        r = np.linalg.pinv(A) @ b_aug

    ratings = {team_ids[i]: float(r[i]) for i in range(n)}
    return MasseyModel(
        ratings=ratings,
        reference_date=reference_date,
        beta=1.0,
        n_games_per_team=n_games_per_team,
    )
