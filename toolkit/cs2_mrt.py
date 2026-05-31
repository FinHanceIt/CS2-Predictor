"""
MRT adapted for Counter-Strike 2.

Team encoding:
  offense_gauss = N(maps_won_per_series_avg, std)         # series dominance
  defense_gauss = N(maps_lost_per_series_avg, std)        # series leakage

The geodesic interpolation produces per-team expected maps-won, which we feed
into a Bo3 / Bo5 series simulator (Bernoulli race to N).

Series stats are recency-weighted (λ = 0.015/day from CLAUDE.md). All map-level
metrics derive from team1_score / team2_score, NOT from the unreliable
maps_score field in bo3.gg's API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import pandas as pd

from toolkit.recency import weight_for_date, weighted_mean, weighted_std, DEFAULT_LAMBDA


# ---------- Pure geometry (re-implemented to keep CS2-Predictor self-contained) ----------

@dataclass(frozen=True)
class GaussianH2:
    mu: float
    sigma: float
    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
    def to_halfplane(self) -> tuple[float, float]:
        return self.mu / math.sqrt(2.0), self.sigma


def fisher_rao_distance_1d(g1: GaussianH2, g2: GaussianH2) -> float:
    x1, y1 = g1.to_halfplane()
    x2, y2 = g2.to_halfplane()
    num = (x1 - x2) ** 2 + (y1 - y2) ** 2
    den = 2.0 * y1 * y2
    return math.acosh(max(1.0 + num / den, 1.0))


def geodesic_point_1d(g1: GaussianH2, g2: GaussianH2, t: float) -> GaussianH2:
    x1, y1 = g1.to_halfplane()
    x2, y2 = g2.to_halfplane()
    if abs(x1 - x2) < 1e-12:
        y_t = (y1 ** (1.0 - t)) * (y2 ** t)
        return GaussianH2(mu=math.sqrt(2.0) * x1, sigma=y_t)
    c = ((x1 * x1 + y1 * y1) - (x2 * x2 + y2 * y2)) / (2.0 * (x1 - x2))
    r = math.sqrt((x1 - c) ** 2 + y1 ** 2)
    theta1 = math.atan2(y1, x1 - c)
    theta2 = math.atan2(y2, x2 - c)
    th1_half = max(math.tan(theta1 / 2.0), 1e-12)
    th2_half = max(math.tan(theta2 / 2.0), 1e-12)
    tan_half_t = (th1_half ** (1.0 - t)) * (th2_half ** t)
    theta_t = 2.0 * math.atan(tan_half_t)
    x_t = c + r * math.cos(theta_t)
    y_t = r * math.sin(theta_t)
    return GaussianH2(mu=math.sqrt(2.0) * x_t, sigma=max(y_t, 1e-9))


def _hyperbolic_angle(a: float, b: float, c: float) -> float:
    sinh_b = math.sinh(b); sinh_c = math.sinh(c)
    if sinh_b * sinh_c < 1e-12:
        return 0.0
    cos_A = (math.cosh(b) * math.cosh(c) - math.cosh(a)) / (sinh_b * sinh_c)
    cos_A = max(-1.0, min(1.0, cos_A))
    return math.acos(cos_A)


def holonomy_defect(path: Sequence[GaussianH2]) -> float:
    if len(path) < 3:
        return 0.0
    p_first, p_last = path[0], path[-1]
    b = fisher_rao_distance_1d(p_first, p_last)
    if b < 1e-6:
        return 0.0
    best = 0.0
    for k in range(1, len(path) - 1):
        p_mid = path[k]
        a = fisher_rao_distance_1d(p_mid, p_last)
        c = fisher_rao_distance_1d(p_first, p_mid)
        if min(a, c) < 1e-6:
            continue
        if not (a + b > c + 1e-9 and a + c > b + 1e-9 and b + c > a + 1e-9):
            continue
        alpha = _hyperbolic_angle(a, b, c)
        beta = _hyperbolic_angle(b, a, c)
        gamma = _hyperbolic_angle(c, a, b)
        defect = math.pi - (alpha + beta + gamma)
        if defect > best:
            best = defect
    return max(best, 0.0)


# ---------- CS2-specific team encoding -------------------------------------

@dataclass
class CS2TeamManifold:
    team_id: int
    name: str
    offense: GaussianH2          # maps won per series (recent, recency-weighted)
    defense: GaussianH2          # maps lost per series (recent, recency-weighted)
    map_win_rate: float          # recency-weighted map win rate
    n_recent_maps: int           # raw count of maps played in window
    recent_offense_path: list = field(default_factory=list)
    recent_defense_path: list = field(default_factory=list)
    last_match_date: pd.Timestamp | None = None

    def holonomy(self) -> tuple[float, float]:
        return holonomy_defect(self.recent_offense_path), holonomy_defect(self.recent_defense_path)


def build_team_manifold(
    matches_df: pd.DataFrame,
    team_id: int,
    team_name: str = "",
    days_back: int = 30,
    last_n_for_path: int = 8,
    path_window: int = 3,
    min_sigma: float = 0.25,
    lam: float = DEFAULT_LAMBDA,
    reference_date: pd.Timestamp | None = None,
) -> CS2TeamManifold | None:
    """Build a recency-weighted CS2TeamManifold from the cleaned matches parquet."""
    df = matches_df[
        (matches_df["team1_id"] == team_id) | (matches_df["team2_id"] == team_id)
    ].copy()
    if len(df) < 3:
        return None

    df = df.sort_values("end_date").reset_index(drop=True)
    ref = reference_date if reference_date is not None else df["end_date"].max()
    cutoff = ref - pd.Timedelta(days=days_back)
    df = df[(df["end_date"] >= cutoff) & (df["end_date"] <= ref)]
    if len(df) < 3:
        return None

    def for_against(row):
        if row["team1_id"] == team_id:
            return float(row["team1_score"] or 0), float(row["team2_score"] or 0)
        return float(row["team2_score"] or 0), float(row["team1_score"] or 0)

    fa = df.apply(for_against, axis=1, result_type="expand")
    fa.columns = ["maps_for", "maps_against"]
    weights = [weight_for_date(d, ref, lam) for d in df["end_date"]]

    mu_off = weighted_mean(fa["maps_for"].tolist(), weights)
    sg_off = max(weighted_std(fa["maps_for"].tolist(), weights, mu_off), min_sigma)
    mu_def = weighted_mean(fa["maps_against"].tolist(), weights)
    sg_def = max(weighted_std(fa["maps_against"].tolist(), weights, mu_def), min_sigma)

    # Map-level win rate (recency-weighted, derived from series scores)
    w_won = 0.0
    w_played = 0.0
    raw_played = 0
    for (_, row), w in zip(df.iterrows(), weights):
        s1 = int(row["team1_score"] or 0)
        s2 = int(row["team2_score"] or 0)
        played = s1 + s2
        if played == 0:
            continue
        is_t1 = (row["team1_id"] == team_id)
        won = s1 if is_t1 else s2
        w_won += w * won
        w_played += w * played
        raw_played += played
    map_win_rate = w_won / w_played if w_played > 0 else 0.5

    # Holonomy path: rolling weighted windows over last N series
    recent = fa.tail(last_n_for_path).reset_index(drop=True)
    recent_dates = df["end_date"].tail(last_n_for_path).tolist()
    recent_weights = [weight_for_date(d, ref, lam) for d in recent_dates]
    off_path, def_path = [], []
    for start in range(0, len(recent) - path_window + 1):
        chunk = recent.iloc[start:start + path_window]
        chunk_w = recent_weights[start:start + path_window]
        mu_o = weighted_mean(chunk["maps_for"].tolist(), chunk_w)
        sg_o = max(weighted_std(chunk["maps_for"].tolist(), chunk_w, mu_o), min_sigma)
        mu_d = weighted_mean(chunk["maps_against"].tolist(), chunk_w)
        sg_d = max(weighted_std(chunk["maps_against"].tolist(), chunk_w, mu_d), min_sigma)
        off_path.append(GaussianH2(mu=mu_o, sigma=sg_o))
        def_path.append(GaussianH2(mu=mu_d, sigma=sg_d))

    return CS2TeamManifold(
        team_id=int(team_id),
        name=team_name or f"Team#{team_id}",
        offense=GaussianH2(mu=mu_off, sigma=sg_off),
        defense=GaussianH2(mu=mu_def, sigma=sg_def),
        map_win_rate=map_win_rate,
        n_recent_maps=raw_played,
        recent_offense_path=off_path,
        recent_defense_path=def_path,
        last_match_date=df["end_date"].max(),
    )


def resonance_scalar(a: CS2TeamManifold, b: CS2TeamManifold) -> float:
    va = np.array([a.offense.sigma ** 2, a.defense.sigma ** 2])
    vb = np.array([b.offense.sigma ** 2, b.defense.sigma ** 2])
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def prediction_curvature(g_match: GaussianH2, n_a: float, n_b: float) -> float:
    if n_a <= 0 or n_b <= 0:
        return 0.0
    n_h = 2.0 / (1.0 / n_a + 1.0 / n_b)
    sat = n_h / (n_h + 25.0)
    return sat * (0.5 + 0.5 * math.exp(-g_match.sigma))


def manifold_confidence(kappa: float, rho: float) -> float:
    raw = kappa * abs(rho)
    return 1.0 / (1.0 + math.exp(-6.0 * (raw - 0.3)))


# ---------- Series simulator (Bo3 / Bo5 / Bo1) ----------------------------

def simulate_series_probs(p_map_a: float, bo: int) -> tuple[float, float, float, float]:
    p = max(0.01, min(0.99, p_map_a))
    q = 1 - p
    if bo == 1:
        return p, q, p, 0.0
    if bo == 3:
        p_a_2_0 = p * p
        p_a_2_1 = 2 * p * p * q
        p_a_total = p_a_2_0 + p_a_2_1
        return p_a_total, 1 - p_a_total, p_a_2_0, p_a_2_1
    if bo == 5:
        p_a_3_0 = p ** 3
        p_a_3_1 = 3 * (p ** 3) * q
        p_a_3_2 = 6 * (p ** 3) * (q ** 2)
        p_a_total = p_a_3_0 + p_a_3_1 + p_a_3_2
        return p_a_total, 1 - p_a_total, p_a_3_0, p_a_3_1 + p_a_3_2
    raise ValueError(f"unsupported bo: {bo}")


# ---------- Full MRT prediction --------------------------------------------

def predict_series(
    team_a: CS2TeamManifold,
    team_b: CS2TeamManifold,
    bo: int = 3,
    holonomy_gain: float = 0.4,
) -> dict:
    d_off = fisher_rao_distance_1d(team_a.offense, team_b.defense)
    d_def = fisher_rao_distance_1d(team_a.defense, team_b.offense)

    h_a_off, h_a_def = team_a.holonomy()
    h_b_off, h_b_def = team_b.holonomy()
    h_a = h_a_off + h_a_def
    h_b = h_b_off + h_b_def

    t_off = 0.5 + math.tanh(holonomy_gain * (h_a - h_b)) * 0.35
    t_def = 0.5 - math.tanh(holonomy_gain * (h_a - h_b)) * 0.35
    t_off = max(0.05, min(0.95, t_off))
    t_def = max(0.05, min(0.95, t_def))

    match_off = geodesic_point_1d(team_a.offense, team_b.defense, t_off)
    match_def = geodesic_point_1d(team_a.defense, team_b.offense, t_def)

    lam_a_maps = max(0.05, match_off.mu)
    lam_b_maps = max(0.05, match_def.mu)

    rho = resonance_scalar(team_a, team_b)
    kappa = prediction_curvature(match_off, team_a.n_recent_maps, team_b.n_recent_maps)
    confidence = manifold_confidence(kappa, rho)

    p_map_a = lam_a_maps / (lam_a_maps + lam_b_maps)
    p_map_a = max(0.05, min(0.95, p_map_a))

    p_a_series, p_b_series, p_a_clean, p_a_close = simulate_series_probs(p_map_a, bo)

    return {
        "team_a": team_a.name,
        "team_b": team_b.name,
        "bo": bo,
        "d_FR_offense": d_off,
        "d_FR_defense": d_def,
        "h_a": h_a,
        "h_b": h_b,
        "t_off": t_off,
        "t_def": t_def,
        "lambda_a_maps": lam_a_maps,
        "lambda_b_maps": lam_b_maps,
        "p_map_a": p_map_a,
        "p_map_b": 1 - p_map_a,
        "resonance": rho,
        "kappa": kappa,
        "confidence": confidence,
        "p_a_series": p_a_series,
        "p_b_series": p_b_series,
        "p_a_clean_sweep": p_a_clean,
        "p_a_close": p_a_close,
        "team_a_recent_maps": team_a.n_recent_maps,
        "team_b_recent_maps": team_b.n_recent_maps,
        "team_a_winrate": team_a.map_win_rate,
        "team_b_winrate": team_b.map_win_rate,
    }


if __name__ == "__main__":
    g1 = GaussianH2(mu=1.5, sigma=0.4)
    g2 = GaussianH2(mu=1.0, sigma=0.6)
    print(f"FR distance: {fisher_rao_distance_1d(g1, g2):.4f}")
    mid = geodesic_point_1d(g1, g2, 0.5)
    print(f"Geodesic mid: mu={mid.mu:.3f}, sigma={mid.sigma:.3f}")
