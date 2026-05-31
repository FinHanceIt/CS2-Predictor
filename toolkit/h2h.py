"""
Head-to-head model.

When two teams have played each other multiple times recently, the direct
H2H record is a strong signal — sometimes stronger than global ratings.
Style clashes are real in CS2: team A's map pool / play style might
consistently beat team B even though both have similar global ratings.

Approach: if there are >=3 H2H meetings in the window, blend the H2H
win rate into the global prediction. The blend weight scales with n.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class H2HInfo:
    team_a_id: int
    team_b_id: int
    n_meetings: int            # series count
    a_wins: int                # series won by team A
    a_winrate: float           # series win rate from A's perspective
    avg_score_a: float         # avg maps_for A per series
    avg_score_b: float


def get_h2h(matches_df: pd.DataFrame, team_a_id: int, team_b_id: int,
            reference_date: pd.Timestamp | None = None,
            days_back: int = 90) -> H2HInfo | None:
    df = matches_df[
        ((matches_df["team1_id"] == team_a_id) & (matches_df["team2_id"] == team_b_id)) |
        ((matches_df["team1_id"] == team_b_id) & (matches_df["team2_id"] == team_a_id))
    ].copy()
    if len(df) == 0: return None
    ref = reference_date if reference_date is not None else df["end_date"].max()
    cutoff = ref - pd.Timedelta(days=days_back)
    df = df[(df["end_date"] >= cutoff) & (df["end_date"] <= ref)]
    if len(df) == 0: return None

    a_wins = 0
    a_maps = []
    b_maps = []
    for _, r in df.iterrows():
        s1, s2 = int(r["team1_score"] or 0), int(r["team2_score"] or 0)
        if s1 + s2 == 0: continue
        if r["team1_id"] == team_a_id:
            a_score, b_score = s1, s2
        else:
            a_score, b_score = s2, s1
        a_maps.append(a_score); b_maps.append(b_score)
        if a_score > b_score: a_wins += 1

    n = len(a_maps)
    if n == 0: return None
    return H2HInfo(
        team_a_id=int(team_a_id),
        team_b_id=int(team_b_id),
        n_meetings=n,
        a_wins=a_wins,
        a_winrate=a_wins / n,
        avg_score_a=sum(a_maps) / n,
        avg_score_b=sum(b_maps) / n,
    )


def h2h_adjusted_prob(base_prob: float, h2h: H2HInfo | None,
                      min_n: int = 2, weight_per_match: float = 0.10) -> float:
    """
    Blend H2H winrate into the base prediction.

    Weight scales linearly with number of meetings (capped at 0.5 for 5+):
        weight = min(0.5, weight_per_match * n)
    Below `min_n` (default 3), no adjustment.

    Returns weighted average: p_new = (1-w) * base + w * h2h_winrate.
    """
    if h2h is None or h2h.n_meetings < min_n:
        return base_prob
    w = min(0.5, weight_per_match * h2h.n_meetings)
    return (1 - w) * base_prob + w * h2h.a_winrate
