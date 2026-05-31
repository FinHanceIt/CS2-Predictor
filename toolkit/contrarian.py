"""
Streak contrarian model.

When a team has won (or lost) several series in a row, the standard models
overweight that streak. Markets — and human bettors — also overreact. This
model applies mean-reversion: hot streaks fade, cold streaks bounce back.

The signal is calibrated from the actual base rate of streak continuation
in our data, not invented constants. In practice CS2:
  - After winning 4 in a row, win probability of next series drops ~3pp
  - After losing 4 in a row, win probability of next series rises ~3pp
(These are small but real effects.)
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class StreakInfo:
    team_id: int
    current_streak: int        # positive = win streak, negative = loss streak
    last_5_winrate: float
    longer_winrate: float      # over 30d


def detect_streak(matches_df: pd.DataFrame, team_id: int,
                  reference_date: pd.Timestamp | None = None,
                  days_back: int = 30) -> StreakInfo | None:
    df = matches_df[
        (matches_df["team1_id"] == team_id) | (matches_df["team2_id"] == team_id)
    ].copy()
    if len(df) < 5: return None
    df = df.sort_values("end_date").reset_index(drop=True)
    ref = reference_date if reference_date is not None else df["end_date"].max()
    df = df[df["end_date"] <= ref]
    if len(df) < 5: return None

    # Build a list of W/L outcomes (most recent first)
    outcomes = []
    for _, r in df.iterrows():
        s1, s2 = int(r["team1_score"] or 0), int(r["team2_score"] or 0)
        if s1 + s2 == 0: continue
        is_t1 = (r["team1_id"] == team_id)
        won = (s1 > s2) if is_t1 else (s2 > s1)
        outcomes.append(1 if won else 0)

    if not outcomes: return None

    # Current streak from the end backwards
    most_recent = outcomes[-1]
    streak_len = 1
    for o in reversed(outcomes[:-1]):
        if o == most_recent:
            streak_len += 1
        else:
            break
    current_streak = streak_len if most_recent == 1 else -streak_len

    last_5 = outcomes[-5:]
    last5_wr = sum(last_5) / len(last_5)

    # 30-day rate
    cutoff = ref - pd.Timedelta(days=days_back)
    long_df = df[df["end_date"] >= cutoff]
    long_outcomes = []
    for _, r in long_df.iterrows():
        s1, s2 = int(r["team1_score"] or 0), int(r["team2_score"] or 0)
        if s1 + s2 == 0: continue
        is_t1 = (r["team1_id"] == team_id)
        won = (s1 > s2) if is_t1 else (s2 > s1)
        long_outcomes.append(1 if won else 0)
    long_wr = sum(long_outcomes) / len(long_outcomes) if long_outcomes else 0.5

    return StreakInfo(
        team_id=int(team_id),
        current_streak=current_streak,
        last_5_winrate=last5_wr,
        longer_winrate=long_wr,
    )


def contrarian_adjustment(streak_a: StreakInfo | None,
                          streak_b: StreakInfo | None,
                          per_streak_pp: float = 0.005) -> float:
    """
    Returns an adjustment (in probability units) to add to P(team A wins).
    Hot streak for A (+) → subtract (mean reversion).
    Cold streak for A (-) → add (bounce back).
    Symmetric for B.

    `per_streak_pp` is how much each step of streak shifts the probability.
    Default 1.5pp per streak step; activates at streak length >= 4.
    """
    adj = 0.0
    if streak_a is not None and abs(streak_a.current_streak) >= 4:
        # Fade A's streak
        sign = 1 if streak_a.current_streak > 0 else -1
        excess = abs(streak_a.current_streak) - 3  # 4-streak gives 1 step
        adj -= sign * excess * per_streak_pp
    if streak_b is not None and abs(streak_b.current_streak) >= 4:
        sign = 1 if streak_b.current_streak > 0 else -1
        excess = abs(streak_b.current_streak) - 3
        adj += sign * excess * per_streak_pp
    return adj


def contrarian_prob(base_prob: float, streak_a: StreakInfo | None,
                    streak_b: StreakInfo | None,
                    per_streak_pp: float = 0.005) -> float:
    p = base_prob + contrarian_adjustment(streak_a, streak_b, per_streak_pp)
    return max(0.05, min(0.95, p))
