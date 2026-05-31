"""
Form-divergence anomaly detector.

Idea: a team whose last 3 series performance is statistically very different
from their 30-day baseline is probably going through a regime change — roster
swap, motivation drop, jet lag, internal drama. The model trained on the
30-day average will be wrong for them.

We compute a z-score:
    z = (recent_3_winrate - baseline_30d_winrate) / baseline_30d_stderr

|z| > 2 = significant divergence. We can use this as either:
  - A confidence downgrade (don't trust predictions on this team)
  - A direction-tilt (if recent < baseline, fade the team)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import pandas as pd


@dataclass
class AnomalyScore:
    team_id: int
    recent_winrate: float       # over last 3 series
    baseline_winrate: float     # over previous 30 days (excluding recent 3)
    z_score: float              # signed z-score: + means improving, - means declining
    n_recent: int               # series in "recent"
    n_baseline: int             # series in "baseline"


def compute_anomaly(matches_df: pd.DataFrame, team_id: int,
                    reference_date: pd.Timestamp | None = None,
                    recent_n: int = 3, baseline_days: int = 30) -> AnomalyScore | None:
    """Returns AnomalyScore or None if insufficient data."""
    df = matches_df[
        (matches_df["team1_id"] == team_id) | (matches_df["team2_id"] == team_id)
    ].copy()
    if len(df) < recent_n + 5:
        return None
    df = df.sort_values("end_date").reset_index(drop=True)
    ref = reference_date if reference_date is not None else df["end_date"].max()
    df = df[df["end_date"] <= ref]
    if len(df) < recent_n + 5:
        return None

    recent = df.tail(recent_n)
    baseline_cutoff = ref - pd.Timedelta(days=baseline_days)
    baseline = df[(df["end_date"] >= baseline_cutoff)].iloc[:-recent_n] if len(df) > recent_n else df.iloc[0:0]

    if len(baseline) < 5:
        return None

    def series_winrate(rows):
        wins, total = 0, 0
        for _, r in rows.iterrows():
            s1, s2 = int(r["team1_score"] or 0), int(r["team2_score"] or 0)
            if s1 + s2 == 0: continue
            is_t1 = (r["team1_id"] == team_id)
            won = (s1 > s2) if is_t1 else (s2 > s1)
            wins += int(won); total += 1
        return wins / total if total else 0.5, total

    rec_wr, n_rec = series_winrate(recent)
    base_wr, n_base = series_winrate(baseline)
    if n_base < 5:
        return None

    # Standard error for binomial proportion
    se = math.sqrt(max(base_wr * (1 - base_wr) / n_base, 1e-9))
    z = (rec_wr - base_wr) / se if se > 0 else 0.0
    return AnomalyScore(
        team_id=int(team_id),
        recent_winrate=rec_wr,
        baseline_winrate=base_wr,
        z_score=z,
        n_recent=n_rec,
        n_baseline=n_base,
    )


def anomaly_adjusted_prob(base_prob: float, anomaly_a: AnomalyScore | None,
                          anomaly_b: AnomalyScore | None,
                          gain: float = 0.015) -> float:
    """
    Adjust the base probability based on each team's anomaly z-scores.
    +z for team A (improving) -> nudge up.
    +z for team B (improving) -> nudge down.

    `gain` is how much each unit of z shifts the probability. A typical
    z=2 with gain=0.04 produces an 8pp adjustment. Capped so we don't go
    past [0.02, 0.98].
    """
    adj = 0.0
    # MEAN REVERSION: extreme recent form tends to regress. A team riding
    # an anomalously hot streak (z > 1) gets faded; cold streaks (z < -1) get
    # a bounce. Backtest confirmed momentum hurt, mean-reversion helps.
    if anomaly_a is not None and abs(anomaly_a.z_score) > 1.0:
        adj -= gain * anomaly_a.z_score
    if anomaly_b is not None and abs(anomaly_b.z_score) > 1.0:
        adj += gain * anomaly_b.z_score
    return max(0.02, min(0.98, base_prob + adj))


def confidence_downgrade(anomaly_a: AnomalyScore | None,
                          anomaly_b: AnomalyScore | None,
                          threshold: float = 2.0) -> float:
    """Returns a multiplier (0-1) to scale confidence by. |z|>threshold = signal
    of regime change; we downgrade existing confidence."""
    z_a = abs(anomaly_a.z_score) if anomaly_a else 0.0
    z_b = abs(anomaly_b.z_score) if anomaly_b else 0.0
    z_max = max(z_a, z_b)
    if z_max <= threshold:
        return 1.0
    # Linear decay: at z=2, mult=1.0; at z=4, mult=0.5
    return max(0.3, 1.0 - 0.25 * (z_max - threshold))
