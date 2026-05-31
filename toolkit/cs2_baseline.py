"""
Flat-space baseline predictor for CS2.

Series scores are used (NOT bo3.gg's maps_score field — that one is unreliable
and was the source of the original bug).

Two important features beyond a naive baseline:
  - Recency decay: each map contributes weight exp(-λ * days_before_reference),
    with λ = 0.015/day (half-life ~46d) per CLAUDE.md.
  - Bayesian shrinkage: Beta(α, β) prior tied to effective sample size.
    Tiny samples regress to 50%; big samples dominate the posterior.
"""
from __future__ import annotations

import pandas as pd
from toolkit.cs2_mrt import simulate_series_probs
from toolkit.recency import weight_for_date, DEFAULT_LAMBDA


def team_map_winrate(
    matches_df: pd.DataFrame,
    team_id: int,
    days_back: int = 30,
    lam: float = DEFAULT_LAMBDA,
    reference_date: pd.Timestamp | None = None,
) -> tuple[float, int, float]:
    """
    Returns (recency_weighted_map_win_rate, n_maps_played_raw, effective_n).
    A series with team1_score=a, team2_score=b means a+b maps were played;
    team1 won `a` of them. Maps get weight exp(-lam * days_before_reference).
    """
    df = matches_df[
        (matches_df["team1_id"] == team_id) | (matches_df["team2_id"] == team_id)
    ].copy()
    if len(df) == 0:
        return 0.5, 0, 0.0
    df = df.sort_values("end_date")
    ref = reference_date if reference_date is not None else df["end_date"].max()
    cutoff = ref - pd.Timedelta(days=days_back)
    df = df[(df["end_date"] >= cutoff) & (df["end_date"] <= ref)]

    weighted_won = 0.0
    weighted_played = 0.0
    raw_played = 0
    for _, row in df.iterrows():
        s1 = int(row["team1_score"] or 0)
        s2 = int(row["team2_score"] or 0)
        played = s1 + s2
        if played == 0:
            continue
        is_t1 = (row["team1_id"] == team_id)
        won = s1 if is_t1 else s2
        w = weight_for_date(row["end_date"], ref, lam)
        weighted_won += w * won
        weighted_played += w * played
        raw_played += played
    if weighted_played <= 0:
        return 0.5, raw_played, 0.0
    return weighted_won / weighted_played, raw_played, weighted_played


def bayesian_shrink(wins: float, played: float, prior_alpha: float = 5.0,
                    prior_beta: float = 5.0) -> float:
    """Posterior mean of Beta(α, β) after `wins` from `played`. Default α=β=5
    is equivalent to a virtual "prior team" of 5W/5L = 50% over 10 maps."""
    return (prior_alpha + wins) / (prior_alpha + prior_beta + played)


def predict_series_flat(
    matches_df: pd.DataFrame,
    team_a_id: int,
    team_b_id: int,
    bo: int = 3,
    days_back: int = 30,
    prior_alpha: float = 5.0,
    prior_beta: float = 5.0,
    reference_date: pd.Timestamp | None = None,
) -> dict:
    """
    Flat prediction with recency-weighted observations + Beta(α,β) shrinkage.
    """
    wr_a, n_a, w_a = team_map_winrate(matches_df, team_a_id, days_back,
                                       reference_date=reference_date)
    wr_b, n_b, w_b = team_map_winrate(matches_df, team_b_id, days_back,
                                       reference_date=reference_date)

    wins_a_eff = wr_a * w_a
    wins_b_eff = wr_b * w_b
    wr_a_s = bayesian_shrink(wins_a_eff, w_a, prior_alpha, prior_beta)
    wr_b_s = bayesian_shrink(wins_b_eff, w_b, prior_alpha, prior_beta)

    odds_a = wr_a_s / max(1 - wr_a_s, 0.01)
    odds_b = wr_b_s / max(1 - wr_b_s, 0.01)
    p_map_a = odds_a / (odds_a + odds_b)
    p_map_a = max(0.05, min(0.95, p_map_a))

    p_a, p_b, clean, close = simulate_series_probs(p_map_a, bo)

    return {
        "bo": bo,
        "team_a_winrate_raw": wr_a,
        "team_b_winrate_raw": wr_b,
        "team_a_winrate_shrunk": wr_a_s,
        "team_b_winrate_shrunk": wr_b_s,
        "team_a_n_maps": n_a,
        "team_b_n_maps": n_b,
        "team_a_eff_n": w_a,
        "team_b_eff_n": w_b,
        "p_map_a": p_map_a,
        "p_map_b": 1 - p_map_a,
        "p_a_series": p_a,
        "p_b_series": p_b,
        "p_a_clean_sweep": clean,
        "p_a_close": close,
    }


if __name__ == "__main__":
    df = pd.read_parquet("data/clean/cs2_matches.parquet")
    tids = pd.concat([df["team1_id"], df["team2_id"]]).value_counts().head(10).index.tolist()
    print(f"Top 10 most-active team_ids: {tids}")
    if len(tids) >= 2:
        t_a, t_b = int(tids[0]), int(tids[1])
        result = predict_series_flat(df, t_a, t_b, bo=3, days_back=60)
        print(f"\nFlat prediction: team#{t_a} vs team#{t_b}, Bo3")
        for k, v in result.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
