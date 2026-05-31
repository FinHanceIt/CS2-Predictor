"""
Walk-forward backtest for CS2-Predictor models.

For each of `n_origins` cutoff dates evenly spaced over the last `window` days:
  1. Fit each model using ONLY data ending at the cutoff.
  2. Predict each map in a forward `holdout_days` window.
  3. Record (predicted_prob, actual_outcome) per prediction per model.

Then compute:
  - Brier score (lower is better)
  - Log loss (lower is better)
  - Accuracy at 50% threshold (higher is better)
  - Calibration buckets (predicted prob vs actual frequency)

This is the only honest way to compare models. CLAUDE.md mandates 8 origins.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd

from toolkit.cs2_mrt import build_team_manifold, predict_series
from toolkit.cs2_baseline import predict_series_flat
from toolkit.massey import fit_massey, calibrate_beta


@dataclass
class BacktestResult:
    model_name: str
    n_predictions: int = 0
    brier_sum: float = 0.0
    logloss_sum: float = 0.0
    correct: int = 0          # accuracy at 50%
    skipped: int = 0          # couldn't predict (insufficient data)
    records: list = field(default_factory=list)  # (p_a, outcome_a, tier, match_id)

    @property
    def brier(self) -> float:
        return self.brier_sum / max(self.n_predictions, 1)

    @property
    def logloss(self) -> float:
        return self.logloss_sum / max(self.n_predictions, 1)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.n_predictions, 1)


def expand_to_maps(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Each series -> (s1+s2) map rows. team1_won is True for first s1 entries.
    For backtest purposes we only need (match_id, team1_id, team2_id,
    team1_won_count, total_played, end_date)."""
    rows = []
    for _, r in matches_df.iterrows():
        s1 = int(r["team1_score"] or 0); s2 = int(r["team2_score"] or 0)
        if s1 + s2 == 0: continue
        for _ in range(s1):
            rows.append({"match_id": r["match_id"], "team1_id": r["team1_id"],
                         "team2_id": r["team2_id"], "team1_won": True,
                         "end_date": r["end_date"], "tier": r["tier"],
                         "bo_type": r["bo_type"]})
        for _ in range(s2):
            rows.append({"match_id": r["match_id"], "team1_id": r["team1_id"],
                         "team2_id": r["team2_id"], "team1_won": False,
                         "end_date": r["end_date"], "tier": r["tier"],
                         "bo_type": r["bo_type"]})
    return pd.DataFrame(rows)


def record_prediction(result: BacktestResult, p_team1_wins_map: float,
                      outcome_team1_won: bool, tier: str, match_id: int) -> None:
    p = max(1e-6, min(1.0 - 1e-6, p_team1_wins_map))
    y = 1.0 if outcome_team1_won else 0.0
    result.brier_sum += (p - y) ** 2
    result.logloss_sum += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    if (p >= 0.5) == outcome_team1_won:
        result.correct += 1
    result.n_predictions += 1
    result.records.append((p, outcome_team1_won, tier, match_id))


def predict_map_mrt(matches_df, t1, t2, ref_date, days_back=60):
    ta = build_team_manifold(matches_df, t1, days_back=days_back, reference_date=ref_date)
    tb = build_team_manifold(matches_df, t2, days_back=days_back, reference_date=ref_date)
    if ta is None or tb is None:
        return None
    return predict_series(ta, tb, bo=3)["p_map_a"]


def predict_map_flat(matches_df, t1, t2, ref_date, days_back=60):
    flat = predict_series_flat(matches_df, t1, t2, bo=3, days_back=days_back,
                               reference_date=ref_date)
    return flat["p_map_a"]


def predict_map_massey(model, t1, t2):
    if t1 not in model.ratings or t2 not in model.ratings:
        return None
    return model.p_map_a(t1, t2)


def run_backtest(matches_df: pd.DataFrame, n_origins: int = 8,
                 holdout_days: int = 5, train_days_back: int = 60,
                 verbose: bool = False) -> dict[str, BacktestResult]:
    """
    Walk-forward backtest. Origins evenly spaced from
      latest_date - (n_origins - 1) * holdout_days   ...   latest_date.
    Each origin trains on data ending at the origin, predicts maps in the
    `holdout_days` window strictly after the origin.
    """
    matches_df = matches_df.sort_values("end_date").reset_index(drop=True)
    latest = matches_df["end_date"].max()
    earliest = matches_df["end_date"].min()

    # Build origins so each gets a holdout window
    origins = []
    for k in range(n_origins):
        origin = latest - pd.Timedelta(days=(n_origins - 1 - k) * holdout_days + holdout_days)
        if origin < earliest + pd.Timedelta(days=train_days_back // 3):
            continue
        origins.append(origin)

    if verbose:
        print(f"origins: {len(origins)}")
        for o in origins:
            print(f"  {o.date()}")

    results = {
        "MRT": BacktestResult(model_name="MRT"),
        "Flat": BacktestResult(model_name="Flat"),
        "Massey": BacktestResult(model_name="Massey"),
    }

    for origin in origins:
        train = matches_df[matches_df["end_date"] <= origin]
        forward_end = origin + pd.Timedelta(days=holdout_days)
        holdout = matches_df[
            (matches_df["end_date"] > origin) & (matches_df["end_date"] <= forward_end)
        ]
        if len(holdout) == 0:
            continue

        # Fit Massey once per origin
        m_model = fit_massey(train, reference_date=origin, days_back=train_days_back)
        calibrate_beta(m_model, train[train["end_date"] >= origin - pd.Timedelta(days=train_days_back)])

        # Expand holdout series into map outcomes
        holdout_maps = expand_to_maps(holdout)

        # For each match in holdout, predict ONCE and record once per map
        for mid, group in holdout_maps.groupby("match_id"):
            t1 = int(group.iloc[0]["team1_id"]); t2 = int(group.iloc[0]["team2_id"])
            tier = group.iloc[0]["tier"]

            p_mrt = predict_map_mrt(train, t1, t2, origin, days_back=train_days_back)
            p_flat = predict_map_flat(train, t1, t2, origin, days_back=train_days_back)
            p_mas = predict_map_massey(m_model, t1, t2)

            for _, m in group.iterrows():
                won = bool(m["team1_won"])
                if p_mrt is not None:
                    record_prediction(results["MRT"], p_mrt, won, tier, mid)
                else:
                    results["MRT"].skipped += 1
                if p_flat is not None:
                    record_prediction(results["Flat"], p_flat, won, tier, mid)
                else:
                    results["Flat"].skipped += 1
                if p_mas is not None:
                    record_prediction(results["Massey"], p_mas, won, tier, mid)
                else:
                    results["Massey"].skipped += 1

        if verbose:
            print(f"  origin {origin.date()}: {len(holdout)} matches, "
                  f"{len(holdout_maps)} maps")

    return results


def report_results(results: dict[str, BacktestResult]) -> None:
    print(f"\n{'Model':<10} {'N':>6}  {'Brier':>7}  {'LogLoss':>8}  {'Acc':>6}  {'Skipped':>8}")
    print("-" * 55)
    # Baselines
    baseline_brier = 0.25  # always-predict-0.5
    print(f"{'(0.5 ref)':<10} {'':>6}  {baseline_brier:>7.4f}  {math.log(2):>8.4f}  {'50.0%':>6}  {'':>8}")
    for name, r in results.items():
        print(f"{name:<10} {r.n_predictions:>6}  {r.brier:>7.4f}  {r.logloss:>8.4f}  "
              f"{r.accuracy*100:>5.1f}%  {r.skipped:>8}")

    # Calibration bins
    print("\nCalibration (predicted prob vs actual freq):")
    for name, r in results.items():
        if r.n_predictions == 0:
            continue
        bins = [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)]
        print(f"  {name}:")
        for lo, hi in bins:
            in_bin = [(p, y) for p, y, *_ in r.records if lo <= p < hi]
            if not in_bin:
                print(f"    [{lo:.1f}–{hi:.1f})  empty")
                continue
            avg_p = sum(p for p, _ in in_bin) / len(in_bin)
            freq  = sum(1 for _, y in in_bin if y) / len(in_bin)
            print(f"    [{lo:.1f}–{hi:.1f})  n={len(in_bin):>4}  avg_p={avg_p:.3f}  freq={freq:.3f}")
