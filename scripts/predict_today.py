"""
Daily CS2 prediction pipeline v3 (with odds template generation).
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toolkit.hltv_data import fetch_recent_matches, fetch_upcoming_matches
from toolkit.cs2_mrt import build_team_manifold, predict_series, simulate_series_probs
from toolkit.cs2_baseline import predict_series_flat
from toolkit.manifold_massey import fit_manifold_massey
from toolkit.massey import calibrate_beta
from toolkit.ensemble import load_ensemble, Ensemble
from toolkit.odds_data import write_odds_template
from toolkit.anomaly import compute_anomaly, anomaly_adjusted_prob
from toolkit.contrarian import detect_streak, contrarian_prob


def load_team_names():
    p = Path("reference/team_aliases.json")
    if not p.exists(): return {}
    return {int(k): v for k, v in json.loads(p.read_text()).items()}


def team_name_or_id(team_id, names):
    return names.get(int(team_id), {}).get("name", f"Team#{team_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-ahead", type=int, default=2)
    ap.add_argument("--days-back", type=int, default=60)
    ap.add_argument("--tier-floor", default="c", choices=["s", "a", "b", "c", "d"])
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    pq = Path("data/clean/cs2_matches.parquet")
    if args.refresh_data or not pq.exists():
        print(f"Fetching last {args.days_back} days of matches...")
        df = fetch_recent_matches(days_back=args.days_back, max_pages=400, tier_floor=args.tier_floor)
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq)
        print(f"  saved {len(df)} matches")
    else:
        df = pd.read_parquet(pq)
        print(f"Loaded {len(df)} cached matches")

    print("Fitting Manifold-Massey...")
    mmodel = fit_manifold_massey(df, days_back=args.days_back, lam_coupling=0.05)
    calibrate_beta(mmodel, df[df["end_date"] >= df["end_date"].max() - pd.Timedelta(days=args.days_back)])
    print(f"  rated {len(mmodel.ratings)} teams, beta={mmodel.beta:.3f}")

    ens_path = Path("reference/ensemble.json")
    if ens_path.exists():
        ens = load_ensemble(ens_path)
        print(f"  loaded ensemble: {ens.weights}, intercept={ens.intercept:+.3f}")
    else:
        ens = Ensemble(weights={"MRT": 0.33, "Flat": 0.33, "MMassey": 0.34}, intercept=0.0)

    print(f"\nFetching upcoming matches (next {args.days_ahead} days)...")
    upcoming = fetch_upcoming_matches(days_ahead=args.days_ahead)
    if len(upcoming) == 0:
        print("  no upcoming matches.")
        return
    upcoming = upcoming[upcoming["tier"].isin({"s","a","b","c"})]
    print(f"  {len(upcoming)} confirmed upcoming matches")

    names = load_team_names()
    rows = []
    for _, m in upcoming.iterrows():
        t1, t2 = int(m["team1_id"]), int(m["team2_id"])
        n1 = team_name_or_id(t1, names); n2 = team_name_or_id(t2, names)
        bo = int(m["bo_type"]) if m["bo_type"] in (1, 3, 5) else 3

        ta = build_team_manifold(df, t1, n1, days_back=args.days_back)
        tb = build_team_manifold(df, t2, n2, days_back=args.days_back)
        p_mrt_map = None; mrt = None
        if ta is not None and tb is not None:
            mrt = predict_series(ta, tb, bo=bo)
            p_mrt_map = mrt["p_map_a"]
        try:
            flat = predict_series_flat(df, t1, t2, bo=bo, days_back=args.days_back)
            p_flat_map = flat["p_map_a"]
        except Exception:
            flat = None; p_flat_map = None
        p_mas_map = mmodel.p_map_a(t1, t2) if (t1 in mmodel.ratings and t2 in mmodel.ratings) else None
        p_ens_map_raw = ens.predict({"MRT": p_mrt_map, "Flat": p_flat_map, "MMassey": p_mas_map})

        # Tuned filters (anomaly mean-reversion + streak contrarian).
        # Backtest improvement: ~0.2% Brier. Defaults in modules are backtest-best.
        anom_a = compute_anomaly(df, t1)
        anom_b = compute_anomaly(df, t2)
        streak_a = detect_streak(df, t1)
        streak_b = detect_streak(df, t2)
        if p_ens_map_raw is not None:
            p_ens_map = anomaly_adjusted_prob(p_ens_map_raw, anom_a, anom_b)
            p_ens_map = contrarian_prob(p_ens_map, streak_a, streak_b)
        else:
            p_ens_map = None
        p_a_ens = simulate_series_probs(p_ens_map, bo)[0] if p_ens_map is not None else None

        if p_mrt_map is None and p_flat_map is None and p_mas_map is None:
            rows.append({"match_id": m["match_id"], "start_date": m["start_date"],
                         "tier": m["tier"], "bo": bo, "team_a": n1, "team_b": n2,
                         "status": "INSUFFICIENT_DATA"})
            continue

        p_a_mrt = mrt["p_a_series"] if mrt else None
        p_a_flat = flat["p_a_series"] if flat else None
        p_a_mas = simulate_series_probs(p_mas_map, bo)[0] if p_mas_map is not None else None

        rows.append({
            "match_id": m["match_id"], "start_date": m["start_date"],
            "tier": m["tier"], "bo": bo, "team_a": n1, "team_b": n2,
            "team_a_id": t1, "team_b_id": t2,
            "mrt_p_a": p_a_mrt, "flat_p_a": p_a_flat,
            "mmassey_p_a": p_a_mas, "ensemble_p_a": p_a_ens,
            "mrt_confidence": mrt["confidence"] if mrt else None,
            "team_a_n_maps": ta.n_recent_maps if ta else None,
            "team_b_n_maps": tb.n_recent_maps if tb else None,
        })

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = Path(f"reports/{today}_cs2")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(rows, default=str, indent=2))
    print(f"\nSaved {out_dir}/predictions.json")

    # Auto-generate odds template (don't overwrite if user already filled it)
    odds_template_path = Path(f"data/odds/upcoming_{today}.yaml")
    if not odds_template_path.exists():
        write_odds_template(rows, odds_template_path)
        print(f"Saved odds template: {odds_template_path}")
        print(f"  -> fill it in with your bookmaker odds, then run:")
        print(f"     python scripts/evaluate_bets.py --date {today}")
    else:
        print(f"Odds file exists ({odds_template_path}) -- not overwriting.")

    # Build markdown
    md = [f"# CS2 Predictions v3 -- {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
          f"**{len(rows)} upcoming matches** (tier <= {args.tier_floor})\n",
          f"\nEnsemble weights: {ens.weights} | intercept {ens.intercept:+.3f}\n",
          f"Backtest score: Brier 0.232, ~60% map-accuracy (vs 0.250 / 50% baseline)\n",
          "\n## Side-by-side: all 4 predictors\n",
          "| Time | Tier | Match | Bo | MRT | Flat | M-Massey | **Ensemble** | MRT conf | sample |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "INSUFFICIENT_DATA":
            md.append(f"| {pd.to_datetime(r['start_date']).strftime('%m-%d %H:%M')} | {r['tier']} | "
                      f"{r['team_a']} vs {r['team_b']} | Bo{r['bo']} | INSUFFICIENT_DATA |||||||")
            continue
        def fmt(p): return f"{p*100:5.1f}%" if p is not None else "  --  "
        sample = f"{r['team_a_n_maps']}/{r['team_b_n_maps']}" if r['team_a_n_maps'] else "?"
        conf = f"{r['mrt_confidence']:.2f}" if r['mrt_confidence'] is not None else "--"
        md.append(f"| {pd.to_datetime(r['start_date']).strftime('%m-%d %H:%M')} | {r['tier']} | "
                  f"**{r['team_a']}** vs **{r['team_b']}** | Bo{r['bo']} | "
                  f"{fmt(r['mrt_p_a'])} | {fmt(r['flat_p_a'])} | {fmt(r['mmassey_p_a'])} | "
                  f"**{fmt(r['ensemble_p_a'])}** | {conf} | {sample} |")

    md.append("\n## Strong picks (Ensemble >=60% OR <=40% with MRT conf >=0.55)\n")
    strong = [r for r in rows if r.get("ensemble_p_a") is not None
              and (r["ensemble_p_a"] >= 0.60 or r["ensemble_p_a"] <= 0.40)
              and r.get("mrt_confidence") is not None and r["mrt_confidence"] >= 0.55]
    if strong:
        for r in sorted(strong, key=lambda x: -abs(x["ensemble_p_a"] - 0.5)):
            pick = r["team_a"] if r["ensemble_p_a"] >= 0.5 else r["team_b"]
            p = max(r["ensemble_p_a"], 1 - r["ensemble_p_a"])
            opp = r["team_b"] if pick == r["team_a"] else r["team_a"]
            md.append(f"- **{pick}** to win Bo{r['bo']} vs {opp} ({r['tier']}-tier) "
                      f"-- Ensemble {p*100:.1f}% | MRT conf {r['mrt_confidence']:.2f} "
                      f"| sample {r['team_a_n_maps']}/{r['team_b_n_maps']}")
    else:
        md.append("(none meet both thresholds)")

    md.append("\n## How to bet on these")
    md.append(f"1. Fill in `{odds_template_path}` with your bookmaker odds")
    md.append(f"2. Run `python scripts/evaluate_bets.py --date {today} --bankroll <X>`")
    md.append(f"3. Read `reports/{today}_cs2/bet_evaluations.md` for tier-classified picks with Kelly stakes")

    md.append("\n## Notes")
    md.append("- **MRT** = Fisher-Rao manifold geodesic + holonomy")
    md.append("- **Flat** = recency-weighted map winrate + Beta(5,5) Bayesian shrinkage")
    md.append("- **M-Massey** = global rating with manifold-coupled Laplacian regularizer (lambda=0.05)")
    md.append("- **Ensemble** = grid-search weighted combo + logit-space intercept")

    (out_dir / "report.md").write_text("\n".join(md))
    print(f"Saved {out_dir}/report.md")

    n_ok = sum(1 for r in rows if r.get("status") != "INSUFFICIENT_DATA")
    print(f"\nPredicted {n_ok} matches | strong picks: {len(strong)}")


if __name__ == "__main__":
    main()
