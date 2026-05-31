"""Bet evaluations page — read-only view of EV/Kelly recommendations."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from toolkit.odds_data import load_odds
from toolkit.domain import evaluate_bet


REPORTS_DIR = Path("reports")
ODDS_DIR = Path("data/odds")


# ---------- helpers ----------

def _list_dates() -> list[str]:
    if not REPORTS_DIR.exists():
        return []
    dates = []
    for p in REPORTS_DIR.glob("*_cs2"):
        if p.is_dir():
            stem = p.name.replace("_cs2", "")
            if len(stem) == 8 and stem.isdigit():
                dates.append(stem)
    return sorted(dates, reverse=True)


@st.cache_data(ttl=10)
def _load_predictions(date_str: str) -> list[dict]:
    path = REPORTS_DIR / f"{date_str}_cs2" / "predictions.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


@st.cache_data(ttl=10)
def _load_existing_odds(date_str: str) -> dict[int, dict]:
    path = ODDS_DIR / f"upcoming_{date_str}.yaml"
    return load_odds(path)


def _tier_rank(t: str) -> int:
    return {"A": 0, "B": 1, "C": 2}.get((t or "C").upper(), 3)


def _row_style(row: pd.Series) -> list[str]:
    t = (row.get("Tier") or "").upper()
    if t == "A":
        bg = "background-color: rgba(46, 160, 67, 0.25)"
    elif t == "B":
        bg = "background-color: rgba(255, 191, 0, 0.25)"
    else:
        bg = ""
    return [bg] * len(row)


# ---------- main render ----------

def render() -> None:
    st.title("Bet evaluations")

    dates = _list_dates()
    if not dates:
        st.warning("No prediction reports found in `reports/`.")
        return

    top = st.columns([1, 1, 3])
    with top[0]:
        date_str = st.selectbox("Report date", dates, index=0)
    with top[1]:
        bankroll = st.number_input(
            "Bankroll",
            min_value=0.0,
            value=1000.0,
            step=50.0,
            format="%.2f",
        )

    predictions = _load_predictions(date_str)
    if not predictions:
        st.error(f"`predictions.json` for {date_str} is missing or empty.")
        return

    predictions = [p for p in predictions if p.get("status") != "INSUFFICIENT_DATA"]
    odds_map = _load_existing_odds(date_str)
    if not odds_map:
        st.info(f"No odds entered for {date_str}. Go to the **Odds entry** page to add odds first.")
        return

    rows: list[dict] = []
    have_any_odds = False
    for p in predictions:
        mid = p["match_id"]
        if mid not in odds_map:
            continue
        odds_row = odds_map[mid].get("odds", {})
        oa = odds_row.get("match_winner_a")
        ob = odds_row.get("match_winner_b")
        if not oa or not ob:
            continue
        have_any_odds = True

        ep_a = p.get("ensemble_p_a")
        if ep_a is None:
            continue
        ep_b = 1.0 - ep_a

        mrt_conf = p.get("mrt_confidence")
        n_a = p.get("team_a_n_maps")
        n_b = p.get("team_b_n_maps")

        eval_a = evaluate_bet(
            market="match_winner", side="a",
            decimal_odds=float(oa), prob_estimate=float(ep_a),
            pair_odds=float(ob),
            mrt_confidence=mrt_conf, n_maps_a=n_a, n_maps_b=n_b,
        )
        eval_b = evaluate_bet(
            market="match_winner", side="b",
            decimal_odds=float(ob), prob_estimate=float(ep_b),
            pair_odds=float(oa),
            mrt_confidence=mrt_conf, n_maps_a=n_a, n_maps_b=n_b,
        )

        rows.append({
            "match_id": mid,
            "match_label": f"{p.get('team_a','?')} vs {p.get('team_b','?')}",
            "book": odds_map[mid].get("book") or "—",
            "team_a": p.get("team_a", ""),
            "team_b": p.get("team_b", ""),
            "ep_a": float(ep_a),
            "ep_b": float(ep_b),
            "odds_a": float(oa),
            "odds_b": float(ob),
            "eval_a": eval_a,
            "eval_b": eval_b,
        })

    if not have_any_odds:
        st.info(f"No odds filled in yet for {date_str}. Go to the **Odds entry** page to add odds first.")
        return

    # ---------- Recommended bets (A/B tier only) ----------
    rec_rows = []
    for r in rows:
        for side_key, ev in (("a", r["eval_a"]), ("b", r["eval_b"])):
            if ev.tier in ("A", "B"):
                pick_team = r["team_a"] if side_key == "a" else r["team_b"]
                rec_rows.append({
                    "Match": r["match_label"],
                    "Pick": pick_team,
                    "Book": r["book"],
                    "Odds": round(ev.decimal_odds, 2),
                    "Our P": round(ev.prob_estimate, 3),
                    "Fair P": round(ev.implied_prob_fair, 3),
                    "Edge": round(ev.edge, 3),
                    "EV": round(ev.ev, 3),
                    "Tier": ev.tier,
                    "Stake": round(ev.kelly_scaled * bankroll, 2),
                    "_kelly_full": ev.kelly_full,
                    "_tier_rank": _tier_rank(ev.tier),
                })

    # ---------- Summary metrics ----------
    n_a = sum(1 for r in rec_rows if r["Tier"] == "A")
    n_b = sum(1 for r in rec_rows if r["Tier"] == "B")
    total_stake = sum(r["Stake"] for r in rec_rows)
    # "Potential profit at full Kelly" = sum over rec bets of
    # kelly_full * bankroll * (odds - 1)  (i.e. if every bet wins, at full Kelly)
    pot_profit_full_k = sum(
        r["_kelly_full"] * bankroll * (r["Odds"] - 1) for r in rec_rows
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("A-tier bets", n_a)
    m2.metric("B-tier bets", n_b)
    m3.metric("Total stake", f"{total_stake:.2f}")
    m4.metric("Potential profit (full Kelly)", f"{pot_profit_full_k:.2f}")

    st.markdown("---")
    st.subheader("Recommended bets")
    if not rec_rows:
        st.info("No A or B tier bets at current odds.")
    else:
        rec_df = pd.DataFrame(rec_rows).sort_values(
            by=["_tier_rank", "EV"], ascending=[True, False]
        ).drop(columns=["_kelly_full", "_tier_rank"]).reset_index(drop=True)
        try:
            styled = rec_df.style.apply(_row_style, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # ---------- All evaluated matches ----------
    st.markdown("---")
    st.subheader("All evaluated matches")
    all_rows = []
    for r in rows:
        ea = r["eval_a"]
        eb = r["eval_b"]
        # Decide which side is "Pick" for table — the better tier / EV.
        pick_a_better = (_tier_rank(ea.tier), -ea.ev) <= (_tier_rank(eb.tier), -eb.ev)
        pick_eval = ea if pick_a_better else eb
        pick_team = r["team_a"] if pick_a_better else r["team_b"]
        all_rows.append({
            "Match": r["match_label"],
            "Pick": pick_team,
            "Book": r["book"],
            "Odds": round(pick_eval.decimal_odds, 2),
            "Our P": round(pick_eval.prob_estimate, 3),
            "Fair P": round(pick_eval.implied_prob_fair, 3),
            "Edge": round(pick_eval.edge, 3),
            "EV": round(pick_eval.ev, 3),
            "EV B": round(eb.ev, 3),
            "Tier": pick_eval.tier,
            "_tier_rank": _tier_rank(pick_eval.tier),
        })

    all_df = pd.DataFrame(all_rows).sort_values(
        by=["_tier_rank", "EV"], ascending=[True, False]
    ).drop(columns=["_tier_rank"]).reset_index(drop=True)

    try:
        styled_all = all_df.style.apply(_row_style, axis=1)
        st.dataframe(styled_all, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(all_df, use_container_width=True, hide_index=True)
