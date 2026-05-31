"""Dashboard page: today's CS2 predictions with strong picks and full table."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def _list_report_dates() -> list[str]:
    """Return sorted (desc) list of YYYYMMDD strings for which a report dir exists."""
    if not REPORTS_DIR.exists():
        return []
    dates = []
    for p in REPORTS_DIR.glob("*_cs2"):
        if p.is_dir():
            stem = p.name.replace("_cs2", "")
            if len(stem) == 8 and stem.isdigit():
                dates.append(stem)
    return sorted(dates, reverse=True)


@st.cache_data(ttl=30)
def _load_predictions(date_str: str) -> list[dict]:
    pred_path = REPORTS_DIR / f"{date_str}_cs2" / "predictions.json"
    if not pred_path.exists():
        return []
    try:
        return json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _pred_mtime(date_str: str) -> datetime | None:
    p = REPORTS_DIR / f"{date_str}_cs2" / "predictions.json"
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime)


def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize start_date
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    return df


def _strong_picks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ensemble_p_a" not in df.columns:
        return pd.DataFrame()
    sub = df[df["ensemble_p_a"].notna() & df.get("mrt_confidence", pd.Series(dtype=float)).notna()].copy()
    if sub.empty:
        return sub
    mask = ((sub["ensemble_p_a"] >= 0.60) | (sub["ensemble_p_a"] <= 0.40)) & (sub["mrt_confidence"] >= 0.55)
    sub = sub[mask].copy()
    if sub.empty:
        return sub
    sub["pick"] = sub.apply(lambda r: r["team_a"] if r["ensemble_p_a"] >= 0.5 else r["team_b"], axis=1)
    sub["opponent"] = sub.apply(lambda r: r["team_b"] if r["ensemble_p_a"] >= 0.5 else r["team_a"], axis=1)
    sub["pick_prob"] = sub["ensemble_p_a"].apply(lambda p: max(p, 1 - p))
    sub["sample"] = sub.apply(
        lambda r: f"{int(r['team_a_n_maps']) if pd.notna(r.get('team_a_n_maps')) else '?'}"
                  f"/{int(r['team_b_n_maps']) if pd.notna(r.get('team_b_n_maps')) else '?'}",
        axis=1,
    )
    sub = sub.sort_values("pick_prob", ascending=False)
    return sub[["start_date", "tier", "bo", "pick", "opponent", "pick_prob", "mrt_confidence", "sample"]]


def render():
    st.title("Dashboard")

    dates = _list_report_dates()
    if not dates:
        st.info(
            "No predictions found yet. Go to the **Pipeline** page and click "
            "**Refresh data** or **Predict (no refresh)** to generate today's predictions."
        )
        return

    # Date selector
    col_a, col_b = st.columns([1, 3])
    with col_a:
        selected = st.selectbox("Report date", dates, index=0)
    mtime = _pred_mtime(selected)
    with col_b:
        if mtime:
            st.caption(f"Last prediction file update: **{mtime:%Y-%m-%d %H:%M:%S}**")
        else:
            st.caption("No predictions.json for this date.")

    rows = _load_predictions(selected)
    if not rows:
        st.warning(f"No predictions.json for {selected}. Try the Pipeline page.")
        return

    df = _to_dataframe(rows)

    # Summary bar
    total = len(df)
    insufficient = int((df.get("status") == "INSUFFICIENT_DATA").sum()) if "status" in df.columns else 0
    strong_df = _strong_picks(df)
    n_strong = len(strong_df)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total matches", total)
    m2.metric("Strong picks", n_strong)
    m3.metric("Insufficient data", insufficient)

    st.markdown("---")

    # Strong picks
    st.subheader("Strong picks")
    st.caption("Ensemble >= 60% or <= 40% AND MRT confidence >= 0.55")
    if strong_df.empty:
        st.info("No matches meet both thresholds today.")
    else:
        st.dataframe(
            strong_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "start_date": st.column_config.DatetimeColumn("Start (UTC)", format="MM-DD HH:mm"),
                "tier": st.column_config.TextColumn("Tier", width="small"),
                "bo": st.column_config.NumberColumn("Bo", width="small"),
                "pick": st.column_config.TextColumn("Pick"),
                "opponent": st.column_config.TextColumn("Opponent"),
                "pick_prob": st.column_config.ProgressColumn(
                    "Pick prob", format="%.2f", min_value=0.5, max_value=1.0,
                    help="Probability that the picked team wins the series (0.5-1.0).",
                ),
                "mrt_confidence": st.column_config.NumberColumn("MRT conf", format="%.2f"),
                "sample": st.column_config.TextColumn("Sample (a/b)"),
            },
        )

    st.markdown("---")

    # Full predictions table
    st.subheader("All predictions")
    display = df.copy()
    # Build a clean display frame
    cols_wanted = [
        "start_date", "tier", "bo", "team_a", "team_b",
        "mrt_p_a", "flat_p_a", "mmassey_p_a", "ensemble_p_a",
        "mrt_confidence", "team_a_n_maps", "team_b_n_maps", "status",
    ]
    for c in cols_wanted:
        if c not in display.columns:
            display[c] = None
    display = display[cols_wanted]

    # Sort: insufficient at bottom, then by start_date
    display["_is_insufficient"] = (display["status"] == "INSUFFICIENT_DATA").astype(int)
    display = display.sort_values(["_is_insufficient", "start_date"], ascending=[True, True])
    display = display.drop(columns=["_is_insufficient"])

    # Scale probabilities to percentages for display (NumberColumn format is applied raw)
    for c in ("mrt_p_a", "flat_p_a", "mmassey_p_a", "ensemble_p_a"):
        display[c] = pd.to_numeric(display[c], errors="coerce") * 100

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "start_date": st.column_config.DatetimeColumn("Start (UTC)", format="MM-DD HH:mm"),
            "tier": st.column_config.TextColumn("Tier", width="small"),
            "bo": st.column_config.NumberColumn("Bo", width="small"),
            "team_a": st.column_config.TextColumn("Team A"),
            "team_b": st.column_config.TextColumn("Team B"),
            "mrt_p_a": st.column_config.NumberColumn("MRT %", format="%.1f",
                                                      help="MRT series prob for team A (percent)"),
            "flat_p_a": st.column_config.NumberColumn("Flat %", format="%.1f"),
            "mmassey_p_a": st.column_config.NumberColumn("M-Massey %", format="%.1f"),
            "ensemble_p_a": st.column_config.NumberColumn("Ensemble %", format="%.1f"),
            "mrt_confidence": st.column_config.NumberColumn("MRT conf", format="%.2f"),
            "team_a_n_maps": st.column_config.NumberColumn("A maps", format="%d"),
            "team_b_n_maps": st.column_config.NumberColumn("B maps", format="%d"),
            "status": st.column_config.TextColumn("Status"),
        },
    )

    # Optional: show raw report.md
    report_md = REPORTS_DIR / f"{selected}_cs2" / "report.md"
    if report_md.exists():
        with st.expander("Markdown report"):
            st.markdown(report_md.read_text(encoding="utf-8"))
