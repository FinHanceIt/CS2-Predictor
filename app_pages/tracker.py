"""Tracker page: historical validation of CS2 predictions vs real outcomes."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
EPS = 1e-6
MODEL_KEYS = [
    ("mrt_p_a", "MRT"),
    ("flat_p_a", "Flat"),
    ("mmassey_p_a", "M-Massey"),
    ("pred_p_a", "Ensemble"),
]


# ---------- Loading ----------

@st.cache_data(ttl=30)
def _load_all_snapshots() -> pd.DataFrame:
    """Load every results_tracker_*.json, dedupe by (date, match_id) keeping the
    latest snapshot per match (by file mtime). Returns a clean DataFrame."""
    if not REPORTS_DIR.exists():
        return pd.DataFrame()

    files = sorted(REPORTS_DIR.glob("results_tracker_*.json"))
    if not files:
        return pd.DataFrame()

    rows: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        mtime = f.stat().st_mtime
        for r in data:
            if not isinstance(r, dict):
                continue
            r = dict(r)
            r["_snapshot_mtime"] = mtime
            r["_snapshot_file"] = f.name
            # Build a match key: use match_id if present, else fall back to date+teams.
            if r.get("match_id") is not None:
                r["_match_key"] = (str(r.get("date", "")), str(r["match_id"]))
            else:
                r["_match_key"] = (
                    str(r.get("date", "")),
                    f"{r.get('team_a','')}|{r.get('team_b','')}",
                )
            rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Keep latest snapshot per (date, match_id)
    df = df.sort_values("_snapshot_mtime").drop_duplicates(
        subset=["_match_key"], keep="last"
    )
    df = df.drop(columns=["_match_key"])

    # Type coercion
    for col in ("pred_p_a", "mrt_p_a", "flat_p_a", "mmassey_p_a", "mrt_conf"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "actual_a_won" in df.columns:
        df["actual_a_won"] = df["actual_a_won"].astype(bool)
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    # Order by date asc by default
    if "date_parsed" in df.columns:
        df = df.sort_values("date_parsed", kind="stable")

    return df.reset_index(drop=True)


# ---------- Metrics ----------

def _metrics_for(df: pd.DataFrame, prob_col: str) -> dict | None:
    """Compute n, accuracy, brier, logloss for one model column."""
    if prob_col not in df.columns or "actual_a_won" not in df.columns:
        return None
    sub = df[[prob_col, "actual_a_won"]].dropna(subset=[prob_col])
    n = len(sub)
    if n == 0:
        return None
    p = sub[prob_col].to_numpy(dtype=float)
    y = sub["actual_a_won"].to_numpy(dtype=float)
    correct = (((p > 0.5).astype(float) == y).sum())
    brier = float(np.mean((p - y) ** 2))
    p_clipped = np.clip(p, EPS, 1.0 - EPS)
    ll = float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))
    return {"n": int(n), "acc": float(correct) / n, "brier": brier, "logloss": ll}


def _strong_picks_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty or "pred_p_a" not in df.columns or "mrt_conf" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    p = df["pred_p_a"]
    c = df["mrt_conf"]
    return ((p >= 0.60) | (p <= 0.40)) & (c >= 0.55) & p.notna() & c.notna()


# ---------- Charts ----------

def _calibration_chart(df: pd.DataFrame) -> go.Figure:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]"]
    sub = df[["pred_p_a", "actual_a_won"]].dropna(subset=["pred_p_a"])
    pts_x, pts_y, pts_n, pts_label = [], [], [], []
    if not sub.empty:
        sub = sub.copy()
        sub["_bin"] = pd.cut(sub["pred_p_a"], bins=bins, labels=labels,
                              right=False, include_lowest=True)
        for label in labels:
            grp = sub[sub["_bin"] == label]
            if len(grp) == 0:
                continue
            mean_pred = float(grp["pred_p_a"].mean())
            actual = float(grp["actual_a_won"].astype(float).mean())
            pts_x.append(mean_pred)
            pts_y.append(actual)
            pts_n.append(int(len(grp)))
            pts_label.append(label)

    fig = go.Figure()
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(150,150,150,0.6)", dash="dash", width=1),
        name="Perfect calibration", hoverinfo="skip", showlegend=True,
    ))
    if pts_x:
        max_n = max(pts_n) if pts_n else 1
        # Larger marker size for more samples; color also encodes count
        sizes = [12 + 30 * (n / max_n) for n in pts_n]
        fig.add_trace(go.Scatter(
            x=pts_x, y=pts_y, mode="markers+text",
            marker=dict(
                size=sizes, color=pts_n, colorscale="Blues",
                cmin=0, cmax=max(max_n, 1),
                showscale=True,
                colorbar=dict(title="Count"),
                line=dict(color="rgba(0,0,0,0.4)", width=1),
            ),
            text=[f"n={n}" for n in pts_n],
            textposition="top center",
            customdata=list(zip(pts_label, pts_n)),
            hovertemplate=(
                "Bin: %{customdata[0]}<br>"
                "Mean pred: %{x:.3f}<br>"
                "Actual rate: %{y:.3f}<br>"
                "N: %{customdata[1]}<extra></extra>"
            ),
            name="Ensemble bins",
        ))
    fig.update_layout(
        height=420,
        xaxis=dict(title="Predicted probability (team A wins)",
                   range=[-0.02, 1.02]),
        yaxis=dict(title="Actual win rate", range=[-0.02, 1.02]),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
    )
    return fig


def _brier_over_time_chart(df: pd.DataFrame, window: int = 10) -> go.Figure:
    sub = df[["date_parsed", "pred_p_a", "actual_a_won"]].dropna(
        subset=["pred_p_a"]
    ).copy()
    if "date_parsed" in sub.columns:
        sub = sub.sort_values("date_parsed", kind="stable").reset_index(drop=True)
    sub["match_idx"] = np.arange(1, len(sub) + 1)
    y = sub["actual_a_won"].astype(float)
    p = sub["pred_p_a"].astype(float)
    sub["sq_err"] = (p - y) ** 2
    sub["rolling_brier"] = (
        sub["sq_err"].rolling(window=window, min_periods=max(2, window // 2)).mean()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["match_idx"], y=sub["rolling_brier"],
        mode="lines+markers",
        line=dict(color="#2c7be5", width=2),
        marker=dict(size=5),
        name=f"Rolling Brier (window={window})",
        hovertemplate="Match #%{x}<br>Brier: %{y:.4f}<extra></extra>",
    ))
    # Reference: 0.25 is a coin-flip Brier
    fig.add_hline(
        y=0.25, line_dash="dash", line_color="rgba(200,80,80,0.6)",
        annotation_text="Coin flip (0.25)", annotation_position="top right",
    )
    fig.update_layout(
        height=360,
        xaxis_title="Match index (chronological)",
        yaxis_title="Brier score (lower = better)",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    return fig


# ---------- Helpers ----------

def _highlight_best(df: pd.DataFrame):
    """Highlight the best value per column. Acc: highest is best.
    Brier/LogLoss: lowest is best. N is informational, not highlighted."""
    def _style(col: pd.Series):
        name = col.name
        out = [""] * len(col)
        try:
            numeric = pd.to_numeric(col, errors="coerce")
        except Exception:
            return out
        if numeric.isna().all():
            return out
        if name in ("Brier", "LogLoss"):
            idx = numeric.idxmin()
        elif name == "Accuracy":
            idx = numeric.idxmax()
        else:
            return out
        for i, lbl in enumerate(col.index):
            if lbl == idx:
                out[i] = "background-color: rgba(46, 204, 113, 0.22); font-weight: 600;"
        return out

    return df.style.apply(_style, axis=0)


def _recent_results_table(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    keep = df.dropna(subset=["pred_p_a"]).copy()
    if "date_parsed" in keep.columns:
        keep = keep.sort_values("date_parsed", ascending=False, kind="stable")
    keep = keep.head(n).copy()
    keep["Match"] = keep["team_a"].astype(str) + " vs " + keep["team_b"].astype(str)
    keep["Predicted"] = keep.apply(
        lambda r: f"{r['team_a']} ({r['pred_p_a']*100:.1f}%)"
        if r["pred_p_a"] >= 0.5
        else f"{r['team_b']} ({(1-r['pred_p_a'])*100:.1f}%)",
        axis=1,
    )
    keep["Actual"] = keep.apply(
        lambda r: (f"{r['team_a']} " if r["actual_a_won"] else f"{r['team_b']} ")
                  + f"({r.get('score','')})",
        axis=1,
    )
    keep["Correct?"] = ((keep["pred_p_a"] >= 0.5) == keep["actual_a_won"]).map(
        {True: "WIN", False: "LOSS"}
    )
    keep["Date"] = keep["date"]
    keep["Tier"] = keep["tier"].astype(str).str.upper()
    return keep[["Date", "Tier", "Match", "Predicted", "Actual", "Correct?"]].reset_index(drop=True)


def _color_correct_col(val: str):
    if val == "WIN":
        return "background-color: rgba(46, 204, 113, 0.28); color: #0a4a1f; font-weight: 600;"
    if val == "LOSS":
        return "background-color: rgba(231, 76, 60, 0.28); color: #5a0e0e; font-weight: 600;"
    return ""


def _run_refresh() -> tuple[int, str, str]:
    """Run the track_results script via subprocess. Returns (returncode, stdout, stderr)."""
    cwd = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "scripts/track_results.py"]
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=120
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), f"Timed out after 120s\n{e.stderr or ''}"
    except Exception as e:
        return 1, "", f"Error launching subprocess: {e}"


# ---------- Render ----------

def render():
    st.title("Results tracker")
    st.caption(
        "Historical validation of CS2 predictions vs actual outcomes. "
        "Lower Brier / LogLoss is better; higher accuracy is better."
    )

    cols = st.columns([1, 4])
    with cols[0]:
        refresh_clicked = st.button("Refresh results from bo3.gg", type="primary")
    with cols[1]:
        st.caption(
            "Runs `python scripts/track_results.py` to pull finished matches from bo3.gg "
            "and write a new `results_tracker_*.json` snapshot."
        )

    if refresh_clicked:
        with st.spinner("Querying bo3.gg for finished matches..."):
            rc, out, err = _run_refresh()
        if rc == 0:
            st.success("Refresh complete.")
        else:
            st.error(f"Refresh failed (exit code {rc}).")
        with st.expander("Subprocess output", expanded=(rc != 0)):
            if out:
                st.code(out, language="text")
            if err:
                st.markdown("**stderr:**")
                st.code(err, language="text")
        # Invalidate cache so the new snapshot loads immediately
        _load_all_snapshots.clear()

    df = _load_all_snapshots()

    if df.empty:
        st.info(
            "No results yet. Run the prediction pipeline first, wait for matches to "
            "finish, then click **Refresh results from bo3.gg**."
        )
        return

    # ---------- Top metrics ----------
    total = len(df)
    ens = _metrics_for(df, "pred_p_a")
    strong_mask = _strong_picks_mask(df)
    strong_df = df[strong_mask]
    strong_metrics = _metrics_for(strong_df, "pred_p_a") if not strong_df.empty else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matches tracked", f"{total}")
    m2.metric(
        "Ensemble accuracy",
        f"{ens['acc']*100:.1f}%" if ens else "—",
        help=f"Predictions where p>0.5 matched the actual winner (n={ens['n']})." if ens else None,
    )
    m3.metric(
        "Ensemble Brier",
        f"{ens['brier']:.4f}" if ens else "—",
        help="Mean squared error of probability vs outcome. Lower is better; 0.25 = coin flip.",
    )
    if strong_metrics:
        m4.metric(
            "Strong-pick accuracy",
            f"{strong_metrics['acc']*100:.1f}%",
            help=f"p_a >=0.60 or <=0.40 AND mrt_conf >=0.55 (n={strong_metrics['n']}).",
        )
    else:
        m4.metric("Strong-pick accuracy", "—",
                  help="No matches meet the strong-pick thresholds yet.")

    st.markdown("---")

    # ---------- Per-model comparison ----------
    st.subheader("Per-model comparison")
    rows = []
    for key, label in MODEL_KEYS:
        m = _metrics_for(df, key)
        if m is None:
            rows.append({"Model": label, "N": 0, "Accuracy": np.nan,
                         "Brier": np.nan, "LogLoss": np.nan})
        else:
            rows.append({
                "Model": label,
                "N": m["n"],
                "Accuracy": m["acc"],
                "Brier": m["brier"],
                "LogLoss": m["logloss"],
            })
    cmp_df = pd.DataFrame(rows).set_index("Model")
    styled = _highlight_best(cmp_df).format({
        "N": "{:d}",
        "Accuracy": "{:.1%}",
        "Brier": "{:.4f}",
        "LogLoss": "{:.4f}",
    }, na_rep="—")
    st.dataframe(styled, use_container_width=True)
    st.caption("Best value in each column is highlighted (acc: highest; brier/logloss: lowest).")

    st.markdown("---")

    # ---------- Calibration chart ----------
    st.subheader("Calibration (Ensemble)")
    st.caption(
        "If the model is well-calibrated, points lie on the diagonal: e.g. "
        "predictions in the [0.6, 0.8) bin should win ~70% of the time. "
        "Marker size and color encode bin sample size."
    )
    cal_fig = _calibration_chart(df)
    st.plotly_chart(cal_fig, use_container_width=True)

    # ---------- Brier-over-time ----------
    st.subheader("Brier score over time (10-match rolling)")
    n_with_pred = int(df["pred_p_a"].notna().sum()) if "pred_p_a" in df.columns else 0
    if n_with_pred < 2:
        st.info("Need at least 2 finished predictions to plot a rolling series.")
    else:
        bot_fig = _brier_over_time_chart(df, window=10)
        st.plotly_chart(bot_fig, use_container_width=True)

    st.markdown("---")

    # ---------- Per-tier breakdown ----------
    st.subheader("Per-tier breakdown")
    if "tier" not in df.columns:
        st.caption("No `tier` column found.")
    else:
        tier_rows = []
        # Standard tier order
        for tier in ["s", "a", "b", "c"]:
            sub = df[df["tier"].astype(str).str.lower() == tier]
            m = _metrics_for(sub, "pred_p_a")
            if m is None:
                continue
            tier_rows.append({
                "Tier": tier.upper(),
                "N": m["n"],
                "Accuracy": m["acc"],
                "Brier": m["brier"],
            })
        # Catch any tier we missed (e.g. unusual labels)
        known = {"s", "a", "b", "c"}
        other = df[~df["tier"].astype(str).str.lower().isin(known)]
        if not other.empty:
            m = _metrics_for(other, "pred_p_a")
            if m is not None:
                tier_rows.append({
                    "Tier": "Other",
                    "N": m["n"],
                    "Accuracy": m["acc"],
                    "Brier": m["brier"],
                })
        if not tier_rows:
            st.info("No tier breakdown available yet.")
        else:
            tier_df = pd.DataFrame(tier_rows).set_index("Tier")
            st.dataframe(
                tier_df.style.format({
                    "N": "{:d}",
                    "Accuracy": "{:.1%}",
                    "Brier": "{:.4f}",
                }, na_rep="—"),
                use_container_width=True,
            )

    st.markdown("---")

    # ---------- Recent results ----------
    st.subheader("Recent results (last 20)")
    recent = _recent_results_table(df, n=20)
    if recent.empty:
        st.info("No completed predictions to show.")
    else:
        styled_recent = recent.style.applymap(_color_correct_col, subset=["Correct?"])
        st.dataframe(styled_recent, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---------- Strong picks history ----------
    st.subheader("Strong picks history")
    st.caption(
        "Subset where the ensemble was confident (p>=0.60 or p<=0.40) AND MRT confidence >=0.55. "
        "These are the matches the user would actually have bet."
    )
    if strong_df.empty:
        st.info("No strong picks have been resolved yet.")
    else:
        # Show strong-pick summary metrics too
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Strong picks N", f"{strong_metrics['n']}")
        sm2.metric("Accuracy", f"{strong_metrics['acc']*100:.1f}%")
        sm3.metric("Brier", f"{strong_metrics['brier']:.4f}")

        strong_recent = _recent_results_table(strong_df, n=20)
        styled_strong = strong_recent.style.applymap(_color_correct_col, subset=["Correct?"])
        st.dataframe(styled_strong, use_container_width=True, hide_index=True)
# end of tracker page
