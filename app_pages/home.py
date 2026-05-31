"""Home: one-screen control room — slate, discipline labels, results, Claude briefing."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS_DIR = ROOT / "reports"

from toolkit import transurfing, claude_brief
from app_pages import pipeline, tracker

VERDICT_LABEL = {"ride": "RIDE", "caution": "CARE", "neutral": "FLAT", "avoid": "AVOID"}
VERDICT_ORDER = {"ride": 0, "caution": 1, "neutral": 2, "avoid": 3}
VERDICT_BG = {
    "RIDE": "background-color: rgba(29,158,117,0.22); font-weight:600;",
    "CARE": "background-color: rgba(186,117,23,0.20); font-weight:600;",
    "FLAT": "background-color: rgba(136,135,128,0.16);",
    "AVOID": "background-color: rgba(226,75,74,0.20); font-weight:600;",
}


def _latest_date() -> str | None:
    return pipeline._latest_report_date()


@st.cache_data(ttl=30)
def _load_predictions(date_str: str) -> list[dict]:
    p = REPORTS_DIR / f"{date_str}_cs2" / "predictions.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _slate_df(rows: list[dict]) -> tuple[pd.DataFrame, list]:
    reads = transurfing.read_predictions(rows)
    meta = {(r["team_a"], r["team_b"]): r for r in rows}
    out = []
    for x in reads:
        r = meta.get((x.team_a, x.team_b), {})
        opp = x.team_b if x.pick == x.team_a else x.team_a
        out.append({
            "Verdict": VERDICT_LABEL[x.verdict_key],
            "_ord": VERDICT_ORDER[x.verdict_key],
            "Pendul": "pendul" if x.is_pendulum else "",
            "Pick": x.pick,
            "Conf": round(x.pick_conf * 100, 1),
            "Opponent": opp,
            "Band": x.band_label,
            "Reliab": round(x.band_hit_rate * 100),
            "Tier": str(r.get("tier", "")).upper(),
            "Start": r.get("start_date", ""),
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df, reads
    df["Start"] = pd.to_datetime(df["Start"], errors="coerce", utc=True)
    df = df.sort_values(["_ord", "Conf"], ascending=[True, False]).drop(columns=["_ord"])
    return df.reset_index(drop=True), reads


def _results_summary() -> dict | None:
    tdf = tracker._load_all_snapshots()
    if tdf.empty:
        return None
    m = tracker._metrics_for(tdf, "pred_p_a")
    return m


def _control_panel():
    st.subheader("Panou de control")
    with st.expander("Setări rulare", expanded=False):
        c1, c2 = st.columns(2)
        days_ahead = c1.number_input("Zile înainte", 1, 14, 2, 1)
        tier_floor = c2.selectbox("Tier minim", ["s", "a", "b", "c", "d"], index=3)
    py = sys.executable or "python"
    b1, b2, b3 = st.columns(3)
    refresh = b1.button("Reîmprospătează + prezice", type="primary", use_container_width=True)
    predict = b2.button("Doar prezice", use_container_width=True)
    track = b3.button("Trage rezultatele", use_container_width=True)
    slot = st.container()
    args = ["--days-ahead", str(int(days_ahead)), "--tier-floor", tier_floor]
    if refresh:
        with slot, st.spinner("Reîmprospătez datele și rulez modelele... (până la 10 min)"):
            rc, out, err = pipeline._run_command([py, "scripts/predict_today.py", "--refresh-data", *args], 600)
        pipeline._show_result(rc, out, err, "refresh"); _load_predictions.clear(); pipeline._latest_report_date.clear()
    elif predict:
        with slot, st.spinner("Rulez modelele pe datele din cache..."):
            rc, out, err = pipeline._run_command([py, "scripts/predict_today.py", *args], 300)
        pipeline._show_result(rc, out, err, "predict"); _load_predictions.clear(); pipeline._latest_report_date.clear()
    elif track:
        with slot, st.spinner("Trag rezultatele de pe bo3.gg..."):
            rc, out, err = pipeline._run_command([py, "scripts/track_results.py"], 300)
        pipeline._show_result(rc, out, err, "track")
        tracker._load_all_snapshots.clear()


def _claude_panel(reads, summary):
    st.subheader("Briefing Claude")
    ok, reason = claude_brief.available()
    if not ok:
        st.info(
            f"Stratul Claude e inactiv — {reason}\n\n"
            "Ca să-l activezi: adaugă `ANTHROPIC_API_KEY=...` în `.env` și rulează "
            "`pip install anthropic`. Restul aplicației merge fără el."
        )
        return
    if st.button("Generează briefing", type="primary"):
        with st.spinner("Claude scrie briefing-ul..."):
            try:
                st.session_state["brief"] = claude_brief.briefing(reads, summary)
            except Exception as e:
                st.session_state["brief"] = f"(Eroare API: {e})"
    if st.session_state.get("brief"):
        st.markdown(st.session_state["brief"])
    st.markdown("---")
    q = st.text_input("Întreabă ceva despre slate", placeholder="de ce e Ence pe AVOID?")
    if st.button("Întreabă") and q:
        with st.spinner("Claude răspunde..."):
            try:
                st.markdown(claude_brief.ask(q, reads))
            except Exception as e:
                st.error(f"Eroare API: {e}")


def render():
    st.title("CS2-Predictor — camera de comandă")

    _control_panel()
    st.markdown("---")

    date = _latest_date()
    if not date:
        st.info("Niciun slate încă. Apasă **Reîmprospătează + prezice** ca să generezi predicții.")
        return
    rows = _load_predictions(date)
    df, reads = _slate_df(rows)
    summary = _results_summary()

    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Slate", date)
    m2.metric("Meciuri jucabile", len(df))
    if not df.empty:
        m3.metric("RIDE / AVOID", f"{int((df['Verdict']=='RIDE').sum())} / {int((df['Verdict']=='AVOID').sum())}")
    if summary:
        m4.metric("Model (istoric)", f"{summary['acc']*100:.0f}% · Brier {summary['brier']:.3f}")

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Ponturile de azi")
        st.caption("Etichete de disciplină din calibrarea pe 79 de meciuri. RIDE = banda fiabilă, AVOID = supra-încredere pedepsită.")
        if df.empty:
            st.info("Niciun meci jucabil în acest slate.")
        else:
            styler = df.style.apply(
                lambda col: [VERDICT_BG.get(v, "") for v in col] if col.name == "Verdict" else ["" for _ in col],
                axis=0,
            )
            st.dataframe(
                styler, hide_index=True, use_container_width=True,
                column_config={
                    "Start": st.column_config.DatetimeColumn("Start", format="MM-DD HH:mm"),
                    "Conf": st.column_config.NumberColumn("Conf %", format="%.1f"),
                    "Reliab": st.column_config.NumberColumn("Fiabil %", format="%d",
                                                            help="Cât a nimerit empiric banda asta pe 79 meciuri."),
                },
            )

    with right:
        _claude_panel(reads, summary)

    st.markdown("---")
    st.caption("Detalii complete: fila **Results tracker** (calibrare, Brier în timp) și **Transurfing** (briefing-ul integral).")
