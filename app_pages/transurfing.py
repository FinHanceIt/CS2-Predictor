"""Transurfing page: the full discipline briefing over a chosen slate."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS_DIR = ROOT / "reports"
DOC = ROOT / "docs" / "strat_transurfing.md"

from toolkit import transurfing

LABEL = {"ride": "RIDE", "caution": "CARE", "neutral": "FLAT", "avoid": "AVOID"}
ORDER = {"ride": 0, "caution": 1, "neutral": 2, "avoid": 3}
BG = {
    "RIDE": "background-color: rgba(29,158,117,0.22); font-weight:600;",
    "CARE": "background-color: rgba(186,117,23,0.20); font-weight:600;",
    "FLAT": "background-color: rgba(136,135,128,0.16);",
    "AVOID": "background-color: rgba(226,75,74,0.20); font-weight:600;",
}


def _report_dates() -> list[str]:
    if not REPORTS_DIR.exists():
        return []
    out = []
    for p in REPORTS_DIR.glob("*_cs2"):
        stem = p.name.replace("_cs2", "")
        if p.is_dir() and len(stem) == 8 and stem.isdigit():
            out.append(stem)
    return sorted(out, reverse=True)


@st.cache_data(ttl=30)
def _load(date_str: str) -> list[dict]:
    p = REPORTS_DIR / f"{date_str}_cs2" / "predictions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def render():
    st.title("Transurfing — briefing de disciplină")
    st.caption(
        "Strat narativ peste motorul determinist. Nu schimbă nicio probabilitate — "
        "doar etichetează fiecare meci după banda lui empirică de încredere."
    )

    dates = _report_dates()
    if not dates:
        st.info("Niciun slate încă. Generează predicții din pagina **Home**.")
        return
    date = st.selectbox("Slate", dates, index=0)
    rows = _load(date)
    reads = transurfing.read_predictions(rows)
    if not reads:
        st.warning("Niciun meci jucabil în acest slate.")
        return

    counts = {k: 0 for k in LABEL}
    for x in reads:
        counts[x.verdict_key] += 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RIDE", counts["ride"], help="Banda calmă 70-80% — 80% real. Pozițiile de încredere.")
    c2.metric("CARE", counts["caution"], help="Favoriți grei 80%+ — 71% real. Atenție la echilibrare.")
    c3.metric("FLAT", counts["neutral"], help="Zona de monedă 50-60% — joci doar dacă e valoare la cotă.")
    c4.metric("AVOID", counts["avoid"], help="Banda 60-70% supra-încrezătoare — 43% real. Semnal de stop.")

    st.markdown("---")

    meta = {(r["team_a"], r["team_b"]): r for r in rows}
    data = []
    for x in reads:
        r = meta.get((x.team_a, x.team_b), {})
        data.append({
            "Verdict": LABEL[x.verdict_key],
            "_ord": ORDER[x.verdict_key],
            "Pendul": "pendul" if x.is_pendulum else "",
            "Pick": x.pick,
            "Conf": round(x.pick_conf * 100, 1),
            "Adversar": x.team_b if x.pick == x.team_a else x.team_a,
            "Bandă": x.band_label,
            "Fiabil": round(x.band_hit_rate * 100),
            "Tier": str(r.get("tier", "")).upper(),
        })
    df = pd.DataFrame(data).sort_values(["_ord", "Conf"], ascending=[True, False]).drop(columns=["_ord"])
    styler = df.style.apply(
        lambda col: [BG.get(v, "") for v in col] if col.name == "Verdict" else ["" for _ in col],
        axis=0,
    )
    st.dataframe(
        styler, hide_index=True, use_container_width=True,
        column_config={
            "Conf": st.column_config.NumberColumn("Conf %", format="%.1f"),
            "Fiabil": st.column_config.NumberColumn("Fiabil %", format="%d"),
        },
    )
    st.caption(
        "Redu importanța: prioritizează RIDE, tratează AVOID ca semnal de stop. "
        "`python -m toolkit.transurfing --date " + date + "` dă același briefing în terminal."
    )

    if DOC.exists():
        with st.expander("Maparea conceptelor (din docs/strat_transurfing.md)"):
            st.markdown(DOC.read_text(encoding="utf-8"))
