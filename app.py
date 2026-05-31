"""
CS2-Predictor — Streamlit GUI.

Personal local dashboard. Run with:
    streamlit run app.py

Pages:
  - Dashboard: today's predictions, strong picks
  - Pipeline:  buttons to refresh data, predict, track results
  - Odds:      enter bookmaker odds per match
  - Bets:      EV / Kelly / tier classification
  - Tracker:   historical accuracy + Brier over time
"""
import sys
from pathlib import Path
import streamlit as st

# Make toolkit/ importable when streamlit runs from project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Page modules
from app_pages import home, dashboard, transurfing, pipeline, odds, evaluations, tracker

# Bridge Streamlit Cloud secrets -> environment variables, so os.getenv-based
# code (toolkit/claude_brief.py, toolkit/odds_data.py) sees the keys when deployed.
import os
try:
    for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ODDS_API_KEY", "PANDASCORE_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

st.set_page_config(
    page_title="CS2-Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar nav
st.sidebar.title("CS2-Predictor")
st.sidebar.markdown("Personal CS2 prediction workbench")

PAGES = {
    "Home": home.render,
    "Dashboard": dashboard.render,
    "Transurfing": transurfing.render,
    "Pipeline":  pipeline.render,
    "Odds entry": odds.render,
    "Bet evaluations": evaluations.render,
    "Results tracker": tracker.render,
}
choice = st.sidebar.radio("Page", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption(f"Working dir: `{ROOT.name}`")

# Render selected page
PAGES[choice]()
