# CS2-Predictor

Counter-Strike 2 match prediction workbench.
Sibling project to Soccer-Predictor; uses the same MRT (Manifold Resonance Theory) framework.

See `CLAUDE.md` for the operating manual and architecture.

## Status

🚧 Project skeleton — under construction.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../Soccer-Predictor/.env .env   # reuse Odds API key
```

## First prediction

```bash
python scripts/fetch_today.py            # pulls today's CS2 matches from HLTV
python scripts/predict_all.py            # runs both predictors on each match
```

## GUI App (local)

A Streamlit-based dashboard is included so you don't have to run everything from the command line.

### First-time setup

```bash
pip install -r requirements.txt
```

### Launch

**Windows**: double-click `run_app.bat`
**macOS/Linux**: `./run_app.sh`
**Manual**: `streamlit run app.py`

The app opens at http://localhost:8501. Pages:

- **Dashboard** — today's predictions, strong picks
- **Pipeline** — buttons to refresh data + run prediction + track results
- **Odds entry** — form to enter Superbet/Betano odds per match
- **Bet evaluations** — EV/Kelly/tier recommendations
- **Results tracker** — historical accuracy + calibration plot + Brier-over-time

The app uses all the same scripts and data files as the CLI — running predictions in the app produces the same `reports/<date>_cs2/predictions.json` etc.

### Daily workflow

1. Open app
2. Pipeline tab → click "Refresh data + predict" (waits 1-3 min)
3. Dashboard tab → see strong picks
4. Odds entry tab → fill in odds for the 3-5 matches you care about
5. Bet evaluations tab → see A/B-tier picks with Kelly stakes
6. After matches finish: Tracker tab → click "Refresh results" to see how you did
