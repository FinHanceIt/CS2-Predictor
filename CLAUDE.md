# CLAUDE.md — CS2-Predictor

> Operating manual for the Counter-Strike 2 prediction workbench.
> Sibling project to Soccer-Predictor; same MRT principles, adapted for CS2.

---

## Project Identity

**CS2-Predictor** is a CS2 match prediction system. It takes any upcoming map or series from a supported tournament and produces calibrated probabilities for the standard markets (Match Winner, Map Winner, Total Maps, Map Handicap, Total Rounds), with explicit EV vs bookmaker odds.

The goal is high-frequency testing: CS2 produces 20-50 maps per day across active tournaments, so the system gets to learn from real outcomes daily — something that football off-season makes impossible.

This project deliberately mirrors the architecture of Soccer-Predictor, but the encoding, the data sources, the flat-space baseline, and the calibration constants are all CS2-specific.

---

## Why CS2 (vs other esports / sports)

- **HLTV.org provides free, standardized stats** at team/map/player level — best public data in any sport.
- **Bo3/Bo5 series produce multiple data points per series**, so a 5-map "match" gives 5 observations.
- **No draws** — clean binary outcome at the map level.
- **Persistent team identity** with detectable roster changes that MRT's holonomy concept can flag explicitly.
- **Daily testability** — even quiet weeks have 10+ maps, busy ones have 50+.

---

## Markets Covered

**Match-level (Bo3 / Bo5):**
- Match Winner (1/2, no draw)
- Total Maps (Over/Under 2.5 for Bo3, Over/Under 3.5/4.5 for Bo5)
- Correct Map Score (2-0 / 2-1 / 1-2 / 0-2 for Bo3)
- Handicap (-1.5 / +1.5 in maps)

**Map-level:**
- Map Winner (1/2)
- Total Rounds (Over/Under 26.5, 27.5, etc.)
- Round Handicap
- First Pistol Round Winner (1/2)
- Race to N Rounds (race to 9, race to 13)

---

## Standard Pipeline

```
User question (match/map, optional odds)
     │
     ▼
[1] Intake — resolve teams, event, format (Bo3/Bo5), map pool veto if known
     │
     ▼
[2] Data fetch:
     - HLTV recent matches (last ~30 maps per team)
     - HLTV per-map stats (round diffs, CT/T splits, pistol round outcomes)
     - HLTV current world ranking (as Elo proxy)
     - The Odds API → current market prices for the series
     │
     ▼
[3] Encoding:
     - Each team → Gaussian on Fisher-Rao manifold over (rounds_won_per_map, rounds_lost_per_map, ct_winrate, t_winrate)
     - Recent fixture path (last 8 maps) for holonomy computation
     │
     ▼
[4] Modeling:
     - MRT geometric prediction (geodesic + holonomy + resonance + curvature)
     - Flat-space baseline (binomial model on round-win-rate, or HLTV Elo lookup)
     - For multi-map series: simulate each map independently, combine into series probs
     │
     ▼
[5] EV vs market:
     - De-vig the bookmaker odds (Shin method, same as Soccer-Predictor)
     - Compute EV per market
     - Compute 1/8 Kelly stake for any +EV market with adequate MRT confidence
     │
     ▼
[6] Report:
     - Per-map prediction with both systems side-by-side
     - Series prediction (Bo3/Bo5 simulation)
     - Recommended bet(s) with confidence tier
     - Holonomy drift flags for any team with regime-shift signature
```

---

## Folder Structure

```
CS2-Predictor/
├── CLAUDE.md                  # this file
├── README.md
├── requirements.txt
├── .gitignore
├── .env                       # API keys (gitignored)
├── skills/
│   ├── cs2-match-predictor/SKILL.md   (TODO)
│   └── cs2-roster-watcher/SKILL.md    (TODO — detects roster changes)
├── toolkit/
│   ├── __init__.py
│   ├── hltv_data.py           # HLTV.org scraper / API
│   ├── odds_data.py           # The Odds API integration
│   ├── cs2_mrt.py             # MRT adapted for CS2 features
│   ├── cs2_baseline.py        # Flat-space binomial baseline
│   ├── series_sim.py          # Bo3/Bo5 series simulation
│   └── domain.py              # Kelly, EV, vig removal (copied from Soccer-Predictor)
├── reference/
│   ├── team_aliases.json      # HLTV name → canonical name lookup
│   └── tier1_teams.json       # current tier-1 roster (for filtering)
├── data/                      # runtime (gitignored)
│   ├── raw/                   # HLTV HTML dumps
│   └── clean/                 # parquet files
├── reports/                   # per-run outputs (gitignored except examples)
└── config/
    └── risk_rules.yaml        # stake limits — independent from Soccer-Predictor's
```

---

## Conventions

### Map DataFrame schema (canonical)

| Column | Type | Notes |
|---|---|---|
| `date` | `datetime64[ns, UTC]` | map start |
| `event` | `str` | tournament name |
| `team_a` | `str` | canonical team A name |
| `team_b` | `str` | canonical team B name |
| `map` | `str` | e.g. "Mirage", "Inferno", "Ancient" |
| `rounds_a` | `int` | rounds won by team A |
| `rounds_b` | `int` | rounds won by team B |
| `winner` | `str` | "team_a" / "team_b" |
| `ct_first` | `str` | which team started on CT side |
| `ct_winrate_a` | `float` | CT-side win rate for team A this map |
| `pistol_a_winner` | `bool` | did team A win the first pistol round |
| `pistol_b_winner` | `bool` | did team A win the second pistol round |
| `format` | `str` | "Bo3" / "Bo5" / "Bo1" |
| `series_id` | `str` | unique key linking maps in same series |
| `hltv_match_id` | `int` | for traceability |

### Confidence tiers (CS2-specific)

- **A-tier (1/4 Kelly)** — EV ≥ 5%, both teams have ≥ 20 maps in last 90 days, MRT confidence ≥ 0.70, no roster change in last 21 days
- **B-tier (1/8 Kelly)** — EV between 2-5%, both teams ≥ 10 maps in last 60 days, MRT confidence ≥ 0.55
- **C-tier (no stake)** — EV < 2% OR roster change within 21 days OR newcomer team OR MRT confidence < 0.55

### Acceptance Gates — strict

An A-tier prediction requires:
1. ✅ Both teams' recent map sample ≥ 20 (last 90 days)
2. ✅ No roster change in the last 21 days for either team
3. ✅ MRT confidence ≥ 0.70
4. ✅ Flat-space + MRT disagree by less than 8pp (otherwise: investigate, downgrade)
5. ✅ Map appears in both teams' active map pool

If any fail → downgrade.

---

## Smart Defaults

- **Historical sample**: last 90 days of maps per team
- **Recency decay**: λ = 0.015 per day (half-life ~46 days — CS2 meta shifts fast)
- **Map-pool elasticity**: 0.50 (teams play very differently on home maps vs away maps)
- **Holonomy path**: last 8 maps, window size 3
- **Min sigma clamp**: 1.0 round (prevents Fisher-Rao blowup on very stable teams)
- **Kelly fraction**: 1/8 (more conservative than Soccer-Predictor's 1/4 because CS2 has higher per-map variance)
- **Backtest origins**: 8 walk-forward over the last 60 days

---

## Anti-patterns — NEVER

- Predict before checking roster changes — a team with a brand new player is a different team entity
- Stake at A-tier on a team with < 20 maps in the last 90 days, regardless of their world ranking
- Treat a Bo3 series as one event — model each map independently, combine via series_sim
- Ignore map pool — a team's average rating is meaningless if you don't know which map is being played
- Use HLTV world ranking as the model — it's a useful sanity check / baseline, not a substitute for MRT
- Predict during a Major or LAN qualifier without explicit calibration — online tournaments and LAN have different variance profiles

---

## API Keys Required

| Service | Env var | Free? | How to get |
|---|---|---|---|
| The Odds API | `ODDS_API_KEY` | Yes (500/mo free) | already in Soccer-Predictor's .env — copy value |
| HLTV.org | none | Yes, scraping (respect robots.txt + rate limits) | scrape directly via library |
| PandaScore (fallback) | `PANDASCORE_KEY` | Yes (1000/mo free tier) | https://pandascore.co |

Put keys in `.env` (gitignored).

---

## Quick Start

When the user mentions a CS2 match:

1. Invoke the **`cs2-match-predictor`** skill.
2. The skill asks for missing info (at most 2 questions, batched).
3. Pipeline runs end-to-end; final report is saved to `./reports/<run_id>/report.md`.

