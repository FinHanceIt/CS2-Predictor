"""
CS2 data fetcher via bo3.gg public API.

bo3.gg is a CS2 statistics site with an open REST API at api.bo3.gg/api/v1/.
It's our primary data source because HLTV is Cloudflare-protected.

Key endpoints:
    GET /matches                — list matches (paginated, 10 per page locked)
    GET /matches/<slug>         — single-match detail
    GET /teams/<id>             — team info
    GET /tournaments/<id>       — tournament info

Important query params:
    filter[end_date][lte]=<iso>     — matches ending on/before date
    filter[end_date][gte]=<iso>
    sort=-end_date                  — most recent first
    page=<n>                        — pagination

Match-level data we get per record:
    id, slug, team1_id, team2_id, winner_team_id, tournament_id,
    team1_score (= maps won), team2_score (= maps won),
    bo_type (3 = Bo3), maps_score (list[bool] — true means team1 won that map),
    tier (s/a/b/c/d — quality of event), start_date, end_date,
    team1_last_game_score, team2_last_game_score (last map's round score).

What we DON'T get from this API:
    - per-map round scores for non-last maps
    - per-map map_name reliably linked to a specific match (the /games endpoint
      doesn't filter properly by match_id)
    - HLTV-style player KDR / ADR / rating

That's acceptable for MVP. We work with map-level binary outcomes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

API_BASE = "https://api.bo3.gg/api/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CS2-Predictor/1.0)",
    "Accept": "application/json",
}
DEFAULT_TIMEOUT = 20
RATE_LIMIT_SLEEP = 0.4  # seconds between requests


def _get_json(path: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET helper with retries and polite rate limiting."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            time.sleep(RATE_LIMIT_SLEEP)
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries}: {last_exc}")


def fetch_recent_matches(
    end_date_lte: datetime | None = None,
    days_back: int = 90,
    max_pages: int = 200,
    tier_floor: str = "c",  # accept tiers s, a, b, c (skip d which is amateur)
) -> pd.DataFrame:
    """
    Pull recent finished matches into a normalized DataFrame.

    Returns columns:
        match_id, slug, end_date, start_date, tier, bo_type,
        team1_id, team2_id, team1_score, team2_score,
        winner_team_id, maps_score (list[bool]),
        tournament_id
    """
    if end_date_lte is None:
        end_date_lte = datetime.now(timezone.utc)
    iso = end_date_lte.strftime("%Y-%m-%dT%H:%M:%S")
    cutoff = end_date_lte - timedelta(days=days_back)

    tier_order = ["s", "a", "b", "c", "d"]
    accept_tiers = set(tier_order[:tier_order.index(tier_floor) + 1])

    rows = []
    seen_ids = set()
    PAGE_SIZE = 10
    for offset in range(0, max_pages * PAGE_SIZE, PAGE_SIZE):
        params = {
            "filter[end_date][lte]": iso,
            "sort": "-end_date",
            "page[offset]": offset,
        }
        data = _get_json("matches", params=params)
        results = data.get("results", [])
        if not results:
            break
        oldest_in_page = None
        for m in results:
            if m["id"] in seen_ids:
                continue
            seen_ids.add(m["id"])
            try:
                end_dt = datetime.fromisoformat(m["end_date"].replace("Z", "+00:00"))
            except Exception:
                continue
            oldest_in_page = end_dt if oldest_in_page is None else min(oldest_in_page, end_dt)
            if m.get("status") != "finished":
                continue
            if m.get("tier") not in accept_tiers:
                continue
            if m["team1_id"] is None or m["team2_id"] is None:
                continue
            rows.append({
                "match_id": m["id"],
                "slug": m["slug"],
                "end_date": end_dt,
                "start_date": datetime.fromisoformat(m["start_date"].replace("Z", "+00:00")),
                "tier": m["tier"],
                "bo_type": m["bo_type"],
                "team1_id": m["team1_id"],
                "team2_id": m["team2_id"],
                "team1_score": m["team1_score"],
                "team2_score": m["team2_score"],
                "winner_team_id": m["winner_team_id"],
                "maps_score": m.get("maps_score") or [],
                "tournament_id": m.get("tournament_id"),
                "last_map_t1_rounds": m.get("team1_last_game_score"),
                "last_map_t2_rounds": m.get("team2_last_game_score"),
            })
        if oldest_in_page is not None and oldest_in_page < cutoff:
            break

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("end_date").reset_index(drop=True)
    return df


def fetch_upcoming_matches(days_ahead: int = 2) -> pd.DataFrame:
    """Matches scheduled in the next `days_ahead` days, with confirmed teams.

    bo3.gg API doesn't honor start_date filters, so we sort descending by
    start_date and walk through pages until we cross our window boundary.
    """
    now = datetime.now(timezone.utc)
    until = now + timedelta(days=days_ahead)

    rows = []
    seen = set()
    PAGE_SIZE = 10
    for offset in range(0, 800, PAGE_SIZE):
        params = {"sort": "-start_date", "page[offset]": offset}
        data = _get_json("matches", params=params)
        results = data.get("results", [])
        if not results:
            break
        oldest = None
        for m in results:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            try:
                start_dt = datetime.fromisoformat(m["start_date"].replace("Z", "+00:00"))
            except Exception:
                continue
            oldest = start_dt if oldest is None else min(oldest, start_dt)
            if m["team1_id"] is None or m["team2_id"] is None:
                continue
            if m.get("status") not in ("upcoming", "not_started"):
                continue
            if start_dt < now or start_dt > until:
                continue
            rows.append({
                "match_id": m["id"],
                "slug": m["slug"],
                "start_date": start_dt,
                "tier": m["tier"],
                "bo_type": m["bo_type"],
                "team1_id": m["team1_id"],
                "team2_id": m["team2_id"],
                "tournament_id": m.get("tournament_id"),
            })
        # Stop when we walked past our window
        if oldest is not None and oldest < now - timedelta(hours=12):
            break

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("start_date").reset_index(drop=True)
    return df


def fetch_team(team_id: int) -> dict:
    """Get team profile (name, slug, rank)."""
    try:
        return _get_json(f"teams/{team_id}")
    except Exception:
        return {"id": team_id, "name": f"Team#{team_id}", "slug": f"unknown_{team_id}", "rank": None}


def build_team_name_index(df_matches: pd.DataFrame, cache_path: str | Path = "reference/team_aliases.json") -> dict:
    """
    Build a {team_id: {"name": ..., "slug": ..., "rank": ...}} index, cached to disk.
    """
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        cache = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}

    team_ids = set(df_matches["team1_id"].dropna().astype(int)) | set(df_matches["team2_id"].dropna().astype(int))
    missing = team_ids - set(cache)
    print(f"Cached: {len(cache)} teams. Fetching {len(missing)} new...")
    for i, tid in enumerate(sorted(missing), 1):
        t = fetch_team(tid)
        cache[int(tid)] = {
            "name": t.get("name") or f"Team#{tid}",
            "slug": t.get("slug") or "",
            "rank": t.get("rank"),
        }
        if i % 25 == 0:
            print(f"  {i}/{len(missing)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({str(k): v for k, v in cache.items()}, indent=2))
    return cache


def explode_maps(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Each Bo3/Bo5 match expands into per-map rows.
    Returns columns: match_id, team_a_id, team_b_id, map_index, team_a_won, end_date, tier, bo_type
    where team_a/team_b are home/away as recorded (team1=A, team2=B).
    """
    rows = []
    for _, m in df_matches.iterrows():
        for i, won_by_t1 in enumerate(m["maps_score"]):
            rows.append({
                "match_id": m["match_id"],
                "team_a_id": m["team1_id"],
                "team_b_id": m["team2_id"],
                "map_index": i,
                "team_a_won": bool(won_by_t1),
                "end_date": m["end_date"],
                "tier": m["tier"],
                "bo_type": m["bo_type"],
            })
    return pd.DataFrame(rows)


def main():
    """CLI: cache last 90 days of matches + per-map exploded version + team index."""
    print("Fetching recent matches (last 90 days)...")
    df = fetch_recent_matches(days_back=90, tier_floor="c")
    print(f"  -> {len(df)} matches")

    out_dir = Path("data/clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "cs2_matches.parquet")
    print(f"  saved {out_dir / 'cs2_matches.parquet'}")

    maps_df = explode_maps(df)
    maps_df.to_parquet(out_dir / "cs2_maps.parquet")
    print(f"  saved {out_dir / 'cs2_maps.parquet'} ({len(maps_df)} per-map rows)")

    print("\nBuilding team index...")
    teams = build_team_name_index(df, "reference/team_aliases.json")
    print(f"  -> {len(teams)} teams")

    print(f"\nDate range: {df['end_date'].min().date()} -> {df['end_date'].max().date()}")
    print(f"Tiers:\n{df['tier'].value_counts()}")
    print(f"\nLatest 5 matches:")
    for _, m in df.tail(5).iterrows():
        t1 = teams.get(int(m['team1_id']), {}).get('name', '?')
        t2 = teams.get(int(m['team2_id']), {}).get('name', '?')
        score = f"{m['team1_score']}-{m['team2_score']}"
        print(f"  {m['end_date'].strftime('%Y-%m-%d %H:%M')} [{m['tier']}] {t1} {score} {t2}")


if __name__ == "__main__":
    main()
