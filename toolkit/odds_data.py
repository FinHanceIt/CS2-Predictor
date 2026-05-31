"""
Odds I/O for CS2-Predictor.

Workflow:
  1. After `predict_today.py` runs, call `write_odds_template(predictions, path)`
     to write a YAML stub with one entry per upcoming match (odds blank).
  2. User opens the file, fills in Superbet/Betano odds.
  3. `load_odds(path)` parses it back into a dict keyed by match_id.

Format example:
  - match_id: 120345
    team_a: "G2 Ares"
    team_b: "Lilmix"
    book: "Superbet"        # optional, free-text
    odds:
      match_winner_a: 1.65
      match_winner_b: 2.30
      # supported markets:
      #   match_winner_a / match_winner_b
      #   handicap_minus_15_a / handicap_plus_15_a
      #   total_maps_over_25 / total_maps_under_25
      # leave any line blank if you don't have odds for that market
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json


# Use stdlib only — yaml is optional. We'll write/read a simple line-based format
# that's still YAML-compatible.

def write_odds_template(predictions: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CS2-Predictor odds template",
        "# Fill in decimal odds from your bookmaker (Superbet/Betano/etc.).",
        "# Leave blank or remove lines for markets you don't have.",
        "# `match_winner_a/b` is the only required market for EV calc.",
        "",
        "matches:",
    ]
    for r in predictions:
        if r.get("status") == "INSUFFICIENT_DATA":
            continue
        lines.append(f"  - match_id: {r['match_id']}")
        lines.append(f"    team_a: \"{r['team_a']}\"")
        lines.append(f"    team_b: \"{r['team_b']}\"")
        lines.append(f"    tier: {r['tier']}")
        lines.append(f"    bo: {r['bo']}")
        # Hint with model prob so user knows roughly what to expect
        ep = r.get('ensemble_p_a')
        if ep is not None:
            lines.append(f"    # ensemble says P(A wins series) = {ep*100:.1f}%")
        lines.append(f"    book: \"\"")
        lines.append(f"    odds:")
        lines.append(f"      match_winner_a:")
        lines.append(f"      match_winner_b:")
        lines.append("")
    p.write_text("\n".join(lines))


def load_odds(path: str | Path) -> dict[int, dict]:
    """
    Naive YAML parser tailored to the format above (no external dep).
    Returns dict keyed by match_id with structure:
      {match_id: {team_a, team_b, book, bo, tier, odds: {market: float}}}
    """
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[int, dict] = {}
    current = None
    in_odds = False
    for raw in p.read_text().splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith('#'):
            continue
        if line.startswith('  - match_id:'):
            if current is not None:
                out[current['match_id']] = current
            mid = int(line.split(':', 1)[1].strip())
            current = {'match_id': mid, 'odds': {}, 'team_a': '', 'team_b': '',
                       'book': '', 'bo': None, 'tier': None}
            in_odds = False
            continue
        if current is None:
            continue
        if line.startswith('    odds:'):
            in_odds = True
            continue
        if in_odds and line.startswith('      '):
            key, _, val = line.strip().partition(':')
            key = key.strip()
            v = val.strip()
            if v:
                try:
                    current['odds'][key] = float(v)
                except ValueError:
                    pass
            continue
        # top-level fields
        if line.startswith('    '):
            in_odds = False
            stripped = line.strip()
            key, _, val = stripped.partition(':')
            key = key.strip()
            v = val.strip().strip('"').strip("'")
            if key == 'team_a': current['team_a'] = v
            elif key == 'team_b': current['team_b'] = v
            elif key == 'book': current['book'] = v
            elif key == 'bo':
                try: current['bo'] = int(v)
                except ValueError: pass
            elif key == 'tier': current['tier'] = v
    if current is not None:
        out[current['match_id']] = current
    return out
