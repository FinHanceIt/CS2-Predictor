"""Odds entry page — form-driven entry of bookmaker decimal odds."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from toolkit.odds_data import load_odds


REPORTS_DIR = Path("reports")
ODDS_DIR = Path("data/odds")


# ---------- helpers ----------

def _list_dates() -> list[str]:
    """Return sorted (newest-first) list of YYYYMMDD strings that have report dirs."""
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


def _odds_path(date_str: str) -> Path:
    return ODDS_DIR / f"upcoming_{date_str}.yaml"


def _write_odds_yaml(date_str: str, entries: list[dict]) -> None:
    """
    Serialize entries to the YAML format expected by toolkit.odds_data.load_odds.

    `entries` is a list of dicts with keys:
      match_id, team_a, team_b, tier, bo, ensemble_p_a (optional, for comment),
      book, odds_a, odds_b
    """
    path = _odds_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CS2-Predictor odds template",
        "# Fill in decimal odds from your bookmaker (Superbet/Betano/etc.).",
        "# Leave blank or remove lines for markets you don't have.",
        "# `match_winner_a/b` is the only required market for EV calc.",
        "",
        "matches:",
    ]
    for e in entries:
        lines.append(f"  - match_id: {e['match_id']}")
        lines.append(f"    team_a: \"{e.get('team_a', '')}\"")
        lines.append(f"    team_b: \"{e.get('team_b', '')}\"")
        if e.get('tier') is not None:
            lines.append(f"    tier: {e['tier']}")
        if e.get('bo') is not None:
            lines.append(f"    bo: {e['bo']}")
        ep = e.get('ensemble_p_a')
        if ep is not None:
            lines.append(f"    # ensemble says P(A wins series) = {ep*100:.1f}%")
        book = (e.get('book') or "").replace('"', '\\"')
        lines.append(f"    book: \"{book}\"")
        lines.append(f"    odds:")
        oa = e.get('odds_a')
        ob = e.get('odds_b')
        lines.append(f"      match_winner_a:{(' ' + format(oa, 'g')) if oa else ''}")
        lines.append(f"      match_winner_b:{(' ' + format(ob, 'g')) if ob else ''}")
        lines.append("")
    path.write_text("\n".join(lines))


def _reset_yaml(date_str: str, predictions: list[dict]) -> None:
    """Wipe odds back to a blank template (preserving all matches)."""
    entries = []
    for p in predictions:
        if p.get("status") == "INSUFFICIENT_DATA":
            continue
        entries.append({
            "match_id": p["match_id"],
            "team_a": p.get("team_a", ""),
            "team_b": p.get("team_b", ""),
            "tier": p.get("tier"),
            "bo": p.get("bo"),
            "ensemble_p_a": p.get("ensemble_p_a"),
            "book": "",
            "odds_a": None,
            "odds_b": None,
        })
    _write_odds_yaml(date_str, entries)


# ---------- main render ----------

def render() -> None:
    st.title("Odds entry")

    dates = _list_dates()
    if not dates:
        st.warning("No prediction reports found in `reports/`. "
                   "Run the prediction pipeline first.")
        return

    col_date, _ = st.columns([1, 3])
    with col_date:
        date_str = st.selectbox("Report date", dates, index=0)

    predictions = _load_predictions(date_str)
    if not predictions:
        st.error(f"`predictions.json` for {date_str} is missing or empty.")
        return

    # Keep only matches with enough data
    predictions = [p for p in predictions if p.get("status") != "INSUFFICIENT_DATA"]
    if not predictions:
        st.info("All matches for this date are INSUFFICIENT_DATA.")
        return

    existing = _load_existing_odds(date_str)

    # Filters
    fcol1, fcol2 = st.columns([2, 2])
    with fcol1:
        all_tiers = sorted({(p.get("tier") or "?").lower() for p in predictions})
        tier_filter = st.multiselect(
            "Tier filter", all_tiers, default=all_tiers,
        )
    with fcol2:
        strong_only = st.checkbox(
            "Show only strong picks (ensemble P ≥ 0.60 or ≤ 0.40)",
            value=False,
        )

    def _keep(p: dict) -> bool:
        t = (p.get("tier") or "?").lower()
        if t not in tier_filter:
            return False
        if strong_only:
            ep = p.get("ensemble_p_a")
            if ep is None:
                return False
            if 0.40 < ep < 0.60:
                return False
        return True

    visible = [p for p in predictions if _keep(p)]
    st.caption(f"Showing {len(visible)} of {len(predictions)} matches.")

    if not visible:
        st.info("No matches match the current filters.")
        # Still allow reset
        _render_reset_button(date_str, predictions)
        return

    # Build form
    with st.form("odds_form", clear_on_submit=False):
        # Header row
        h = st.columns([2.0, 2.0, 0.6, 0.5, 1.0, 1.0, 1.0, 1.4])
        h[0].markdown("**Team A**")
        h[1].markdown("**Team B**")
        h[2].markdown("**Tier**")
        h[3].markdown("**Bo**")
        h[4].markdown("**P(A)**")
        h[5].markdown("**Odds A**")
        h[6].markdown("**Odds B**")
        h[7].markdown("**Book**")

        entered: dict[int, dict] = {}
        for p in visible:
            mid = p["match_id"]
            ex = existing.get(mid, {})
            ex_odds = ex.get("odds", {}) if ex else {}
            default_a = ex_odds.get("match_winner_a")
            default_b = ex_odds.get("match_winner_b")
            default_book = ex.get("book") or ""

            row = st.columns([2.0, 2.0, 0.6, 0.5, 1.0, 1.0, 1.0, 1.4])
            row[0].write(p.get("team_a", ""))
            row[1].write(p.get("team_b", ""))
            row[2].write((p.get("tier") or "?").upper())
            row[3].write(str(p.get("bo", "")))
            ep = p.get("ensemble_p_a")
            row[4].write(f"{ep*100:.1f}%" if ep is not None else "—")

            odds_a = row[5].number_input(
                "Odds A",
                key=f"oa_{date_str}_{mid}",
                min_value=0.0,
                max_value=100.0,
                step=0.01,
                value=float(default_a) if default_a else 0.0,
                format="%.2f",
                label_visibility="collapsed",
            )
            odds_b = row[6].number_input(
                "Odds B",
                key=f"ob_{date_str}_{mid}",
                min_value=0.0,
                max_value=100.0,
                step=0.01,
                value=float(default_b) if default_b else 0.0,
                format="%.2f",
                label_visibility="collapsed",
            )
            book = row[7].text_input(
                "Book",
                key=f"bk_{date_str}_{mid}",
                value=default_book,
                label_visibility="collapsed",
            )

            entered[mid] = {
                "odds_a": odds_a if odds_a > 0 else None,
                "odds_b": odds_b if odds_b > 0 else None,
                "book": book,
            }

        submitted = st.form_submit_button("Save odds", type="primary")

    if submitted:
        # Merge with all predictions (not just visible) so we don't lose
        # filtered-out matches.
        merged_entries: list[dict] = []
        for p in predictions:
            mid = p["match_id"]
            if mid in entered:
                # use just-entered values
                e = entered[mid]
                merged_entries.append({
                    "match_id": mid,
                    "team_a": p.get("team_a", ""),
                    "team_b": p.get("team_b", ""),
                    "tier": p.get("tier"),
                    "bo": p.get("bo"),
                    "ensemble_p_a": p.get("ensemble_p_a"),
                    "book": e["book"],
                    "odds_a": e["odds_a"],
                    "odds_b": e["odds_b"],
                })
            else:
                # preserve prior values from disk
                ex = existing.get(mid, {})
                ex_odds = ex.get("odds", {}) if ex else {}
                merged_entries.append({
                    "match_id": mid,
                    "team_a": p.get("team_a", ""),
                    "team_b": p.get("team_b", ""),
                    "tier": p.get("tier"),
                    "bo": p.get("bo"),
                    "ensemble_p_a": p.get("ensemble_p_a"),
                    "book": ex.get("book", ""),
                    "odds_a": ex_odds.get("match_winner_a"),
                    "odds_b": ex_odds.get("match_winner_b"),
                })
        _write_odds_yaml(date_str, merged_entries)
        _load_existing_odds.clear()
        try:
            st.toast("Odds saved.", icon="✅")
        except Exception:
            pass
        st.success(f"Saved to `{_odds_path(date_str)}`.")

    st.markdown("---")
    _render_reset_button(date_str, predictions)


def _render_reset_button(date_str: str, predictions: list[dict]) -> None:
    """Render the reset block with confirmation."""
    confirm_key = f"confirm_reset_{date_str}"
    st.session_state.setdefault(confirm_key, False)

    cols = st.columns([1, 1, 4])
    if not st.session_state[confirm_key]:
        if cols[0].button("Reset all odds", type="secondary"):
            st.session_state[confirm_key] = True
            st.rerun()
    else:
        cols[0].warning("Confirm reset?")
        if cols[1].button("Yes, wipe all odds", type="primary"):
            _reset_yaml(date_str, predictions)
            _load_existing_odds.clear()
            st.session_state[confirm_key] = False
            # Clear all per-input session_state values so widgets reflect reset
            for p in predictions:
                mid = p["match_id"]
                for prefix in ("oa_", "ob_", "bk_"):
                    k = f"{prefix}{date_str}_{mid}"
                    if k in st.session_state:
                        del st.session_state[k]
            st.success("All odds reset to blank template.")
            st.rerun()
        if cols[2].button("Cancel"):
            st.session_state[confirm_key] = False
            st.rerun()
