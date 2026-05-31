"""
Read predictions + filled-in odds YAML → output EV / Kelly / tier per bet.

Usage:
    # First: run predict_today.py to generate predictions.json + odds_template.yaml
    python scripts/predict_today.py

    # Then: open data/odds/upcoming_<date>.yaml, fill in odds, save.

    # Then: run this to evaluate:
    python scripts/evaluate_bets.py
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toolkit.odds_data import load_odds
from toolkit.domain import evaluate_bet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYYMMDD; default = today UTC")
    ap.add_argument("--bankroll", type=float, default=1000.0, help="bankroll in your currency")
    args = ap.parse_args()

    date = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
    report_dir = Path(f"reports/{date}_cs2")
    pred_path = report_dir / "predictions.json"
    odds_path = Path(f"data/odds/upcoming_{date}.yaml")

    if not pred_path.exists():
        print(f"!! {pred_path} not found. Run predict_today.py first.")
        sys.exit(1)
    if not odds_path.exists():
        print(f"!! {odds_path} not found. Fill in the template from data/odds/.")
        sys.exit(1)

    preds = json.loads(pred_path.read_text())
    odds_map = load_odds(odds_path)

    rows = []
    for r in preds:
        if r.get("status") == "INSUFFICIENT_DATA":
            continue
        mid = r["match_id"]
        entry = odds_map.get(mid)
        if not entry or not entry["odds"]:
            continue
        o = entry["odds"]
        o_a = o.get("match_winner_a")
        o_b = o.get("match_winner_b")
        if o_a is None or o_b is None:
            continue
        p_a = r.get("ensemble_p_a")
        if p_a is None:
            continue
        eval_a = evaluate_bet(
            "Match Winner", "a", o_a, p_a,
            pair_odds=o_b,
            mrt_confidence=r.get("mrt_confidence"),
            n_maps_a=r.get("team_a_n_maps"),
            n_maps_b=r.get("team_b_n_maps"),
        )
        eval_b = evaluate_bet(
            "Match Winner", "b", o_b, 1 - p_a,
            pair_odds=o_a,
            mrt_confidence=r.get("mrt_confidence"),
            n_maps_a=r.get("team_a_n_maps"),
            n_maps_b=r.get("team_b_n_maps"),
        )
        rows.append({"match": r, "book": entry.get("book", ""), "eval_a": eval_a, "eval_b": eval_b})

    if not rows:
        print("No matches have both predictions and odds filled in.")
        return

    # Report
    lines = [
        f"# Bet Evaluations — {date}",
        f"\nBankroll reference: {args.bankroll:,.0f}",
        f"\nMatches evaluated: {len(rows)}\n",
        "## Recommended bets (tier A or B only)\n",
        "| Match | Side | Pick | Book | Odds | Our P | Fair P | Edge | EV | Tier | Stake |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    bets = []
    for r in rows:
        for side_label, ev in [("A", r["eval_a"]), ("B", r["eval_b"])]:
            if ev.tier in ("A", "B"):
                pick = r["match"]["team_a"] if side_label == "A" else r["match"]["team_b"]
                opp = r["match"]["team_b"] if side_label == "A" else r["match"]["team_a"]
                bets.append((ev.tier, ev.ev, r, side_label, pick, opp))
    bets.sort(key=lambda x: (x[0], -x[1]))  # A first, then by EV desc

    if not bets:
        lines.append("| (none meet A/B tier criteria) |||||||||||")
    for tier_label, _, r, side_label, pick, opp in bets:
        ev = r["eval_a"] if side_label == "A" else r["eval_b"]
        stake = ev.kelly_scaled * args.bankroll
        lines.append(
            f"| {r['match']['team_a']} vs {r['match']['team_b']} (Bo{r['match']['bo']}, {r['match']['tier']}) "
            f"| {side_label} | **{pick}** | {r['book']} | {ev.decimal_odds:.2f} "
            f"| {ev.prob_estimate*100:.1f}% | {ev.implied_prob_fair*100:.1f}% "
            f"| {ev.edge*100:+.1f}pp | {ev.ev*100:+.1f}% | **{ev.tier}** "
            f"| {stake:.0f} ({ev.kelly_scaled*100:.2f}% bankroll) |"
        )

    # Full detail table for everything
    lines.append("\n## All evaluated matches (incl. C-tier / no stake)\n")
    lines.append("| Match | Book | Odds | Our P(A) | Fair P(A) | Edge A | EV A | EV B | Tier A | Tier B |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["match"]
        ea, eb = r["eval_a"], r["eval_b"]
        lines.append(
            f"| {m['team_a']} vs {m['team_b']} | {r['book']} "
            f"| {ea.decimal_odds:.2f}/{eb.decimal_odds:.2f} "
            f"| {ea.prob_estimate*100:.1f}% | {ea.implied_prob_fair*100:.1f}% "
            f"| {ea.edge*100:+.1f}pp | {ea.ev*100:+.1f}% | {eb.ev*100:+.1f}% "
            f"| {ea.tier} | {eb.tier} |"
        )

    lines.append("\n## Tier system (per CLAUDE.md)")
    lines.append("- **A-tier**: edge ≥ 5pp, both teams ≥ 20 maps in 60d, MRT conf ≥ 0.70. Stake = ¼ Kelly, capped at 2%.")
    lines.append("- **B-tier**: edge ≥ 2pp, both teams ≥ 10 maps in 60d, MRT conf ≥ 0.55. Stake = ⅛ Kelly, capped at 2%.")
    lines.append("- **C-tier**: anything else — no stake.")
    lines.append("\n*The cap at 2% of bankroll is a hard safety override.*")

    out = report_dir / "bet_evaluations.md"
    out.write_text("\n".join(lines))
    print(f"Saved {out}")
    print(f"Recommended bets: {len(bets)}")
    if bets:
        a_tier = sum(1 for b in bets if b[0] == "A")
        b_tier = sum(1 for b in bets if b[0] == "B")
        print(f"  A-tier: {a_tier} | B-tier: {b_tier}")


if __name__ == "__main__":
    main()
