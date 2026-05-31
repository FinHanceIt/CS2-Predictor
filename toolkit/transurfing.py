"""
Transurfing layer — a narrative + discipline post-processor over model output.

This module does NOT predict anything and does NOT touch reality. It takes the
predictions the real models already produced and annotates each one with:

  1. the empirical confidence band it falls into (from the 79-match validation),
  2. a "pendulum" flag (is this a heavy crowd favourite?),
  3. an "importance" verdict — how reliable that band has actually been.

The whole point is harm reduction: the bands where the model put *too much
importance* (60-70% and 80%+) historically underperformed, while the *calm*
band (70-80%) overperformed. This module makes that visible per match so the
user fades overconfidence instead of chasing it.

Vadim Zeland's vocabulary is used as a mnemonic skin — nothing here is
metaphysical. "Outer intention" = disciplined selection; "balancing forces" =
regression to the mean; "importance" = overconfidence. See docs/strat_transurfing.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# Empirical hit-rate per pick-confidence band, measured on 79 finished
# predictions (reports/results_tracker_*.json, May 2026). Update as the sample
# grows by re-running scripts/track_results.py and recomputing.
CALIBRATION_BANDS = [
    # (lo, hi, empirical_hit_rate, n, label, verdict_key)
    (0.50, 0.60, 0.52, 31, "monedă",      "neutral"),
    (0.60, 0.70, 0.43, 21, "importanță",  "avoid"),
    (0.70, 0.80, 0.80, 20, "calmă",       "ride"),
    (0.80, 1.01, 0.71, 7,  "favorit greu", "caution"),
]

VERDICTS = {
    "neutral": "Aproape de monedă. Joacă doar dacă există valoare clară la cotă.",
    "avoid":   "Importanță excesivă — empiric doar 43%. Redu miza sau sari peste.",
    "ride":    "Banda calmă — cea mai fiabilă (80% real). Poziție de încredere.",
    "caution": "Favorit greu — empiric 71%, sub cât pare. Atenție la echilibrare.",
}

PENDULUM_THRESHOLD = 0.70  # above this, the pick is a heavy public favourite


@dataclass
class TransurfingReading:
    team_a: str
    team_b: str
    pick: str
    pick_conf: float          # confidence in the picked side (0.5-1.0)
    band_label: str           # monedă / importanță / calmă / favorit greu
    band_hit_rate: float      # empirical hit rate of that band
    band_n: int               # sample size behind that hit rate
    is_pendulum: bool         # heavy crowd favourite?
    verdict_key: str          # ride / caution / avoid / neutral
    verdict: str              # human-readable discipline note

    def as_dict(self) -> dict:
        return asdict(self)


def _band_for(conf: float):
    for lo, hi, hit, n, label, key in CALIBRATION_BANDS:
        if lo <= conf < hi:
            return hit, n, label, key
    return CALIBRATION_BANDS[0][2:]  # fallback


def read_prediction(team_a: str, team_b: str, p_a: float) -> TransurfingReading:
    """Annotate a single prediction. p_a = ensemble prob that team_a wins."""
    if p_a >= 0.5:
        pick, conf = team_a, p_a
    else:
        pick, conf = team_b, 1.0 - p_a
    hit, n, label, key = _band_for(conf)
    return TransurfingReading(
        team_a=team_a, team_b=team_b, pick=pick, pick_conf=conf,
        band_label=label, band_hit_rate=hit, band_n=n,
        is_pendulum=conf >= PENDULUM_THRESHOLD,
        verdict_key=key, verdict=VERDICTS[key],
    )


def read_predictions(predictions: list[dict]) -> list[TransurfingReading]:
    """Annotate a list of prediction records (predictions.json schema)."""
    out = []
    for r in predictions:
        p_a = r.get("ensemble_p_a", r.get("pred_p_a"))
        if p_a is None or r.get("status") == "INSUFFICIENT_DATA":
            continue
        out.append(read_prediction(r["team_a"], r["team_b"], p_a))
    return out


def briefing(predictions: list[dict]) -> str:
    """Build a printable Transurfing briefing, sorted best-discipline first."""
    order = {"ride": 0, "caution": 1, "neutral": 2, "avoid": 3}
    reads = sorted(read_predictions(predictions), key=lambda x: (order[x.verdict_key], -x.pick_conf))
    icon = {"ride": "[RIDE ]", "caution": "[CARE ]", "neutral": "[FLAT ]", "avoid": "[AVOID]"}
    lines = []
    lines.append("TRANSURFING BRIEFING — disciplină peste predicțiile reale")
    lines.append(f"{'verdict':8} {'pendul':7} {'pick':26} {'conf':>5} {'bandă':>13} {'fiabil':>7}")
    lines.append("-" * 78)
    for x in reads:
        pend = "pendul" if x.is_pendulum else "  -   "
        lines.append(
            f"{icon[x.verdict_key]:8} {pend:7} {x.pick[:26]:26} "
            f"{x.pick_conf*100:>4.0f}% {x.band_label:>13} {x.band_hit_rate*100:>5.0f}%"
        )
    # Footing summary
    from collections import Counter
    c = Counter(x.verdict_key for x in reads)
    lines.append("-" * 78)
    lines.append(
        f"ride={c.get('ride',0)}  caution={c.get('caution',0)}  "
        f"flat={c.get('neutral',0)}  avoid={c.get('avoid',0)}  "
        f"(total {len(reads)})"
    )
    lines.append("Redu importanța: prioritizează RIDE, tratează AVOID ca semnal de stop.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse, json, sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Transurfing briefing over a predictions.json")
    ap.add_argument("--date", default=None, help="YYYYMMDD (default: latest report)")
    args = ap.parse_args()

    reports = Path(__file__).resolve().parents[1] / "reports"
    if args.date:
        pf = reports / f"{args.date}_cs2" / "predictions.json"
    else:
        candidates = sorted(reports.glob("*_cs2/predictions.json"))
        if not candidates:
            print("no predictions.json found"); sys.exit(1)
        pf = candidates[-1]

    preds = json.loads(pf.read_text())
    print(f"# source: {pf.parent.name}\n")
    print(briefing(preds))
