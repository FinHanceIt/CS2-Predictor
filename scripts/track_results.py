"""
Track real outcomes of predictions made on previous days.

Walks reports/*_cs2/predictions.json, fetches current finished status from
bo3.gg, joins on match_id, computes per-prediction-bucket accuracy + Brier.

Usage:
    python scripts/track_results.py            # cumulative since reports started
    python scripts/track_results.py --date 20260525   # just this day
"""
import argparse, json, sys, math
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toolkit.hltv_data import _get_json


def fetch_finished_status(match_ids: set[int]) -> dict[int, dict]:
    """Pull from bo3.gg until we find all match_ids OR walk past them."""
    found = {}
    seen = set()
    # Sort by end_date desc; walk pages
    for offset in range(0, 100, 10):
        if not (match_ids - found.keys()):
            break
        data = _get_json('matches', params={'sort': '-end_date', 'page[offset]': offset})
        results = data.get('results', [])
        if not results:
            break
        oldest_id_seen = min((m['id'] for m in results), default=None)
        for m in results:
            if m['id'] in seen: continue
            seen.add(m['id'])
            if m['id'] in match_ids and m.get('status') == 'finished':
                found[m['id']] = m
        # Stop heuristic: once we've walked enough pages, give up.
        if len(seen) > 100 and len(match_ids - found.keys()) > 0:
            break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYYMMDD; default = all dates")
    args = ap.parse_args()

    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("no reports/ directory")
        return

    pattern = f"{args.date}_cs2/predictions.json" if args.date else "*_cs2/predictions.json"
    pred_files = sorted(reports_dir.glob(pattern))
    if not pred_files:
        print(f"no predictions.json files matching {pattern}")
        return

    all_preds = []
    for pf in pred_files:
        date = pf.parent.name.split('_')[0]
        for r in json.loads(pf.read_text()):
            if r.get('status') == 'INSUFFICIENT_DATA': continue
            if r.get('ensemble_p_a') is None: continue
            r['_pred_date'] = date
            all_preds.append(r)
    print(f"loaded {len(all_preds)} predictions across {len(pred_files)} report files")

    match_ids = {p['match_id'] for p in all_preds}
    print(f"querying bo3.gg for {len(match_ids)} match outcomes...")
    finished = fetch_finished_status(match_ids)
    print(f"  -> {len(finished)} matches are now finished\n")

    rows = []
    for p in all_preds:
        m = finished.get(p['match_id'])
        if m is None: continue
        t1_won = (m['winner_team_id'] == m['team1_id'])
        rows.append({
            "date": p['_pred_date'], "tier": p['tier'], "bo": p['bo'],
            "team_a": p['team_a'], "team_b": p['team_b'],
            "score": f"{m['team1_score']}-{m['team2_score']}",
            "actual_a_won": t1_won,
            "pred_p_a": p['ensemble_p_a'],
            "mrt_p_a": p.get('mrt_p_a'),
            "flat_p_a": p.get('flat_p_a'),
            "mmassey_p_a": p.get('mmassey_p_a'),
            "mrt_conf": p.get('mrt_confidence'),
            "n_maps_a": p.get('team_a_n_maps'),
            "n_maps_b": p.get('team_b_n_maps'),
        })
    if not rows:
        print("(no completed predictions yet)")
        return

    # Per-model metrics
    def metrics(model_key):
        preds = [(r[model_key], r['actual_a_won']) for r in rows if r.get(model_key) is not None]
        n = len(preds)
        if n == 0: return None
        correct = sum(1 for p, y in preds if (p > 0.5) == y)
        brier = sum((p - (1.0 if y else 0.0))**2 for p, y in preds) / n
        eps = 1e-6
        ll = -sum(math.log(max(p if y else 1-p, eps)) for p, y in preds) / n
        return {'n': n, 'acc': correct/n, 'brier': brier, 'logloss': ll}

    print(f"{'Model':<10} {'N':>4} {'Acc':>7} {'Brier':>7} {'LogLoss':>8}")
    print("-" * 45)
    for k, label in [('mrt_p_a','MRT'),('flat_p_a','Flat'),('mmassey_p_a','M-Massey'),('pred_p_a','Ensemble')]:
        m = metrics(k)
        if m:
            print(f"{label:<10} {m['n']:>4} {m['acc']*100:>6.1f}% {m['brier']:>7.4f} {m['logloss']:>8.4f}")

    # Strong-picks subset
    strong = [r for r in rows if (r['pred_p_a'] >= 0.60 or r['pred_p_a'] <= 0.40)
              and r.get('mrt_conf') is not None and r['mrt_conf'] >= 0.55]
    if strong:
        sp_correct = sum(1 for r in strong if (r['pred_p_a'] >= 0.5) == r['actual_a_won'])
        print(f"\nStrong picks (>=60% / <=40% + MRT conf >=0.55):")
        print(f"  {sp_correct}/{len(strong)} = {sp_correct/len(strong)*100:.1f}% accuracy")

    # Per-match table
    print(f"\n{'Date':<10} {'Tier':<5} {'Match':<55} {'Pred':<7} {'Score':<7} {'Won':<5} {'Verdict':<10}")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: x['date']):
        v = "✓" if (r['pred_p_a'] >= 0.5) == r['actual_a_won'] else "✗"
        label = f"{r['team_a'][:25]} vs {r['team_b'][:25]}"
        print(f"{r['date']:<10} {r['tier']:<5} {label:<55} {r['pred_p_a']*100:>5.1f}% {r['score']:>5}   {'A' if r['actual_a_won'] else 'B':<5} {v:<10}")

    # Save snapshot
    out = Path(f"reports/results_tracker_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved snapshot: {out}")


if __name__ == "__main__":
    main()
