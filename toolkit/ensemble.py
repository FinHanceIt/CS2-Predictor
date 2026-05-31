"""
Ensemble of MRT + Flat + Massey predictions with proper grid search + CV.
"""
from __future__ import annotations
import math, json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Ensemble:
    weights: dict[str, float]
    intercept: float = 0.0

    def predict(self, model_probs: dict[str, float | None]) -> float | None:
        available = {k: v for k, v in model_probs.items() if v is not None and k in self.weights}
        if not available:
            return None
        total_w = sum(self.weights[k] for k in available)
        if total_w <= 0:
            return None
        p = sum(self.weights[k] * available[k] for k in available) / total_w
        if abs(self.intercept) > 1e-9:
            logit = math.log(max(p, 1e-9) / max(1 - p, 1e-9)) + self.intercept
            p = 1.0 / (1.0 + math.exp(-logit))
        return max(1e-6, min(1.0 - 1e-6, p))


def logloss(p, y):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def align_predictions(per_model_records: dict[str, list]) -> dict:
    """Returns dict(P, y, model_names, keys) for records where ALL models predicted."""
    indexed = {}
    for name, recs in per_model_records.items():
        idx = {}; cur_mid = None; pos = 0
        for r in recs:
            mid = r[3]
            if mid != cur_mid:
                cur_mid = mid; pos = 0
            idx[(mid, pos)] = r
            pos += 1
        indexed[name] = idx
    common = set.intersection(*(set(idx.keys()) for idx in indexed.values()))
    model_names = list(per_model_records.keys())
    keys = sorted(common)
    n = len(keys)
    P = np.zeros((len(model_names), n))
    y = np.zeros(n)
    for ki, key in enumerate(keys):
        rec0 = indexed[model_names[0]][key]
        y[ki] = 1.0 if rec0[1] else 0.0
        for mi, name in enumerate(model_names):
            P[mi, ki] = max(1e-6, min(1.0 - 1e-6, indexed[name][key][0]))
    return {"P": P, "y": y, "model_names": model_names, "keys": keys}


def fit_intercept(p, y, max_iter=200, lr=0.5) -> float:
    """Find b that minimizes log-loss of σ(logit(p) + b)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    base = np.log(p / (1 - p))
    b = 0.0
    for _ in range(max_iter):
        q = 1.0 / (1.0 + np.exp(-(base + b)))
        grad = float(np.mean(q - y))
        b -= lr * grad
        if abs(grad) < 1e-5: break
    return float(b)


def grid_search_weights(P, y, step=0.05, fit_b=True) -> tuple[np.ndarray, float, float]:
    """Best (w, b, loss) over simplex of 3 model weights with given step."""
    n_models = P.shape[0]
    best = (None, 0.0, math.inf)
    # Enumerate simplex with 3 components
    grid = np.arange(0, 1 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9: continue
            w3 = max(0.0, min(1.0, w3))
            w = np.array([w1, w2, w3])
            avg = P.T @ w
            if fit_b:
                b = fit_intercept(avg, y)
                logits = np.log(np.clip(avg, 1e-6, 1-1e-6) / np.clip(1-avg, 1e-6, 1-1e-6)) + b
                pred = 1.0 / (1.0 + np.exp(-logits))
            else:
                b = 0.0
                pred = avg
            loss = logloss(pred, y)
            if loss < best[2]:
                best = (w, b, loss)
    return best


def fit_ensemble_cv(per_model_records: dict[str, list], split_match_id_threshold=None) -> dict:
    """Split records by match_id into early/late halves. Fit on early, score on late.
    Returns dict of {ensemble: Ensemble, in_sample: {...}, out_of_sample: {...}}."""
    aligned = align_predictions(per_model_records)
    P, y, names, keys = aligned["P"], aligned["y"], aligned["model_names"], aligned["keys"]

    # Split by match_id (preserves time order ~roughly)
    if split_match_id_threshold is None:
        # use median match_id as splitter
        mids = [k[0] for k in keys]
        split_match_id_threshold = sorted(mids)[len(mids) // 2]

    early_mask = np.array([k[0] < split_match_id_threshold for k in keys])
    late_mask  = ~early_mask
    P_train, y_train = P[:, early_mask], y[early_mask]
    P_test,  y_test  = P[:, late_mask],  y[late_mask]

    # Grid search on training set
    w_train, b_train, loss_train = grid_search_weights(P_train, y_train, step=0.05)
    # Score on test set
    avg_test = P_test.T @ w_train
    logits = np.log(np.clip(avg_test, 1e-6, 1-1e-6) / np.clip(1-avg_test, 1e-6, 1-1e-6)) + b_train
    pred_test = 1.0 / (1.0 + np.exp(-logits))

    ens = Ensemble(weights={n: float(w_train[i]) for i, n in enumerate(names)},
                   intercept=float(b_train))

    return {
        "ensemble": ens,
        "train_size": int(early_mask.sum()),
        "test_size":  int(late_mask.sum()),
        "in_sample":  {"brier": brier(1.0 / (1.0 + np.exp(-(np.log(np.clip(P_train.T @ w_train, 1e-6, 1-1e-6) / np.clip(1-(P_train.T @ w_train), 1e-6, 1-1e-6)) + b_train))), y_train),
                       "logloss": loss_train},
        "out_of_sample": {"brier": brier(pred_test, y_test), "logloss": logloss(pred_test, y_test)},
        "per_model_oos": {n: {"brier": brier(P_test[i], y_test), "logloss": logloss(P_test[i], y_test)}
                          for i, n in enumerate(names)},
    }


def save_ensemble(ens: Ensemble, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"weights": ens.weights, "intercept": ens.intercept}, indent=2))


def load_ensemble(path) -> Ensemble:
    d = json.loads(Path(path).read_text())
    return Ensemble(weights=d["weights"], intercept=d.get("intercept", 0.0))
