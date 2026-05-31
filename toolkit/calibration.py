"""
Isotonic calibration. Trained on backtest (predicted_prob, outcome) pairs,
maps raw probabilities to calibrated frequencies via Pool-Adjacent-Violators.

Simple implementation without sklearn dependency.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class IsotonicCalibrator:
    """Piecewise-constant monotonic mapping from raw probs to calibrated probs."""
    breakpoints: list[float]    # sorted x-coordinates
    values: list[float]         # corresponding calibrated probs

    def calibrate(self, p: float) -> float:
        if not self.breakpoints:
            return p
        if p <= self.breakpoints[0]:
            return self.values[0]
        if p >= self.breakpoints[-1]:
            return self.values[-1]
        # Linear interpolation between adjacent breakpoints
        for i in range(1, len(self.breakpoints)):
            if p <= self.breakpoints[i]:
                x0, x1 = self.breakpoints[i-1], self.breakpoints[i]
                y0, y1 = self.values[i-1], self.values[i]
                t = (p - x0) / max(x1 - x0, 1e-9)
                return y0 + t * (y1 - y0)
        return self.values[-1]


def pava_isotonic_regression(x, y):
    """
    Pool-Adjacent-Violators algorithm for isotonic (monotonically increasing)
    regression. Returns sorted x's and corresponding fitted y values.
    """
    order = np.argsort(x)
    xs = x[order]; ys = y[order]
    n = len(xs)
    # Block representation
    block_starts = list(range(n))
    block_sums = ys.astype(float).copy().tolist()
    block_counts = [1] * n

    # Merge non-increasing adjacent blocks
    i = 0
    while i < len(block_sums) - 1:
        avg_i = block_sums[i] / block_counts[i]
        avg_j = block_sums[i+1] / block_counts[i+1]
        if avg_i > avg_j:
            block_sums[i] += block_sums[i+1]
            block_counts[i] += block_counts[i+1]
            del block_sums[i+1]; del block_counts[i+1]; del block_starts[i+1]
            if i > 0: i -= 1
        else:
            i += 1

    # Expand blocks back
    fitted = np.zeros(n)
    pos = 0
    for s, c in zip(block_sums, block_counts):
        avg = s / c
        for j in range(c):
            fitted[pos + j] = avg
        pos += c

    return xs, fitted


def fit_isotonic(probs, outcomes, n_breakpoints: int = 50) -> IsotonicCalibrator:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    xs, fitted = pava_isotonic_regression(probs, outcomes)

    # Subsample to keep the calibrator compact
    if len(xs) > n_breakpoints:
        idx = np.linspace(0, len(xs) - 1, n_breakpoints, dtype=int)
        xs = xs[idx]; fitted = fitted[idx]
    # Deduplicate consecutive equal x values
    bps, vals = [], []
    last_x = -1
    for x, v in zip(xs.tolist(), fitted.tolist()):
        if x > last_x + 1e-9:
            bps.append(float(x)); vals.append(float(v))
            last_x = x
        else:
            vals[-1] = (vals[-1] + v) / 2  # average
    return IsotonicCalibrator(breakpoints=bps, values=vals)


def save_calibrator(cal: IsotonicCalibrator, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"breakpoints": cal.breakpoints, "values": cal.values}, indent=2))


def load_calibrator(path) -> IsotonicCalibrator:
    d = json.loads(Path(path).read_text())
    return IsotonicCalibrator(breakpoints=d["breakpoints"], values=d["values"])
