"""Small, dependency-free statistics: bootstrap CIs over cases, paired differences."""

from __future__ import annotations

import random
from statistics import mean


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(
    values: list[float], *, n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    Pass one value per *case* (the mean over that case's reps), not one per row: reps of
    the same case are correlated, and resampling rows would understate the uncertainty.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = [mean(rng.choices(values, k=n)) for _ in range(n_resamples)]
    alpha = (1 - confidence) / 2 * 100
    return (percentile(means, alpha), percentile(means, 100 - alpha))


def paired_diff(
    a: dict[str, float], b: dict[str, float], *, n_resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Mean of (b - a) over shared case ids, with a paired bootstrap CI.

    Pairing removes per-case difficulty from the comparison, so it detects smaller
    differences than comparing two independent CIs would.
    """
    shared = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in shared]
    if not diffs:
        return {"n": 0, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)
    return {"n": len(diffs), "mean_diff": mean(diffs), "ci_low": lo, "ci_high": hi}
