"""Uncertainty for per-question metrics: bootstrap CIs and paired comparisons."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, low, high) percentile bootstrap CI. NaNs (metric undefined) are dropped."""
    x = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired comparison of system A vs B on the same questions.

    Returns the mean difference (A - B) with a bootstrap CI and a two-sided sign-flip
    permutation p-value. Pairs where either value is undefined are dropped.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and not np.isnan(x) and not np.isnan(y)]
    if not pairs:
        return {"n": 0, "diff": float("nan"), "low": float("nan"), "high": float("nan"), "p_value": float("nan")}
    d = np.asarray([x - y for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_boot, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    p = (np.sum(perm >= observed - 1e-12) + 1) / (n_boot + 1)
    return {
        "n": len(d),
        "diff": float(d.mean()),
        "low": float(np.quantile(boots, alpha / 2)),
        "high": float(np.quantile(boots, 1 - alpha / 2)),
        "p_value": float(p),
    }
