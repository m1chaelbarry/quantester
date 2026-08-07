"""Advanced bootstrap and drawdown bounding (MC Report section 4).

Drawdown is path-dependent and order-sensitive: a single-loop bootstrap
resampling the OOS returns UNDERESTIMATES catastrophic drawdown by more than a
factor of 10, because the OOS sample itself is a volatile draw from the parent
population (anti-conservative). Masters' nested double bootstrap ("bound on a
bound") accounts for both:

- inner loop (N_inner): path-dependent sequencing -> max drawdown distribution
  over horizon H; record the DD_conf quantile
- outer loop (N_outer): resamples the empirical population -> sampling error of
  the historical dataset; the final bound is the Bound_conf quantile of the
  outer distribution

Recommended: DD_conf = 0.95, Bound_conf = 0.70, N_outer = 10,000, N_inner = 1,000.
(Exact quantile indices were not covered by the notebook; quantile forms per
the Monte Carlo report's algorithm description.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def max_drawdown_of_returns(returns: np.ndarray) -> float:
    """Fractional peak-to-trough drawdown of a **simple-return** path.

    Wealth is constructed geometrically:

        equity = cumprod(1 + simple_returns)   (starts at 1.0)

    and the returned depth is ``max((HWM - equity) / HWM)`` — a non-negative
    fraction comparable in magnitude to ``abs(analytics.performance.max_drawdown)``.

    Do not pass additive P&L or log returns here; convert explicitly first.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    if np.any(~np.isfinite(r)):
        raise ValueError("returns contain NaN/inf")
    if np.any(r <= -1.0):
        raise ValueError("simple returns must be > -1 for wealth construction")
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    hwm = np.maximum.accumulate(equity)
    return float(((hwm - equity) / np.maximum(hwm, 1e-12)).max())


def single_loop_dd_quantile(returns, horizon: int, n_sims: int, conf: float,
                            seed: int | None = None) -> float:
    """Anti-conservative benchmark: resample OOS returns, take conf quantile."""
    pool = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    dds = np.empty(n_sims)
    for i in range(n_sims):
        sample = pool[rng.integers(0, len(pool), size=horizon)]
        dds[i] = max_drawdown_of_returns(sample)
    return float(np.quantile(dds, conf))


@dataclass
class DDBoundResult:
    bound: float
    outer_distribution: np.ndarray
    dd_conf: float
    bound_conf: float
    horizon: int


def double_bootstrap_dd_bound(returns, horizon: int | None = None,
                              dd_conf: float = 0.95, bound_conf: float = 0.70,
                              n_outer: int = 10_000, n_inner: int = 1_000,
                              seed: int | None = None) -> DDBoundResult:
    """Masters' nested double bootstrap 'bound on a bound'."""
    pool = np.asarray(returns, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0:
        raise ValueError("empty return pool")
    horizon = horizon or len(pool)
    rng = np.random.default_rng(seed)

    dd_outer = np.empty(n_outer)
    for outer in range(n_outer):
        outer_sample = pool[rng.integers(0, len(pool), size=len(pool))]
        dd_inner = np.empty(n_inner)
        for inner in range(n_inner):
            path = outer_sample[rng.integers(0, len(outer_sample), size=horizon)]
            dd_inner[inner] = max_drawdown_of_returns(path)
        dd_outer[outer] = np.quantile(dd_inner, dd_conf)

    return DDBoundResult(
        bound=float(np.quantile(dd_outer, bound_conf)),
        outer_distribution=dd_outer,
        dd_conf=dd_conf,
        bound_conf=bound_conf,
        horizon=horizon,
    )
