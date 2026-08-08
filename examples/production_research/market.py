"""Synthetic market with a *planted* momentum edge (teaching fiction).

Why synthetic?
    Offline, reproducible, and we *know* the answer: AR(1) persistence
    (``phi > 0``) gives a chronological pattern a delay-1 momentum rule can
    detect. MCPT shuffles chronology while keeping the return distribution —
    so if the edge were only unconditional drift, permutations would score
    equally well.

This is NOT a claim about real markets. Real data will not gift you this.
The point of the example is the **shape of the research process**.

Economic story (made up for teaching)
-------------------------------------
"Asset exhibits short-horizon trend persistence from slow institutional
rebalancing — exploitable by a delay-1 momentum rule, destroyed by shuffling
log changes."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_momentum_edge_ohlcv(
    symbol: str = "EDGE",
    n_bars: int = 1_512,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.16,
    phi: float = 0.30,
    start: str = "2018-01-01",
    seed: int = 7,
) -> pd.DataFrame:
    """Build daily OHLCV where log-returns follow a seeded AR(1).

    Parameters you might tweak
    --------------------------
    mu, sigma :
        Annualized drift and volatility of the innovation process.
    phi :
        AR(1) coefficient on lagged returns.
        ``phi > 0`` plants the momentum edge.
        ``phi = 0`` → ordinary GBM (strategy should then fail MCPT).
    seed :
        All RNG goes through ``numpy.random.Generator(seed)`` — no global state.
    """
    if not -0.95 < phi < 0.95:
        raise ValueError("phi must lie in (-0.95, 0.95) for a stationary AR(1)")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bars, tz="UTC")

    # Innovations ~ N(daily drift, daily vol).
    eps = rng.normal(mu / 252.0, sigma / np.sqrt(252.0), size=n_bars)

    # AR(1): r_t = phi * r_{t-1} + eps_t  ← this is the planted edge.
    rets = np.empty(n_bars)
    rets[0] = eps[0]
    for t in range(1, n_bars):
        rets[t] = phi * rets[t - 1] + eps[t]

    # Price path from cumulative log-returns.
    close = s0 * np.exp(np.cumsum(rets))

    # Open ≈ previous close + tiny noise; high/low = open/close ± band.
    open_ = np.concatenate([[s0], close[:-1]]) * (1.0 + rng.normal(0.0, 0.0008, n_bars))
    band = np.abs(rng.normal(0.0035, 0.0015, n_bars))
    high = np.maximum(open_, close) * (1.0 + band)
    low = np.minimum(open_, close) * (1.0 - band)

    # Keep OHLC physically valid after noise (high ≥ max(o,c), low ≤ min(o,c)).
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.lognormal(mean=14.0, sigma=0.35, size=n_bars).round()

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="datetime"),
    )


def split_is_oos(
    df: pd.DataFrame, oos_fraction: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the final ``oos_fraction`` of bars as untouched out-of-sample.

    Rules of the game
    -----------------
    - Optimize / tune / walk-forward **only** on the IS slice.
    - Look at OOS **once**, with locked parameters, at the very end.
    - Re-optimizing on OOS voids the validation.
    """
    if not 0.05 <= oos_fraction <= 0.5:
        raise ValueError("oos_fraction should lie in [0.05, 0.5]")
    cut = int(len(df) * (1.0 - oos_fraction))
    if cut < 252 or len(df) - cut < 63:
        raise ValueError("split leaves too little IS or OOS history")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
