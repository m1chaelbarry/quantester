"""Planted-edge synthetic market for the production-research tutorial.

The series is NOT a claim about real markets. It embeds a known AR(1)
autocorrelation in daily log-returns so a time-series momentum rule has a
genuine chronological pattern to detect. MCPT (which destroys chronology while
preserving the return distribution's moments) is then a fair null: if the
strategy's edge were only the unconditional drift, permutations would look
equally good.

Economic story (teaching fiction):
  "Asset exhibits short-horizon trend persistence from slow institutional
   rebalancing — exploitable by a delay-1 momentum rule, destroyed by
   shuffling log changes."
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
    """Daily OHLCV with AR(1) log-return persistence (seeded).

    Parameters
    ----------
    mu, sigma :
        Annualized drift and volatility of the innovation process.
    phi :
        AR(1) coefficient on lagged returns. ``phi > 0`` plants the momentum
        edge; ``phi = 0`` reduces to ordinary GBM (strategy should then fail
        MCPT unless luck intervenes).
    """
    if not -0.95 < phi < 0.95:
        raise ValueError("phi must lie in (-0.95, 0.95) for a stationary AR(1)")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bars, tz="UTC")
    eps = rng.normal(mu / 252.0, sigma / np.sqrt(252.0), size=n_bars)
    rets = np.empty(n_bars)
    rets[0] = eps[0]
    for t in range(1, n_bars):
        rets[t] = phi * rets[t - 1] + eps[t]
    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[s0], close[:-1]]) * (1.0 + rng.normal(0.0, 0.0008, n_bars))
    band = np.abs(rng.normal(0.0035, 0.0015, n_bars))
    high = np.maximum(open_, close) * (1.0 + band)
    low = np.minimum(open_, close) * (1.0 - band)
    # Keep OHLC physically valid after noise.
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
    """Hold out the final ``oos_fraction`` of bars as untouched OOS."""
    if not 0.05 <= oos_fraction <= 0.5:
        raise ValueError("oos_fraction should lie in [0.05, 0.5]")
    cut = int(len(df) * (1.0 - oos_fraction))
    if cut < 252 or len(df) - cut < 63:
        raise ValueError("split leaves too little IS or OOS history")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
