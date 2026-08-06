"""Seeded synthetic OHLCV data for examples and tests.

Geometric Brownian motion closes with intra-bar high/low spreads; volume is
lognormal. Deterministic under a fixed seed. One symbol's calendar can be made
strictly shorter (missing bars) to exercise availability masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(symbol: str = "SYN", n_bars: int = 750, s0: float = 100.0,
                         mu: float = 0.08, sigma: float = 0.20,
                         start: str = "2020-01-01", seed: int = 42,
                         missing_every: int | None = None) -> pd.DataFrame:
    """GBM daily bars. missing_every=k drops every k-th bar (after warmup) to
    simulate illiquid/stress gaps without erasing them from other symbols."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bars)
    rets = rng.normal(mu / 252, sigma / np.sqrt(252), size=n_bars)
    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[s0], close[:-1]]) * (1 + rng.normal(0, 0.001, n_bars))
    spread = np.abs(rng.normal(0.004, 0.002, n_bars))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=12.0, sigma=0.4, size=n_bars).round()

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="datetime"),
    )
    if missing_every:
        mask = np.ones(n_bars, dtype=bool)
        mask[50::missing_every] = False
        df = df.iloc[mask]
    return df


def write_csvs(data: dict, directory) -> dict:
    """Persist {symbol: DataFrame} as CSVs; returns {symbol: path}."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for symbol, df in data.items():
        path = directory / f"{symbol}.csv"
        df.to_csv(path, index_label="datetime")
        paths[symbol] = path
    return paths
