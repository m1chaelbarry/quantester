"""Seeded synthetic OHLCV data for examples and tests.

Geometric Brownian motion closes with intra-bar high/low spreads; volume is
lognormal. Deterministic under a fixed seed. One symbol's calendar can be made
strictly shorter (missing bars) to exercise availability masks.
`make_cointegrated_pair` builds a GLD/GDX-like pair with a known hedge ratio
and a stationary AR(1) log-spread for pairs-trading diagnostics.
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
    dates = pd.bdate_range(start=start, periods=n_bars, tz="UTC")
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


def make_cointegrated_pair(n_bars: int = 750, beta: float = 1.4,
                           spread_phi: float = 0.95, spread_sigma: float = 0.02,
                           gld_s0: float = 180.0, gdx_s0: float = 30.0,
                           gdx_mu: float = 0.04, gdx_sigma: float = 0.28,
                           start: str = "2020-01-01", seed: int = 7,
                           gdx_missing_every: int | None = None) -> dict:
    """Seeded synthetic cointegrated GLD/GDX-like daily OHLCV bars.

    ln GDX follows GBM; ln GLD = alpha + beta * ln GDX + e, where e is a
    stationary AR(1) spread (discrete Ornstein-Uhlenbeck: e_t = phi * e_{t-1}
    + sigma * eps_t), so the pair is cointegrated by construction with a KNOWN
    hedge ratio. alpha = ln(gld_s0) - beta * ln(gdx_s0), anchoring GLD at
    gld_s0 on the first bar. Deterministic under a fixed seed.

    gdx_missing_every=k drops every k-th GDX bar (after warmup) to exercise
    multi-symbol availability masks; GLD keeps the full calendar, so the
    master union calendar still contains those timestamps.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bars, tz="UTC")

    gdx_rets = rng.normal(gdx_mu / 252, gdx_sigma / np.sqrt(252), size=n_bars)
    log_gdx = np.log(gdx_s0) + np.cumsum(gdx_rets)

    spread = np.empty(n_bars)
    spread[0] = 0.0
    innovations = rng.normal(0.0, spread_sigma, size=n_bars)
    for t in range(1, n_bars):
        spread[t] = spread_phi * spread[t - 1] + innovations[t]
    alpha = np.log(gld_s0) - beta * np.log(gdx_s0)
    log_gld = alpha + beta * log_gdx + spread

    frames = {}
    for symbol, log_close in (("GLD", log_gld), ("GDX", log_gdx)):
        close = np.exp(log_close)
        open_ = np.concatenate([[close[0]], close[:-1]]) * (
            1 + rng.normal(0, 0.001, n_bars)
        )
        band = np.abs(rng.normal(0.004, 0.002, n_bars))
        high = np.maximum(open_, close) * (1 + band)
        low = np.minimum(open_, close) * (1 - band)
        volume = rng.lognormal(mean=15.0, sigma=0.4, size=n_bars).round()
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low,
             "close": close, "volume": volume},
            index=pd.DatetimeIndex(dates, name="datetime"),
        )
        if symbol == "GDX" and gdx_missing_every:
            mask = np.ones(n_bars, dtype=bool)
            mask[50::gdx_missing_every] = False
            df = df.iloc[mask]
        frames[symbol] = df
    return frames


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
