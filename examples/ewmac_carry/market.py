"""Synthetic BTC perpetual with funding / OI / DVOL extras (offline demo)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantester.utils.synthetic import make_synthetic_ohlcv

SYMBOL = "BTC/USDT:USDT"


def make_perp_ohlcv(n_bars: int = 800, seed: int = 8) -> pd.DataFrame:
    """GBM daily bars plus extras. Funding is mildly pro-cyclical with returns."""
    df = make_synthetic_ohlcv(
        "BTC", n_bars=n_bars, s0=40_000.0, mu=0.18, sigma=0.55,
        start="2021-01-01", seed=seed,
    )
    rng = np.random.default_rng(seed)
    rets = df["close"].pct_change().fillna(0.0)
    funding = (0.00015 + 0.4 * rets.clip(-0.02, 0.02) + rng.normal(0, 4e-5, n_bars))
    df["funding_rate"] = funding.to_numpy()
    df["open_interest"] = 8e9 * np.exp(np.cumsum(rng.normal(0.0, 0.008, n_bars)))
    df["dvol"] = 55.0 + 8.0 * rng.normal(size=n_bars)
    df.index.name = "datetime"
    return df


def split_is_oos(df: pd.DataFrame, oos_fraction: float = 0.25):
    cut = int(len(df) * (1.0 - oos_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
