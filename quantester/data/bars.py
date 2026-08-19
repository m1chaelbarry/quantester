"""Information-driven bar sampling (AFML ch.2).

- Dollar bars: sample whenever a constant market value has exchanged hands,
  robust to price volatility and corporate actions (Report 2 section 3).
- Tick imbalance bars (TIBs): b_t = b_{t-1} if dp == 0 else sign(dp);
  theta_T = sum(b_t); expected imbalance E_0[theta_T] = E_0[T] * (2P[b=1] - 1)
  with E_0[T] an EWMA of prior bar lengths and (2P[b=1]-1) an EWMA of prior
  bars' mean signed flow (theta_T / T). A bar is emitted at
  T* = argmin{ |theta_T| >= E_0[theta_T] }.
- Dollar/volume imbalance bars: theta_T = sum(b_t v_t);
  E_0[theta_T] = E_0[T] * (2v+ - E_0[v_t]) via EWMA of per-bar mean signed size.

Notebook-verified: AFML ch.2 sampling *condition* (product E_0[T] × |imbalance|).
Bar-level imbalance EWMA (one observation per completed bar, not concatenated
ticks) is from AFML ch.2 / literature-remediation research 02 (Quant.SE TIB
extract + de Prado 2017 LBNL slides). Warmup constant threshold and leftover
flush are extra vs the text and are unchanged.

Input ticks: pd.DataFrame indexed by datetime with columns price, volume.
Output bars: pd.DataFrame with open/high/low/close/volume indexed by bar-close time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_bar(ticks: pd.DataFrame) -> pd.Series:
    prices = ticks["price"]
    return pd.Series(
        {
            "open": float(prices.iloc[0]),
            "high": float(prices.max()),
            "low": float(prices.min()),
            "close": float(prices.iloc[-1]),
            "volume": float(ticks["volume"].sum()),
        },
        name=ticks.index[-1],
    )


def dollar_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Sample a bar every time cumulative dollar value (price*volume) >= threshold."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    bars = []
    buffer_idx, buffer_px, buffer_vol = [], [], []
    cumulative = 0.0
    for ts, price, volume in zip(ticks.index, ticks["price"].to_numpy(), ticks["volume"].to_numpy()):
        buffer_idx.append(ts)
        buffer_px.append(price)
        buffer_vol.append(volume)
        cumulative += price * volume
        if cumulative >= threshold:
            chunk = pd.DataFrame(
                {"price": buffer_px, "volume": buffer_vol}, index=pd.DatetimeIndex(buffer_idx)
            )
            bars.append(_make_bar(chunk))
            buffer_idx, buffer_px, buffer_vol = [], [], []
            cumulative = 0.0
    if buffer_idx:
        chunk = pd.DataFrame(
            {"price": buffer_px, "volume": buffer_vol}, index=pd.DatetimeIndex(buffer_idx)
        )
        bars.append(_make_bar(chunk))
    out = pd.DataFrame(bars)
    if not out.empty:
        out.index = pd.DatetimeIndex(out.index)
    return out


def _tick_rule(prices: np.ndarray) -> np.ndarray:
    """b_t = b_{t-1} if dp == 0 else sign(dp); b_0 = sign of first nonzero move."""
    b = np.zeros(len(prices))
    prev = 1.0
    for i in range(len(prices)):
        if i == 0:
            b[i] = prev
            continue
        dp = prices[i] - prices[i - 1]
        if dp > 0:
            prev = 1.0
        elif dp < 0:
            prev = -1.0
        b[i] = prev
    return b


def _ewma_last(values: list, span: int) -> float:
    return float(pd.Series(values, dtype=float).ewm(span=span).mean().iloc[-1])


def _imbalance_bars(ticks: pd.DataFrame, weighted: bool, span: int, warmup: int,
                    initial_expected_len: float) -> pd.DataFrame:
    prices = ticks["price"].to_numpy()
    volumes = ticks["volume"].to_numpy()
    index = ticks.index
    b = _tick_rule(prices)
    flow = b * volumes if weighted else b

    bars = []
    bar_lengths: list = []
    bar_imbalances: list = []  # per-bar mean signed flow (AFML: prior bars)

    start = 0
    theta = 0.0
    for t in range(len(ticks)):
        theta += flow[t]
        length = t - start + 1
        if len(bar_lengths) >= warmup:
            expected_len = _ewma_last(bar_lengths, span)
            expected_imb = abs(_ewma_last(bar_imbalances, span))
            threshold = max(expected_len * expected_imb, 1e-12)
        else:
            threshold = max(initial_expected_len, 1.0)
        if abs(theta) >= threshold:
            chunk = ticks.iloc[start : t + 1]
            bars.append(_make_bar(chunk))
            bar_lengths.append(length)
            bar_imbalances.append(float(theta) / float(length))
            start = t + 1
            theta = 0.0
    if start < len(ticks):
        bars.append(_make_bar(ticks.iloc[start:]))
    out = pd.DataFrame(bars)
    if not out.empty:
        out.index = pd.DatetimeIndex(out.index)
    return out


def tick_imbalance_bars(ticks: pd.DataFrame, span: int = 10, warmup: int = 3,
                        initial_expected_len: float = 50.0) -> pd.DataFrame:
    """TIBs: theta_T = sum(b_t); E_0[theta_T] = E_0[T] * (2P[b=1] - 1)."""
    return _imbalance_bars(ticks, weighted=False, span=span, warmup=warmup,
                           initial_expected_len=initial_expected_len)


def dollar_imbalance_bars(ticks: pd.DataFrame, span: int = 10, warmup: int = 3,
                          initial_expected_len: float = 50.0) -> pd.DataFrame:
    """DIBs: theta_T = sum(b_t * dollar_t); E_0[theta_T] = E_0[T] * (2v+ - E_0[v_t])."""
    weighted_ticks = ticks.copy()
    weighted_ticks["volume"] = weighted_ticks["price"] * weighted_ticks["volume"]
    return _imbalance_bars(weighted_ticks, weighted=True, span=span, warmup=warmup,
                           initial_expected_len=initial_expected_len)


def volume_imbalance_bars(ticks: pd.DataFrame, span: int = 10, warmup: int = 3,
                          initial_expected_len: float = 50.0) -> pd.DataFrame:
    """VIBs: theta_T = sum(b_t * v_t) with v_t in shares/contracts."""
    return _imbalance_bars(ticks, weighted=True, span=span, warmup=warmup,
                           initial_expected_len=initial_expected_len)
