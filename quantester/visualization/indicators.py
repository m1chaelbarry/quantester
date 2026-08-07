"""Technical indicators for charting and strategy research.

Verification status: not covered by the notebook — standard TA definitions
implemented from canonical sources: Wilder (1978) for RSI and ATR (Wilder
smoothing, alpha = 1/window), Appel (1979) for MACD, Bollinger for Bollinger
Bands (population standard deviation, ddof=0).

These helpers are pure functions on price series; they are research/display
tooling. Strategies must still compute their own signals through the
DataHandler interface under the temporal firewall — never feed these
full-history series into a live signal path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (span convention, adjust=False)."""
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # All-gain window (avg_loss == 0) pins RSI at 100; all-loss at 0.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram (Appel)."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "signal": signal_line, "histogram": line - signal_line},
        index=close.index,
    )


def bollinger_bands(close: pd.Series, window: int = 20,
                    n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: mid = SMA, bands at +/- n_std * population std."""
    mid = close.rolling(window).mean()
    sigma = close.rolling(window).std(ddof=0)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + n_std * sigma, "lower": mid - n_std * sigma},
        index=close.index,
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def rolling_volatility(close: pd.Series, window: int = 21,
                       periods: int = 252) -> pd.Series:
    """Annualized rolling volatility of log returns."""
    log_rets = np.log(close / close.shift(1))
    return log_rets.rolling(window).std(ddof=1) * np.sqrt(periods)
