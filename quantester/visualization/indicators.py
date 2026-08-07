"""Technical indicators for charting and strategy research.

Verification status: not covered by the notebook — standard TA definitions
implemented from canonical sources: Wilder (1978) for RSI, ATR, and ADX
(Wilder smoothing, alpha = 1/window), Appel (1979) for MACD, Bollinger for
Bollinger Bands (population standard deviation, ddof=0), Donchian for channel
highs/lows.

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


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 14) -> pd.DataFrame:
    """Wilder's Average Directional Index with +DI / -DI.

    Returns a DataFrame with columns ``adx``, ``plus_di``, ``minus_di``.
    Directional movement and true range are Wilder-smoothed (alpha = 1/window);
    DX is then Wilder-smoothed into ADX.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / window
    atr_s = true_range.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    plus_di = 100.0 * (
        plus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr_s
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr_s
    )
    di_sum = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0.0)
    adx_line = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    return pd.DataFrame(
        {"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di},
        index=close.index,
    )


def donchian(high: pd.Series, low: pd.Series, window: int = 20,
             shift: int = 1) -> pd.DataFrame:
    """Donchian channel from prior ``window`` bars (default excludes the current).

    ``shift=1`` yields B_up,t = max(High_{t-1..t-window}) and
    B_down,t = min(Low_{t-1..t-window}), the breakout boundaries used by
    delay-1 channel systems so the signal bar is not in its own channel.
    """
    prior_high = high.shift(shift)
    prior_low = low.shift(shift)
    return pd.DataFrame(
        {
            "upper": prior_high.rolling(window).max(),
            "lower": prior_low.rolling(window).min(),
        },
        index=high.index,
    )


def rolling_volatility(close: pd.Series, window: int = 21,
                       periods: int = 252) -> pd.Series:
    """Annualized rolling volatility of log returns."""
    log_rets = np.log(close / close.shift(1))
    return log_rets.rolling(window).std(ddof=1) * np.sqrt(periods)
