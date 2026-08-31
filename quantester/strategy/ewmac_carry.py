"""EWMAC + crypto-carry Combined Forecast (delay=1).

One net ``SignalEvent`` per symbol: trend (EWMAC) and Carry Forecast are
weighted, rescaled by Carver's expanding forecast scalar and
``D_f = 1/sqrt(w'Ωw)``, then capped at ±20. Strength is ``|F|/20``.

Crowded-long (OI 3d growth and daily funding sum) sets ``cap_long_increase``
so the portfolio will not add to a long. DVOL above the threshold halves F.

Missing ``funding_rate`` → Carry Forecast 0 (no settlement is emitted by the
handler). Missing DVOL/OI → that filter is off.

Verification status: Combined Forecast / Carry Forecast sign / Inertia Buffer
/ Funding Settlement / Drawdown De-lever — notebook-verified design intent
from the EWMAC+carry architecture report. Numeric hyperparameters are
research defaults, not product locks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import EXIT, LONG, SHORT, SignalEvent
from .base import Strategy

FORECAST_CAP = 20.0
SCALAR_TARGET = 10.0
_DENOM_FLOOR = 1e-12


def _garman_klass_daily(df: pd.DataFrame, span: int = 20) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    valid = (high > 0) & (low > 0) & (close > 0) & (open_ > 0)
    log_hl = np.log(high.where(valid) / low.where(valid))
    log_co = np.log(close.where(valid) / open_.where(valid))
    var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    return np.sqrt(var.ewm(span=span, min_periods=2).mean())


def _expanding_scalar(raw: pd.Series, target: float = SCALAR_TARGET) -> pd.Series:
    mean_abs = raw.abs().expanding(min_periods=2).mean()
    return target / mean_abs.clip(lower=_DENOM_FLOOR)


def combined_forecast_frame(
    df: pd.DataFrame,
    *,
    fast: int = 16,
    slow: int = 64,
    trend_weight: float = 0.60,
    carry_weight: float = 0.40,
    forecast_cap: float = FORECAST_CAP,
    scalar_target: float = SCALAR_TARGET,
    carry_ema_span: int = 2,
    gk_span: int = 20,
    dvol_threshold: float = 80.0,
    dvol_scale: float = 0.50,
    oi_growth_bars: int = 3,
    oi_growth_threshold: float = 0.25,
    crowding_funding_threshold: float = 0.0015,
) -> pd.DataFrame:
    """Return columns ``forecast`` (capped Combined Forecast) and ``crowded``.

    Expanding statistics use the whole visible history — call this on bars
    from inception through T (firewall already truncated).
    """
    close = df["close"].astype(float)
    gk_daily = _garman_klass_daily(df, span=gk_span)
    sigma_price = (gk_daily * close).replace(0.0, np.nan)
    ema_fast = close.ewm(span=fast, min_periods=slow).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    trend_raw = (ema_fast - ema_slow) / sigma_price
    trend = (_expanding_scalar(trend_raw, scalar_target) * trend_raw).clip(
        -forecast_cap, forecast_cap
    )

    if "funding_rate" in df.columns:
        funding = pd.to_numeric(df["funding_rate"], errors="coerce")
    else:
        funding = pd.Series(0.0, index=df.index)
    smoothed = funding.ewm(span=carry_ema_span, min_periods=1).mean()
    # Desired-position space: positive funding → short the perpetual.
    carry_raw = -smoothed.fillna(0.0)
    carry = (_expanding_scalar(carry_raw, scalar_target) * carry_raw).clip(
        -forecast_cap, forecast_cap
    )

    rho = trend.expanding(min_periods=slow).corr(carry).clip(-0.99, 0.99).fillna(0.0)
    wt, wc = float(trend_weight), float(carry_weight)
    dfac = 1.0 / np.sqrt(wt ** 2 + wc ** 2 + 2.0 * wt * wc * rho)
    combined = (dfac * (wt * trend + wc * carry)).clip(-forecast_cap, forecast_cap)

    if "dvol" in df.columns:
        dvol = pd.to_numeric(df["dvol"], errors="coerce")
        hot = dvol.notna() & (dvol > dvol_threshold)
        combined = combined.where(~hot, combined * dvol_scale)

    crowded = pd.Series(False, index=df.index)
    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce")
        oi_growth = oi.pct_change(oi_growth_bars)
        crowded = (
            oi_growth.notna()
            & funding.notna()
            & (oi_growth > oi_growth_threshold)
            & (funding > crowding_funding_threshold)
        )

    return pd.DataFrame({"forecast": combined, "crowded": crowded})


def combined_forecast_positions(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Vectorized twin: target in [-1, 1] as Combined Forecast / cap."""
    cap = float(kwargs.get("forecast_cap", FORECAST_CAP))
    frame = combined_forecast_frame(df, **kwargs)
    return (frame["forecast"] / cap).fillna(0.0)


class EWMACCarryStrategy(Strategy):
    """Delay-1 Combined Forecast on one perpetual. Emits every close."""

    delay = 1
    fill_at = "open"

    def __init__(
        self,
        data_handler,
        symbol: str,
        fast: int = 16,
        slow: int = 64,
        trend_weight: float = 0.60,
        carry_weight: float = 0.40,
        forecast_cap: float = FORECAST_CAP,
        delay: int = 1,
        **forecast_kwargs,
    ):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError(f"delay must be an integer >= 0; got {delay!r}")
        self.data_handler = data_handler
        self.symbol = symbol
        self.fast = int(fast)
        self.slow = int(slow)
        self.trend_weight = float(trend_weight)
        self.carry_weight = float(carry_weight)
        self.forecast_cap = float(forecast_cap)
        self.delay = delay
        self._forecast_kwargs = dict(
            fast=self.fast,
            slow=self.slow,
            trend_weight=self.trend_weight,
            carry_weight=self.carry_weight,
            forecast_cap=self.forecast_cap,
            **forecast_kwargs,
        )

    def _frame(self, event) -> pd.DataFrame | None:
        if event.bars.get(self.symbol) is None:
            return None
        bars = self.data_handler.get_latest_bars(self.symbol, 10**9)
        if bars is None or len(bars) < self.slow + 2:
            return None
        return bars

    def calculate_signals(self, event, events_queue):
        bars = self._frame(event)
        if bars is None:
            return
        stats = combined_forecast_frame(bars, **self._forecast_kwargs)
        forecast = float(stats["forecast"].iloc[-1])
        crowded = bool(stats["crowded"].iloc[-1])
        if not np.isfinite(forecast) or abs(forecast) < 1e-12:
            events_queue.put(
                SignalEvent(
                    event.timestamp,
                    self.symbol,
                    EXIT,
                    strength=1.0,
                    delay=self.delay,
                    cap_long_increase=crowded,
                )
            )
            return
        signal_type = LONG if forecast > 0 else SHORT
        events_queue.put(
            SignalEvent(
                event.timestamp,
                self.symbol,
                signal_type,
                strength=min(abs(forecast) / self.forecast_cap, 1.0),
                delay=self.delay,
                cap_long_increase=crowded,
            )
        )

    def vectorized_signals(self, data: dict):
        df = data[self.symbol]
        return {self.symbol: combined_forecast_positions(df, **self._forecast_kwargs)}
