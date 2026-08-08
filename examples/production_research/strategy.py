"""Volatility-scaled time-series momentum (delay=1) with a vectorized twin.

Hypothesis
----------
Short-horizon return persistence is exploitable by a close-to-close momentum
rule sized inversely to realized volatility (Carver-style risk targeting).
Signals fire on bar T's close; orders fill at bar T+1's open (``delay=1``).

The event form and ``vectorized_signals`` share ``momentum_positions`` so the
Monte Carlo fast-track is a true twin of the event engine (parity-tested in
``run.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantester.events import EXIT, LONG, SHORT, SignalEvent
from quantester.strategy.base import Strategy


def momentum_positions(
    close: pd.Series,
    lookback: int,
    vol_lookback: int = 20,
    target_vol: float = 0.15,
    allow_short: bool = True,
) -> pd.Series:
    """Target position after each close in [-1, 0, +1] * vol_scale.

    ``vol_scale`` = clip(target_vol / realized_vol, 0, 2) so stronger sizing
    only when recent vol is below the target — never a free leverage dial.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    mom = close / close.shift(lookback) - 1.0
    # Realized vol from simple returns; annualize with sqrt(252).
    rets = close.pct_change()
    realized = rets.rolling(vol_lookback).std() * np.sqrt(252.0)
    scale = (target_vol / realized.replace(0.0, np.nan)).clip(0.0, 2.0).fillna(0.0)
    raw = pd.Series(0.0, index=close.index)
    raw[mom > 0.0] = 1.0
    if allow_short:
        raw[mom < 0.0] = -1.0
    # Warmup: no position until both momentum and vol windows are live.
    valid = mom.notna() & realized.notna()
    return (raw * scale).where(valid, 0.0)


class TrendMomentumStrategy(Strategy):
    """Event-driven twin of ``momentum_positions``; delay=1 market fills."""

    def __init__(
        self,
        data_handler,
        symbol: str,
        lookback: int = 40,
        vol_lookback: int = 20,
        target_vol: float = 0.15,
        allow_short: bool = True,
    ):
        self.data_handler = data_handler
        self.symbol = symbol
        self.lookback = int(lookback)
        self.vol_lookback = int(vol_lookback)
        self.target_vol = float(target_vol)
        self.allow_short = bool(allow_short)
        self.delay = 1
        self._position = 0.0

    def _needed(self) -> int:
        # Match vectorized warmup: mom uses shift(lookback), vol uses
        # rolling(vol_lookback) — first valid bar is max(lookback, vol_lookback).
        return max(self.lookback, self.vol_lookback) + 1

    def calculate_signals(self, event, events_queue):
        if event.bars.get(self.symbol) is None:
            return
        bars = self.data_handler.get_latest_bars(self.symbol, self._needed())
        if len(bars) < self._needed():
            return
        target = float(
            momentum_positions(
                bars["close"],
                lookback=self.lookback,
                vol_lookback=self.vol_lookback,
                target_vol=self.target_vol,
                allow_short=self.allow_short,
            ).iloc[-1]
        )
        if abs(target - self._position) < 1e-12:
            return
        if abs(target) < 1e-12:
            signal_type = EXIT
            strength = 1.0
        elif target > 0:
            signal_type = LONG
            strength = abs(target)
        else:
            signal_type = SHORT
            strength = abs(target)
        events_queue.put(
            SignalEvent(
                event.timestamp,
                self.symbol,
                signal_type,
                strength=strength,
                delay=self.delay,
            )
        )
        self._position = target

    def vectorized_signals(self, data: dict):
        close = data[self.symbol]["close"]
        return {
            self.symbol: momentum_positions(
                close,
                lookback=self.lookback,
                vol_lookback=self.vol_lookback,
                target_vol=self.target_vol,
                allow_short=self.allow_short,
            )
        }
