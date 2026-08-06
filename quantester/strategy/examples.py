"""Example strategies.

MovingAverageCrossStrategy uses a single shared signal function for both its
event-driven form and its vectorized twin, guaranteeing fast-track parity:
signals are computed on bar T's close and executed at bar T+1's open (delay=1).
"""

from __future__ import annotations

import pandas as pd

from ..events import EXIT, LONG, SHORT, SignalEvent
from .base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Enters a long position in every symbol once, at the first available bar."""

    def __init__(self, data_handler):
        self.data_handler = data_handler
        self._entered = False
        self.delay = 1

    def calculate_signals(self, event, events_queue):
        if self._entered:
            return
        self._entered = True
        for symbol in self.data_handler.symbols:
            if event.bars.get(symbol) is not None:
                events_queue.put(
                    SignalEvent(event.timestamp, symbol, LONG, strength=1.0, delay=self.delay)
                )

    def vectorized_signals(self, data: dict):
        return {s: pd.Series(1.0, index=df.index) for s, df in data.items()}


def crossover_positions(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Target position after each close: 1 long / -1 short / 0 flat.

    Cross detection only changes position on the bar where the fast SMA crosses
    the slow SMA; otherwise the previous position is held.
    """
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    diff = sma_fast - sma_slow
    raw = pd.Series(index=close.index, dtype=float)
    crossed_up = (diff > 0) & (diff.shift(1) <= 0)
    crossed_down = (diff < 0) & (diff.shift(1) >= 0)
    raw[crossed_up] = 1.0
    raw[crossed_down] = -1.0
    valid = sma_slow.notna()
    return raw.where(valid).ffill().fillna(0.0)


class MovingAverageCrossStrategy(Strategy):
    """SMA crossover on one symbol; delay=1 (signal at close T, fill at open T+1)."""

    def __init__(self, data_handler, symbol: str, fast: int = 10, slow: int = 30,
                 direction: str = "both", delay: int = 1):
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.data_handler = data_handler
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.direction = direction  # 'long', 'short', or 'both'
        self.delay = delay
        self._position = 0.0

    def _emit(self, timestamp, events_queue, target: float):
        if target == self._position:
            return
        if target == 0.0:
            signal_type = EXIT
        elif target > 0:
            signal_type = LONG
        else:
            signal_type = SHORT
        events_queue.put(
            SignalEvent(timestamp, self.symbol, signal_type,
                        strength=abs(target), delay=self.delay)
        )
        self._position = target

    def calculate_signals(self, event, events_queue):
        if event.bars.get(self.symbol) is None:
            return
        bars = self.data_handler.get_latest_bars(self.symbol, self.slow + 2)
        if len(bars) < self.slow + 1:
            return
        sma_fast = bars["close"].rolling(self.fast).mean()
        sma_slow = bars["close"].rolling(self.slow).mean()
        diff_now = sma_fast.iloc[-1] - sma_slow.iloc[-1]
        diff_prev = sma_fast.iloc[-2] - sma_slow.iloc[-2]
        if diff_prev <= 0 < diff_now:
            target = 1.0
        elif diff_prev >= 0 > diff_now:
            target = -1.0
        else:
            target = self._position  # no cross at this bar: hold state
        if self.direction == "long" and target < 0:
            target = 0.0
        elif self.direction == "short" and target > 0:
            target = 0.0
        self._emit(event.timestamp, events_queue, float(target))

    def vectorized_signals(self, data: dict):
        close = data[self.symbol]["close"]
        positions = crossover_positions(close, self.fast, self.slow)
        if self.direction == "long":
            positions = positions.clip(lower=0.0)
        elif self.direction == "short":
            positions = positions.clip(upper=0.0)
        return {self.symbol: positions}
