"""Dual-EMA trend follower with Kakushadze Δ protective filter for x2 LETFs.

Verification status:
- Notebook-verified: delay-1 temporal firewall (signal at close T, fill at
  open T+1); Kaufman principle that doubled price volatility requires halved
  position size (wired via ``portfolio.sizing.letf_equity_fraction`` /
  ``PercentEquitySizer(0.5)`` for leverage=2).
- Not covered by the notebook — implemented from the LETF strategy report:
  dual EMA(10/30) long-only golden-cross entries, death-cross exits, and
  Kakushadze's daily close-to-close Δ filter (emergency flat when
  ``P < (1-Δ)·P₁``, default Δ=2% for x2). This Δ stop is distinct from
  ``kakushadze_effective_returns`` (cost-aware edge zeroing before weight
  optimization).

Mathematical model (all levels from closes under the firewall):

- Fast / slow EMAs use the span convention ``α = 2/(span+1)``,
  ``EMA_t = α·P_t + (1-α)·EMA_{t-1}`` (Appel / pandas ``ewm(span, adjust=False)``).
  The report's finite-window weighted sum with discount ``λ`` is the truncated
  EWMA of the same family; the recursive form is the live-standard equivalent.
- Golden cross entry (LONG, delay=1): ``EMA_fast`` crosses from ≤ to >
  ``EMA_slow`` (defaults 10 / 30). On the first post-warm-up bar the book
  also arms if already bullish (so a cross inside the slow-EMA window is not
  missed).
- Death-cross exit (EXIT, delay=1): ``EMA_fast`` crosses from ≥ to <
  ``EMA_slow``.
- Protective Δ exit (EXIT, delay=1): while long, if
  ``close_t < (1 - Δ) · close_{t-1}`` flatten at the next open — evacuates
  before LETF daily rebalancing compounds a shock. Default ``Δ = 0.02`` (x2).
- After a Δ stop the book stays flat until the **next golden cross** (report
  entry rule is a cross, not a level). A death cross while flat is a no-op.
- Long-only by design: LETF x2 shorting is a different product; stay in cash
  outside an active long.

Wire sizing with half equity for x2 so dollar volatility matches an unlevered
book: ``PercentEquitySizer(letf_equity_fraction(1.0, leverage=2.0))``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import EXIT, LONG, SignalEvent
from .base import Strategy


def dual_ema_delta_positions(
    close: pd.Series,
    fast: int = 10,
    slow: int = 30,
    delta: float = 0.02,
) -> pd.Series:
    """Target position after each close: 1 long / 0 flat.

    Pure function shared by the event-driven strategy and its vectorized twin.
    Enters on an EMA golden cross, exits on a death cross or a Kakushadze Δ
    daily filter hit; after a Δ stop, waits for the next golden cross.

    On the first post-warm-up bar, if ``EMA_fast > EMA_slow`` already (cross
    occurred during the slow-EMA window), the book arms long — otherwise an
    early golden cross inside the warm-up would be silently missed.
    """
    if fast >= slow:
        raise ValueError("fast span must be smaller than slow span")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    diff = ema_fast - ema_slow
    bullish = (diff > 0).to_numpy(dtype=bool)
    # Require `slow` bars of history before arming (report window T on slow EMA).
    prior = close.shift(1)
    n = np.arange(len(close))
    ready = ((n >= slow) & prior.notna().to_numpy()).astype(bool)
    cross_up = ((diff > 0) & (diff.shift(1) <= 0)).fillna(False).to_numpy(dtype=bool)
    cross_down = ((diff < 0) & (diff.shift(1) >= 0)).fillna(False).to_numpy(dtype=bool)
    delta_hit = (close < (1.0 - delta) * prior).fillna(False).to_numpy(dtype=bool)

    out = np.zeros(len(close), dtype=float)
    position = 0.0
    seen_ready = False
    for i in range(len(close)):
        if not ready[i]:
            position = 0.0
            out[i] = 0.0
            continue
        if not seen_ready:
            # First tradeable bar: arm if already in a bullish EMA regime.
            seen_ready = True
            position = 1.0 if (bullish[i] and not delta_hit[i]) else 0.0
            out[i] = position
            continue
        if position > 0.0:
            if cross_down[i] or delta_hit[i]:
                position = 0.0
        elif cross_up[i] and not delta_hit[i]:
            position = 1.0
        out[i] = position
    return pd.Series(out, index=close.index, dtype=float)


class LetfDualEmaDeltaStrategy(Strategy):
    """Long-only dual EMA(10/30) cross with Kakushadze Δ filter; delay=1.

    Designed for a single x2 leveraged ETF: trend exposure only after a golden
    cross, cash on death cross or daily Δ shock, half-sized via Kaufman leverage
    scaling at the portfolio layer.
    """

    def __init__(
        self,
        data_handler,
        symbol: str,
        fast: int = 10,
        slow: int = 30,
        delta: float = 0.02,
        delay: int = 1,
    ):
        if fast >= slow:
            raise ValueError("fast span must be smaller than slow span")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.data_handler = data_handler
        self.symbol = symbol
        self.fast = int(fast)
        self.slow = int(slow)
        self.delta = float(delta)
        self.delay = int(delay)
        self._position = 0.0

    def _emit(self, timestamp, events_queue, target: float) -> None:
        if target == self._position:
            return
        signal_type = LONG if target > 0.0 else EXIT
        events_queue.put(
            SignalEvent(
                timestamp,
                self.symbol,
                signal_type,
                strength=1.0,
                delay=self.delay,
            )
        )
        self._position = target

    def calculate_signals(self, event, events_queue):
        if event.bars.get(self.symbol) is None:
            return
        # Full visible history so recursive EMA matches the vectorized twin.
        bars = self.data_handler.get_latest_bars(self.symbol, n=10**9)
        if len(bars) < self.slow + 1:
            return
        close = bars["close"]
        target = float(
            dual_ema_delta_positions(
                close, fast=self.fast, slow=self.slow, delta=self.delta
            ).iloc[-1]
        )
        self._emit(event.timestamp, events_queue, target)

    def vectorized_signals(self, data: dict):
        close = data[self.symbol]["close"]
        return {
            self.symbol: dual_ema_delta_positions(
                close, fast=self.fast, slow=self.slow, delta=self.delta
            )
        }
