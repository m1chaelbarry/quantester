"""The strategy under study: volatility-scaled time-series momentum.

If you are new to Quantester, read this in order:

1. ``momentum_positions`` — pure function: closes → target positions.
2. ``TrendMomentumStrategy`` — event form that calls (1) bar-by-bar.
3. ``vectorized_signals`` — bulk form of (1) for Monte Carlo / grids.

Why share one pure function?
    The event engine and the fast-track *must* produce the same positions.
    Parity is checked in ``run.py`` stage [3]. If they diverge, MCPT lies.

Hypothesis (what we claim before touching data)
-----------------------------------------------
Short-horizon return persistence is exploitable by a close-to-close momentum
rule sized inversely to realized volatility (Carver-style risk targeting).

Timing contract
---------------
``delay=1``: signal fires on bar T's **close**; the fill happens at bar T+1's
**open**. Strategies never peek at T's close to fill at T's open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantester.events import EXIT, LONG, SHORT, SignalEvent
from quantester.strategy.base import Strategy


# ===========================================================================
# PURE SIGNAL FUNCTION — shared by event form and fast-track
# ===========================================================================
def momentum_positions(
    close: pd.Series,
    lookback: int,
    vol_lookback: int = 20,
    target_vol: float = 0.15,
    allow_short: bool = True,
) -> pd.Series:
    """Map closes → target position in roughly [-2, +2] (signed × vol scale).

    Steps
    -----
    1. Momentum = close / close[t-lookback] − 1.
    2. Realized vol = rolling std of daily returns, annualized (√252).
    3. Scale = clip(target_vol / realized_vol, 0, 2) — size up when calm,
       size down when wild. Cap at 2× so vol targeting is not free leverage.
    4. Sign: +1 if mom > 0, −1 if mom < 0 (when shorts allowed), else 0.
    5. Warmup: stay flat until both momentum and vol windows are live.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    # 1) Directional edge: recent return over ``lookback`` bars.
    mom = close / close.shift(lookback) - 1.0

    # 2) Risk: annualized realized volatility.
    rets = close.pct_change()
    realized = rets.rolling(vol_lookback).std() * np.sqrt(252.0)

    # 3) Vol targeting (Carver-style). Replace 0-vol with NaN so scale → 0.
    scale = (target_vol / realized.replace(0.0, np.nan)).clip(0.0, 2.0).fillna(0.0)

    # 4) Signed unit position, then scale.
    raw = pd.Series(0.0, index=close.index)
    raw[mom > 0.0] = 1.0
    if allow_short:
        raw[mom < 0.0] = -1.0

    # 5) No trade until indicators are defined.
    valid = mom.notna() & realized.notna()
    return (raw * scale).where(valid, 0.0)


# ===========================================================================
# EVENT-DRIVEN STRATEGY — what the BacktestEngine calls each bar
# ===========================================================================
class TrendMomentumStrategy(Strategy):
    """Event twin of ``momentum_positions``. Emits signals only on *changes*.

    Required Strategy contract
    --------------------------
    - Subclass ``Strategy``.
    - Set ``self.delay`` (here: 1).
    - Implement ``calculate_signals(event, events_queue)`` → put ``SignalEvent``s.
    - Optionally implement ``vectorized_signals(data)`` for MCPT / grids.
    - Read prices **only** via ``data_handler.get_latest_bars`` — never raw frames.
    - Check ``event.bars.get(symbol)`` — ``None`` means untradeable this bar.
    """

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
        # delay=1: safest default — signal at close T, fill at open T+1.
        self.delay = 1
        # Remember last emitted target so we only fire on changes.
        self._position = 0.0

    def _needed(self) -> int:
        """Bars required before the first valid signal (matches vectorized warmup)."""
        # mom uses shift(lookback); vol uses rolling(vol_lookback).
        return max(self.lookback, self.vol_lookback) + 1

    def calculate_signals(self, event, events_queue):
        # Availability mask: missing bar ⇒ do nothing this timestamp.
        if event.bars.get(self.symbol) is None:
            return

        # Only ask the DataHandler — never reach into a raw DataFrame.
        bars = self.data_handler.get_latest_bars(self.symbol, self._needed())
        if len(bars) < self._needed():
            return  # still warming up

        # Same pure function the fast-track uses → parity is possible.
        target = float(
            momentum_positions(
                bars["close"],
                lookback=self.lookback,
                vol_lookback=self.vol_lookback,
                target_vol=self.target_vol,
                allow_short=self.allow_short,
            ).iloc[-1]
        )

        # Emit only when the target actually changed.
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
        """Bulk positions for ``fast_backtest`` / MCPT. Must match event form."""
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
