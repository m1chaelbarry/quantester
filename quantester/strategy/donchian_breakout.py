"""Donchian breakout with SMA regime gate, ADX filter, and ATR risk sizing.

Verification status:
- Notebook-verified: Wilder ATR (via `visualization.indicators.atr`); Vince
  fractional bet sizing principle (HPR/TWR/f* family in `portfolio.sizing`);
  delay-1 temporal firewall (signal at close T, fill at open T+1); ETF-trick
  friction booking via `ConservativeFrictionCostModel` (c_t / phi_t split).
- Not covered by the notebook: the specific Donchian(20) + SMA(200) + ADX(14)>25
  entry stack, the 10-period Donchian trailing exit, the SMA(20) mean-reversion
  exit, or the 2×ATR protective floor — implemented from the user's hourly BTC
  trend-following breakout specification. Wilder ADX is the canonical 1978
  definition (same smoothing as ATR) via `visualization.indicators.adx`.

Mathematical model (all levels from closes/highs/lows under the firewall):

- Regime gate: Bullish iff Close_t > SMA_200(t); Bearish iff Close_t < SMA_200(t).
- Entry boundaries (prior 20 bars, excluding t):
    B_up,t   = max(High_{t-1}, ..., High_{t-20})
    B_down,t = min(Low_{t-1},  ..., Low_{t-20})
- Trend strength: enter only when ADX_14(t) > adx_threshold (default 25).
- Long entry at close T: Close_T > B_up,T AND Close_T > SMA_200 AND ADX > 25.
- Short entry at close T: Close_T < B_down,T AND Close_T < SMA_200 AND ADX > 25.
- Execution: delay=1 market fill at the open of bar T+1.
- Protective floor: latched at the fill-bar open ∓ stop_atr_mult × ATR_14
  (ATR taken from the signal bar). Triggered when the bar's high/low touches
  the stop at close; execution is delay=1 at the **next bar's open** (OHLC
  backtests must not fill the same bar's close after observing that bar's
  extremes — that hybrid is not live-tradable without an intrabar event).
- Trailing stop: opposite 10-period Donchian boundary from prior bars
  (long: min Low_{t-1..t-10}; short: max High_{t-1..t-10}); close breach exits
  at the next open (delay=1).
- Mean-reversion exit: Close_t crosses back through SMA_20 → EXIT delay=1.
- Fractional sizing: q = (E × risk_fraction) / (stop_atr_mult × ATR_14) via
  `FractionalRiskSizer` reading `SignalEvent.stop_distance`. For multi-coin
  books set risk_fraction = book_budget / N so concurrent names cannot stack
  full per-name risk. `long_only=True` disables short entries.

State machine (FLAT / ENTERING_* / LONG / SHORT / EXITING) mirrors the
execution ledger: ENTERING latches the protective stop from the fill bar's
open once that bar exists; EXITING cools for one bar with data so in-flight
exits settle before re-arming.

Risk safeguards deliberately live outside the strategy: wire
`ConservativeFrictionCostModel` for the 2×(half-spread + fee) friction and
optionally `DailyDrawdownBreaker` as a portfolio overlay.
"""

from __future__ import annotations

import numpy as np

from ..events import EXIT, LONG, OPEN, SHORT, SignalEvent
from ..visualization.indicators import adx as wilder_adx
from ..visualization.indicators import atr as wilder_atr
from ..visualization.indicators import donchian
from .base import Strategy

FLAT = "flat"
ENTERING_LONG = "entering_long"
ENTERING_SHORT = "entering_short"
LONG_STATE = "long"
SHORT_STATE = "short"
EXITING = "exiting"


class DonchianBreakoutStrategy(Strategy):
    """SMA-gated Donchian breakout with ADX filter; delay=1.

    Set ``long_only=True`` to suppress short entries (recommended for BTC
    and multi-coin daily sleeves).
    """

    def __init__(
        self,
        data_handler,
        symbol: str,
        regime_window: int = 200,
        entry_window: int = 20,
        trail_window: int = 10,
        exit_window: int = 20,
        atr_window: int = 14,
        adx_window: int = 14,
        adx_threshold: float = 25.0,
        stop_atr_mult: float = 2.0,
        risk_fraction: float = 0.02,
        long_only: bool = False,
    ):
        if regime_window < 2:
            raise ValueError("regime_window must be >= 2")
        if entry_window < 1 or trail_window < 1 or exit_window < 1:
            raise ValueError("channel/exit windows must be >= 1")
        if atr_window < 1 or adx_window < 1:
            raise ValueError("atr/adx windows must be >= 1")
        if adx_threshold < 0:
            raise ValueError("adx_threshold must be non-negative")
        if stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be positive")
        if not 0.0 < risk_fraction <= 1.0:
            raise ValueError("risk_fraction must lie in (0, 1]")

        self.data_handler = data_handler
        self.symbol = symbol
        self.regime_window = int(regime_window)
        self.entry_window = int(entry_window)
        self.trail_window = int(trail_window)
        self.exit_window = int(exit_window)
        self.atr_window = int(atr_window)
        self.adx_window = int(adx_window)
        self.adx_threshold = float(adx_threshold)
        self.stop_atr_mult = float(stop_atr_mult)
        self.risk_fraction = float(risk_fraction)
        self.long_only = bool(long_only)
        self.delay = 1

        # ADX needs two Wilder passes (~2× window) after the first TR bar.
        self._history = max(
            self.regime_window,
            self.entry_window + 1,
            self.trail_window + 1,
            self.exit_window,
            2 * self.adx_window + 2,
            self.atr_window + 2,
        )
        self._state = FLAT
        self._signal_atr: float | None = None
        self._protective_stop: float | None = None

    # ------------------------------------------------------------- state I/O

    def _reset(self):
        self._state = FLAT
        self._signal_atr = None
        self._protective_stop = None

    def _emit_entry(self, timestamp, events_queue, side: str, atr_value: float):
        stop_distance = self.stop_atr_mult * atr_value
        if stop_distance <= 0 or not np.isfinite(stop_distance):
            return
        signal_type = LONG if side == "long" else SHORT
        events_queue.put(
            SignalEvent(
                timestamp,
                self.symbol,
                signal_type,
                strength=1.0,
                delay=self.delay,
                stop_distance=stop_distance,
            )
        )
        self._signal_atr = float(atr_value)
        self._protective_stop = None
        self._state = ENTERING_LONG if side == "long" else ENTERING_SHORT

    def _emit_exit(self, event, events_queue, fill_at=OPEN):
        events_queue.put(
            SignalEvent(
                event.timestamp,
                self.symbol,
                EXIT,
                strength=1.0,
                delay=self.delay,
                fill_at=fill_at,
            )
        )
        self._state = EXITING

    def _latch_protective(self, entry_price: float):
        atr_value = self._signal_atr
        if atr_value is None or not np.isfinite(entry_price):
            return
        gap = self.stop_atr_mult * atr_value
        if self._state == ENTERING_LONG:
            self._protective_stop = entry_price - gap
            self._state = LONG_STATE
        elif self._state == ENTERING_SHORT:
            self._protective_stop = entry_price + gap
            self._state = SHORT_STATE

    # ---------------------------------------------------------------- signals

    def calculate_signals(self, event, events_queue):
        bar = event.bars.get(self.symbol)
        if bar is None:
            return  # availability mask: untradeable at this timestamp
        bars = self.data_handler.get_latest_bars(self.symbol, self._history)
        if len(bars) < self._history:
            return  # warmup: regime SMA / ADX not yet defined

        high = bars["high"]
        low = bars["low"]
        close = bars["close"]
        close_t = float(close.iloc[-1])
        high_t = float(bar["high"])
        low_t = float(bar["low"])
        open_t = float(bar["open"])

        sma_regime = float(close.rolling(self.regime_window).mean().iloc[-1])
        sma_exit = float(close.rolling(self.exit_window).mean().iloc[-1])
        channel = donchian(high, low, self.entry_window, shift=1)
        upper = float(channel["upper"].iloc[-1])
        lower = float(channel["lower"].iloc[-1])
        trail = donchian(high, low, self.trail_window, shift=1)
        trail_long = float(trail["lower"].iloc[-1])
        trail_short = float(trail["upper"].iloc[-1])
        atr_t = float(wilder_atr(high, low, close, self.atr_window).iloc[-1])
        adx_t = float(wilder_adx(high, low, close, self.adx_window)["adx"].iloc[-1])

        levels = [
            close_t, high_t, low_t, open_t, sma_regime, sma_exit,
            upper, lower, trail_long, trail_short, atr_t, adx_t,
        ]
        if not np.all(np.isfinite(levels)):
            return

        if self._state == EXITING:
            # Exit market order filled at this bar's open (or MOC on the prior
            # bar); cool one bar with data before re-arming.
            self._reset()

        if self._state in (ENTERING_LONG, ENTERING_SHORT):
            # Delay-1 fill reference is this bar's open.
            self._latch_protective(open_t)

        if self._state == FLAT:
            strong = adx_t > self.adx_threshold
            if strong and close_t > upper and close_t > sma_regime:
                self._emit_entry(event.timestamp, events_queue, "long", atr_t)
            elif (
                not self.long_only
                and strong
                and close_t < lower
                and close_t < sma_regime
            ):
                self._emit_entry(event.timestamp, events_queue, "short", atr_t)
            return

        if self._state == LONG_STATE:
            protective = self._protective_stop
            if protective is not None and low_t <= protective:
                # Stop touch observed at close → fill next open (delay=1).
                self._emit_exit(event, events_queue)
            elif close_t < sma_exit or close_t < trail_long:
                self._emit_exit(event, events_queue)
            return

        if self._state == SHORT_STATE:
            protective = self._protective_stop
            if protective is not None and high_t >= protective:
                self._emit_exit(event, events_queue)
            elif close_t > sma_exit or close_t > trail_short:
                self._emit_exit(event, events_queue)

    def vectorized_signals(self, data: dict):
        # Protective-stop latching + path-dependent exits have no closed-form
        # twin; Monte Carlo fast-track validation is unavailable by design.
        return super().vectorized_signals(data)
