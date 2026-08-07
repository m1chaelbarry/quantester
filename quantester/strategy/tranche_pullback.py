"""Tranche pullback strategy: volatility-spaced dip-buying ladder.

Verification status: not covered by the notebook — implemented from the user's
strategy specification (three-tranche pullback ladder with latching state
machine, regime filter, SMA exit and ATR stop). ATR uses Wilder's definition
(shared with `visualization.indicators.atr`, Wilder 1978).

Mathematical model (all levels in price terms, computed from closes):

- Regime filter: long entries only while close_t > SMA_200(t); a "Flat" regime
  prohibits arming new ladders (no falling-knife entries in bear regimes).
- Anchor peak: rolling 20-period peak close P_peak = max(close_{t-19..t})
  (rolling(20) convention: current bar plus 19 prior bars).
- Volatility spacing: tranche thresholds T_k = P_peak - k * 1.5 * ATR_14 for
  k in {1, 2, 3}; spacing widens with volatility.
- Capital mapping: q_k = A_t * fraction_k / T_k with fractions
  (0.25, 0.35, 0.40) of equity A_t at latch time. The strategy emits each
  fraction on SignalEvent.strength with limit_price=T_k; the portfolio sizes
  the target AT the limit price, so wire the PortfolioManager with
  PercentEquitySizer(1.0) for exact spec mapping (pct scales total deployment).
- Latching state machine: while no tranche is filled the ladder RE-ANCHORS
  every bar to the current peak/ATR (cancel + replace), so the system always
  buys a dip from the current peak rather than a stale one. The moment the
  first tranche fills, the live levels FREEZE: the latched P_peak and ATR_14
  govern the remaining tranche entries, the exit and the stop until the
  position is completely closed (the spec's Flat -> Active transition is read
  as flat -> position-open; a ladder that never fills must refresh, otherwise
  a runaway market strands the capital at a dead anchor forever).
- Mean-reversion exit: once any tranche is filled, close all tranches (market,
  next bar's open) when close_t >= SMA_5(t).
- Hard stop: once any tranche is filled, close all tranches when
  close_t <= P_peak - 5.0 * ATR_14. The trigger is close-based (per the spec's
  "Instant Unwind" wording) and fills at the next bar's open — gap risk is
  honored, the stop price is never guaranteed (Cross-Ref-2 section 4.2).

Execution mechanics under the temporal firewall (delay=1):

- At placement (close of bar T) three LONG signals carrying limit prices
  become resting LIMIT orders eligible from bar T+1; the execution ledger
  fills tranche k when a later bar's low touches T_k (min(open, T_k) on
  gaps). While unfilled the ladder is canceled and re-anchored at each
  close; the first fill freezes the levels.
- The strategy mirrors the ledger's fill condition (bar low <= T_k) at each
  close to track which tranches are live; the portfolio holds actual sizes.
- Exits emit EXIT with cancel_orders=True: the portfolio flattens the
  position AND purges every resting order for the symbol, so unfilled
  tranches can never re-enter after an exit.
- After an exit the machine cools down for exactly one bar with data (the bar
  whose open fills the exit) before it may re-arm; this keeps target-vs-
  position deltas consistent while the exit order is in flight.

Risk safeguards are deliberately NOT baked into the strategy: the 2x spread +
fee friction is a CostModel (`execution.costs.ConservativeFrictionCostModel`)
and the 4.5% daily drawdown circuit breaker is a portfolio overlay
(`portfolio.risk.DailyDrawdownBreaker`). A breaker liquidation purges the
resting ladder and suspends ALL signal flow until the daily rollover; the
state machine then self-heals at the next SMA_5 cross or stop touch
(conservative post-breaker cooldown).

Live CCXT mapping: temporal firewall = state check before acting on the 1m
stream; PercentEquitySizer = the 25/35/40 portfolio sizer; resting LIMIT
orders = post-only create_order calls (backtest charges taker-grade friction
on them anyway, which is conservative); EXIT/cancel = market close plus
cancel-all on Binance/ByBit; DailyDrawdownBreaker = the daily-loss kill
switch with rollover reset.
"""

from __future__ import annotations

import numpy as np

from ..events import EXIT, LONG, SignalEvent
from ..visualization.indicators import atr as wilder_atr
from .base import Strategy

FLAT = "flat"
ACTIVE = "active"
EXITING = "exiting"


class TranchePullbackStrategy(Strategy):
    """Latched three-tranche pullback ladder on a single symbol; delay=1."""

    def __init__(self, data_handler, symbol: str, regime_window: int = 200,
                 peak_window: int = 20, atr_window: int = 14,
                 atr_spacing: float = 1.5,
                 tranche_fractions: tuple = (0.25, 0.35, 0.40),
                 exit_window: int = 5, stop_atr_mult: float = 5.0):
        if not tranche_fractions or any(f <= 0 for f in tranche_fractions):
            raise ValueError("tranche_fractions must be positive")
        if abs(sum(tranche_fractions) - 1.0) > 1e-9:
            raise ValueError("tranche_fractions must sum to 1.0")
        if atr_spacing <= 0 or stop_atr_mult <= atr_spacing * len(tranche_fractions):
            raise ValueError("stop must sit wider than the deepest tranche")
        self.data_handler = data_handler
        self.symbol = symbol
        self.regime_window = int(regime_window)
        self.peak_window = int(peak_window)
        self.atr_window = int(atr_window)
        self.atr_spacing = float(atr_spacing)
        self.tranche_fractions = tuple(float(f) for f in tranche_fractions)
        self.exit_window = int(exit_window)
        self.stop_atr_mult = float(stop_atr_mult)
        self.delay = 1  # signals at close T, orders live from bar T+1

        self._history = max(
            self.regime_window, self.peak_window, self.atr_window + 1,
            self.exit_window,
        )
        self._state = FLAT
        self._peak = None
        self._atr = None
        self._thresholds: list = []
        self._stop = None
        self._filled: list = []
        self._latched_at = None

    # ------------------------------------------------------------- state I/O

    def _place_ladder(self, timestamp, peak: float, atr_value: float,
                      events_queue, cancel_first: bool = False):
        """(Re)anchor the ladder at the current peak/ATR and rest the tranche
        limits. While hunting (no fills yet) this runs every bar; the freeze
        happens at the first fill. `cancel_first` purges the previously
        resting ladder before replacing it — the cancel rides on the FIRST
        emitted signal only, since each cancel would otherwise purge the
        replacement orders from the signals drained just before it."""
        self._peak = peak
        self._atr = atr_value
        self._thresholds = [
            peak - (k + 1) * self.atr_spacing * atr_value
            for k in range(len(self.tranche_fractions))
        ]
        self._stop = peak - self.stop_atr_mult * atr_value
        self._filled = [False] * len(self.tranche_fractions)
        self._latched_at = timestamp
        self._state = ACTIVE
        pending_cancel = cancel_first
        for fraction, threshold in zip(self.tranche_fractions, self._thresholds):
            if threshold <= 0:
                continue  # degenerate volatility spike: skip untradeable level
            events_queue.put(
                SignalEvent(timestamp, self.symbol, LONG, strength=fraction,
                            delay=self.delay, limit_price=threshold,
                            cancel_orders=pending_cancel)
            )
            pending_cancel = False
        if pending_cancel:
            # Every threshold was degenerate: still purge the stale ladder.
            events_queue.put(
                SignalEvent(timestamp, self.symbol, EXIT, delay=self.delay,
                            cancel_orders=True)
            )

    def _reset(self):
        self._state = FLAT
        self._peak = None
        self._atr = None
        self._thresholds = []
        self._stop = None
        self._filled = []
        self._latched_at = None

    def _mark_fills(self, timestamp, low: float):
        """Mirror the execution ledger: tranche k fills once a bar's low
        touches T_k. Only bars after the latch bar count — resting orders are
        eligible from the next bar onward (earliest_fill_time = T+1)."""
        if self._latched_at is None or timestamp <= self._latched_at:
            return
        for k, threshold in enumerate(self._thresholds):
            if not self._filled[k] and low <= threshold:
                self._filled[k] = True

    # ---------------------------------------------------------------- signals

    def calculate_signals(self, event, events_queue):
        bar = event.bars.get(self.symbol)
        if bar is None:
            return  # availability mask: untradeable at this timestamp
        bars = self.data_handler.get_latest_bars(self.symbol, self._history)
        if len(bars) < self._history:
            return  # warmup: regime SMA not yet defined

        close = bars["close"]
        close_t = float(close.iloc[-1])
        sma_regime = float(close.rolling(self.regime_window).mean().iloc[-1])
        peak = float(close.rolling(self.peak_window).max().iloc[-1])
        atr_t = float(
            wilder_atr(bars["high"], bars["low"], close, self.atr_window).iloc[-1]
        )
        sma_exit = float(close.rolling(self.exit_window).mean().iloc[-1])
        if not np.all(np.isfinite([close_t, sma_regime, peak, atr_t, sma_exit])):
            return

        if self._state == EXITING:
            # The exit market order filled at this bar's open (the bar exists,
            # else we returned above); the machine may re-arm immediately.
            self._reset()

        if self._state == FLAT:
            if close_t > sma_regime:
                self._place_ladder(event.timestamp, peak, atr_t, events_queue)
            return

        # ACTIVE: mirror the ledger's fills against the levels that rested
        # during this bar (placed earlier, live from the next bar onward).
        self._mark_fills(event.timestamp, float(bar["low"]))

        if any(self._filled):
            # Frozen: latched levels govern the remaining tranches, the stop
            # and the exit until the position is completely closed.
            if close_t <= self._stop:
                self._emit_exit(event, events_queue)  # hard stop
            elif close_t >= sma_exit:
                self._emit_exit(event, events_queue)  # mean reversion to SMA_5
            return

        # Hunting: nothing filled yet, so the ladder re-anchors. The regime
        # filter applies to entries, and a resting limit IS a latent entry:
        # losing the bull regime pulls the ladder; keeping it refreshes the
        # anchor to the current peak/ATR.
        if close_t <= sma_regime:
            self._emit_exit(event, events_queue)  # flat: pure book purge
            self._reset()
        else:
            self._place_ladder(event.timestamp, peak, atr_t, events_queue,
                               cancel_first=True)

    def _emit_exit(self, event, events_queue):
        events_queue.put(
            SignalEvent(event.timestamp, self.symbol, EXIT, strength=1.0,
                        delay=self.delay, cancel_orders=True)
        )
        self._state = EXITING

    def vectorized_signals(self, data: dict):
        # The latched state machine (path-dependent levels + tranche fills) has
        # no closed-form vectorized twin; Monte Carlo fast-track validation is
        # unavailable for this strategy by design.
        return super().vectorized_signals(data)
