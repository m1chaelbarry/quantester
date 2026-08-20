"""Tranche pullback strategy: volatility-spaced dip-buying ladder.

Verification status:
- Notebook-verified: Wilder ATR (via `quantester.indicators.atr`); the
  Kaufman stop *trigger* on the intra-bar low (TSM ch. 23) including the
  must-reenter-when-trend-intact behavior (the machine re-arms after any exit
  whenever the regime is bullish). Same-bar close MOC after observing the low
  is **not** used here — OHLC backtests lack an intrabar trigger event, so the
  exit is delay=1 at the next open (look-ahead-safe). Ehlers' wide
  catastrophic-stop-only warning and Chan's mean-reversion stop-loss fallacy
  (both motivate the 5x ATR width and keep the SMA_5 exit stop-free).
- Not covered by the notebook: the tranche grid, fractions, regime filter and
  re-anchor-until-first-fill semantics — implemented from the user's
  specification; the re-anchoring variant was forced by real-data testing
  (a one-shot latch traded once in 13 years on 2013-2026 BTC).

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
  every `reanchor_every` bars to the current peak/ATR (cancel + replace), so
  the system always buys a dip from a current peak rather than a stale one.
  Default `reanchor_every=1` matches daily data (refresh every bar). On
  intraday data set it to bars-per-day so the ladder refreshes once per
  calendar day while fills and stops still resolve on every bar. The moment
  the first tranche fills, the live levels FREEZE: the latched P_peak and
  ATR_14 govern the remaining tranche entries, the exit and the stop until
  the position is completely closed (the spec's Flat -> Active transition is
  read as flat -> position-open; a ladder that never fills must refresh,
  otherwise a runaway market strands the capital at a dead anchor forever).
- Mean-reversion exit: once any tranche is filled, close all tranches (market,
  next bar's open) when close_t >= SMA_5(t).
- Hard stop at P_peak - 5.0 * ATR_14. For OHLC backtests the bar's LOW
  touching the latched stop is detected at close and the exit is delay=1 at
  the **next bar's open** (not same-bar MOC: observing the low and filling
  that bar's close is not a live-tradable path without an intrabar trigger
  event). Gap risk through the stop is honored on the next open
  (Cross-Ref-2 section 4.2). The 5x ATR width follows Ehlers' catastrophic-
  stop-only warning; Chan's stop-loss fallacy motivates keeping the SMA_5
  reversion exit stop-free.

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
(`portfolio.risk.DailyDrawdownBreaker`, rolling on the configured session
close per D11). A breaker liquidation purges the
resting ladder and suspends ALL signal flow until the session rollover; the
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

from ..events import EXIT, LONG, OPEN, SignalEvent
from ..indicators import atr as wilder_atr
from .base import Strategy

FLAT = "flat"
ACTIVE = "active"
EXITING = "exiting"


class TranchePullbackStrategy(Strategy):
    """Latched three-tranche pullback ladder on a single symbol; delay=1.

    ``resting_stops=True`` rests a flatten STOP_ORDER after the first fill
    (gap-through). Default ``False`` keeps delay-1 EXIT-on-touch for the
    5×ATR catastrophic stop.
    """

    def __init__(self, data_handler, symbol: str, regime_window: int = 200,
                 peak_window: int = 20, atr_window: int = 14,
                 atr_spacing: float = 1.5,
                 tranche_fractions: tuple = (0.25, 0.35, 0.40),
                 exit_window: int = 5, stop_atr_mult: float = 5.0,
                 reanchor_every: int = 1, cooldown_bars: int = 0,
                 resting_stops: bool = False):
        if not tranche_fractions or any(f <= 0 for f in tranche_fractions):
            raise ValueError("tranche_fractions must be positive")
        if abs(sum(tranche_fractions) - 1.0) > 1e-9:
            raise ValueError("tranche_fractions must sum to 1.0")
        if atr_spacing <= 0 or stop_atr_mult <= atr_spacing * len(tranche_fractions):
            raise ValueError("stop must sit wider than the deepest tranche")
        if reanchor_every < 1:
            raise ValueError("reanchor_every must be >= 1")
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")
        self.data_handler = data_handler
        self.symbol = symbol
        self.regime_window = int(regime_window)
        self.peak_window = int(peak_window)
        self.atr_window = int(atr_window)
        self.atr_spacing = float(atr_spacing)
        self.tranche_fractions = tuple(float(f) for f in tranche_fractions)
        self.exit_window = int(exit_window)
        self.stop_atr_mult = float(stop_atr_mult)
        # Bars between ladder re-anchors while hunting. On daily data leave at
        # 1 (every bar). On intraday data set to bars-per-day so the ladder
        # refreshes once per calendar day while fills/stops still resolve on
        # every bar — the higher-resolution benefit without hourly churn.
        self.reanchor_every = int(reanchor_every)
        # Bars to stay flat after an exit before re-arming. On daily data leave
        # at 0 (next bar = next day). On intraday ports set to bars-per-day - 1
        # so a same-day stop/exit cannot immediately re-enter.
        self.cooldown_bars = int(cooldown_bars)
        self.resting_stops = bool(resting_stops)
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
        self._bars_since_anchor = 0
        self._cooldown_remaining = 0
        self._stop_frac_armed = 0.0

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
        self._bars_since_anchor = 0
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
        self._bars_since_anchor = 0
        # _cooldown_remaining is intentionally preserved across reset
        self._stop_frac_armed = 0.0

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
            # else we returned above); cooldown (if any) then governs re-arm.
            self._reset()

        if self._state == FLAT:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                return
            if close_t > sma_regime:
                self._place_ladder(event.timestamp, peak, atr_t, events_queue)
            return

        # ACTIVE: mirror the ledger's fills against the levels that rested
        # during this bar (placed earlier, live from the next bar onward).
        self._mark_fills(event.timestamp, float(bar["low"]))

        if any(self._filled):
            # Frozen: latched levels govern the remaining tranches, the stop
            # and the exit until the position is completely closed.
            if self.resting_stops:
                filled_frac = sum(
                    frac for frac, on in zip(
                        self.tranche_fractions, self._filled
                    ) if on
                )
                if (
                    filled_frac > 1e-12
                    and self._stop_frac_armed == 0.0
                    and self._stop is not None
                ):
                    events_queue.put(
                        SignalEvent(
                            event.timestamp, self.symbol, LONG,
                            strength=filled_frac, delay=self.delay,
                            stop_price=self._stop, stop_only=True,
                        )
                    )
                    self._stop_frac_armed = filled_frac
            if float(bar["low"]) <= self._stop:
                # Resting stop (if armed) fills this close; EXIT flattens any
                # remainder (later tranche fills) and purges leftover limits.
                self._emit_exit(event, events_queue)
            elif close_t >= sma_exit:
                self._emit_exit(event, events_queue)  # mean reversion to SMA_5
            return

        # Hunting: nothing filled yet. Regime loss pulls the ladder (a resting
        # limit is a latent entry). Otherwise re-anchor every `reanchor_every`
        # bars — on daily data that's every bar; on intraday ports set it to
        # bars-per-day so fills/stops resolve every bar while the ladder only
        # refreshes once per calendar day.
        self._bars_since_anchor += 1
        if close_t <= sma_regime:
            self._emit_exit(event, events_queue)  # flat: pure book purge
            self._reset()
        elif self._bars_since_anchor >= self.reanchor_every:
            self._place_ladder(event.timestamp, peak, atr_t, events_queue,
                               cancel_first=True)

    def _emit_exit(self, event, events_queue, fill_at=OPEN):
        events_queue.put(
            SignalEvent(event.timestamp, self.symbol, EXIT, strength=1.0,
                        delay=self.delay, fill_at=fill_at, cancel_orders=True)
        )
        self._state = EXITING
        self._cooldown_remaining = self.cooldown_bars

    def vectorized_signals(self, data: dict):
        # The latched state machine (path-dependent levels + tranche fills) has
        # no closed-form vectorized twin; Monte Carlo fast-track validation is
        # unavailable for this strategy by design.
        return super().vectorized_signals(data)
