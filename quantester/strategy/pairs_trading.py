"""GLD/GDX spread mean-reversion pairs strategy (diagnostic).

Spread formulation (Ernest Chan, "Quantitative Trading", 2nd ed. ch. 2-3;
rolling hedge-ratio pair trading):

    z_t = ln(P_GLD,t) - beta_t * ln(P_GDX,t) - alpha_t

where (alpha_t, beta_t) come from an OLS regression of ln(P_GLD) on ln(P_GDX)
over a rolling 252-day training window (scikit-learn ``LinearRegression``),
normalized as

    s_t = (z_t - mu_{z,t}) / sigma_{z,t}

with mu_{z,t} / sigma_{z,t} the 20-day rolling mean and sample standard
deviation (ddof=1) of z_t.

Signal protocol (Chan's z-score bands):
- s_t < -entry_z  -> long spread  (LONG GLD, SHORT GDX)
- s_t > +entry_z  -> short spread (SHORT GLD, LONG GDX)
- |s_t| <= exit_z -> flat both legs

Sizing: hedge-leg entry signals carry ``hedge_ratio=beta_t``. Wire the
portfolio with ``HedgeRatioSizer(primary_symbol=leg_y)`` for cointegrating-
residual sizing q_X = -beta_t * q_Y (Kaufman TSM; synthesis §1.13); the
default PercentEquitySizer sizes each leg independently off its own price,
which is NOT dollar-neutral in the residual.

Execution window: ``delay=1`` (default) computes signals at bar T's close and
the engine's State-Based Temporal Firewall fills the resulting orders at bar
T+1's open, so no same-bar close information can leak into a fill.

Synchronization and availability: both legs are signalled in the SAME event
cycle (simultaneous, opposite SignalEvents). If either leg has no bar at T
(availability mask) or the paired training window is incomplete, the strategy
holds its state and emits nothing: gaps pause the strategy rather than
fabricate a spread from stale or one-sided data. Timestamps are never erased.

Verification status: the rolling-OLS pairs construction is not covered by the
user's quant-literature notebook; implemented from Chan (bands, hedge ratio)
and the repo's temporal-firewall contract (de Prado-style look-ahead hygiene).
The event-driven form and the vectorized twin share the same pure helpers
(``ols_spread``, ``zscore_of``, ``next_spread_state``), so parity holds by
construction on synchronized calendars (the GLD/GDX case: both trade on the
same NYSE Arca calendar).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from ..events import EXIT, LONG, SHORT, SignalEvent
from .base import Strategy

FLAT = 0
LONG_SPREAD = 1
SHORT_SPREAD = -1


def ols_spread(log_y: NDArray[np.float64],
               log_x: NDArray[np.float64]) -> tuple[float, float, float]:
    """One causal OLS fit of ``log_y = alpha + beta * log_x`` on a window.

    Returns ``(alpha, beta, z)`` where z is the spread residual at the LAST
    observation of the window. Uses scikit-learn's ``LinearRegression``
    (deterministic least-squares solver), so the event-driven form and the
    vectorized twin produce bit-identical estimates on identical windows.
    """
    x = np.asarray(log_x, dtype=float).reshape(-1, 1)
    y = np.asarray(log_y, dtype=float)
    model = LinearRegression().fit(x, y)
    beta = float(model.coef_[0])
    alpha = float(model.intercept_)
    z = float(y[-1] - beta * x[-1, 0] - alpha)
    return alpha, beta, z


def zscore_of(z_window: NDArray[np.float64]) -> Optional[float]:
    """Z-score of the last element against its trailing window (ddof=1)."""
    window = np.asarray(z_window, dtype=float)
    sigma = float(window.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return None
    return float((window[-1] - window.mean()) / sigma)


def next_spread_state(state: int, s: float, entry_z: float, exit_z: float) -> int:
    """Chan band state machine.

    Flat enters long/short spread beyond +/-entry_z; an open spread exits ONLY
    when |s| <= exit_z (spec-pure: no direct long->short flips, no stop-loss).
    """
    if state == FLAT:
        if s < -entry_z:
            return LONG_SPREAD
        if s > entry_z:
            return SHORT_SPREAD
        return FLAT
    if abs(s) <= exit_z:
        return FLAT
    return state


class PairsTradingStrategy(Strategy):
    """Rolling-OLS GLD/GDX log-spread mean-reversion (delay=1 by default).

    Parameters
    ----------
    data_handler:
        Point-in-time DataHandler; the strategy reads ONLY through it
        (temporal firewall) and never touches raw DataFrames.
    leg_y, leg_x:
        Dependent (GLD) and explanatory (GDX) legs of the regression.
    ols_window:
        Rolling training window for the hedge-ratio regression (252 days).
    zscore_window:
        Rolling normalization window for the spread z-score (20 days).
    entry_z, exit_z:
        Entry and liquidation z-score bands (2.0 / 0.5).
    delay:
        Bars until execution; 1 = signal at close T fills at open T+1.
    min_train_obs:
        Minimum paired observations required to fit the hedge ratio
        (defaults to ``ols_window`` — strict full-window, conservative under
        availability gaps).

    Diagnostics
    -----------
    ``history_`` lists per-bar (timestamp, alpha, beta, z, s, state) so tests
    and audits can introspect the hedge-ratio path and band crossings.
    """

    def __init__(self, data_handler, leg_y: str = "GLD", leg_x: str = "GDX",
                 ols_window: int = 252, zscore_window: int = 20,
                 entry_z: float = 2.0, exit_z: float = 0.5, delay: int = 1,
                 min_train_obs: Optional[int] = None):
        if ols_window < 10:
            raise ValueError("ols_window must be >= 10 for a stable hedge ratio")
        if zscore_window < 2:
            raise ValueError("zscore_window must be >= 2 (sample std)")
        if entry_z <= exit_z:
            raise ValueError("entry_z must exceed exit_z for a coherent band protocol")
        self.data_handler = data_handler
        self.leg_y = leg_y
        self.leg_x = leg_x
        self.ols_window = int(ols_window)
        self.zscore_window = int(zscore_window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.delay = int(delay)
        self.min_train_obs = (
            self.ols_window if min_train_obs is None else int(min_train_obs)
        )
        if not 2 <= self.min_train_obs <= self.ols_window:
            raise ValueError("min_train_obs must lie in [2, ols_window]")

        self._state = FLAT
        self._z: list[float] = []          # causal trailing spread residuals
        self._beta: Optional[float] = None  # latest fitted hedge ratio
        self.history_: list[dict] = []     # diagnostic trail per computed bar

    # ------------------------------------------------------------ computation

    def _paired_log_window(self) -> Optional[tuple[NDArray, NDArray]]:
        """Trailing paired log-closes of both legs, inner-joined on timestamp.

        Availability-mask aware: each leg is queried through the firewall, so
        only data visible at the current phase is returned; a leg missing bars
        inside the window shrinks the paired count and can veto the fit.
        """
        bars_y = self.data_handler.get_latest_bars(self.leg_y, self.ols_window)
        bars_x = self.data_handler.get_latest_bars(self.leg_x, self.ols_window)
        if bars_y.empty or bars_x.empty:
            return None
        joined = pd.concat(
            [bars_y["close"].rename("y"), bars_x["close"].rename("x")],
            axis=1,
            join="inner",
        ).dropna()
        if len(joined) < self.min_train_obs:
            return None
        log_y = np.log(joined["y"].to_numpy(dtype=float))
        log_x = np.log(joined["x"].to_numpy(dtype=float))
        if not (np.isfinite(log_y).all() and np.isfinite(log_x).all()):
            return None  # non-positive prices are invalid input, not a signal
        return log_y, log_x

    def _update_spread(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Advance the causal z/z-score chain one bar; return s_t if defined."""
        window = self._paired_log_window()
        if window is None:
            return None
        alpha, beta, z = ols_spread(*window)
        if not np.isfinite(z):
            return None
        self._beta = beta
        self._z.append(z)
        if len(self._z) < self.zscore_window:
            s = None
        else:
            s = zscore_of(np.asarray(self._z[-self.zscore_window:]))
        self.history_.append(
            {
                "timestamp": timestamp,
                "alpha": alpha,
                "beta": beta,
                "z": z,
                "s": s,
                "state": self._state,
            }
        )
        return s

    # ---------------------------------------------------------------- signals

    def _emit_transition(self, timestamp: pd.Timestamp, events_queue,
                         target: int) -> None:
        """Emit simultaneous, opposite leg signals on a state change only.

        Hedge-leg (leg_x) entries carry ``hedge_ratio=beta_t`` so a
        ``HedgeRatioSizer`` can size q_X = -beta_t * q_Y; sizers that do not
        read the field are unaffected.
        """
        if target == self._state:
            return
        if target == FLAT:
            legs = [(self.leg_y, EXIT), (self.leg_x, EXIT)]
        elif target == LONG_SPREAD:
            legs = [(self.leg_y, LONG), (self.leg_x, SHORT)]
        else:
            legs = [(self.leg_y, SHORT), (self.leg_x, LONG)]
        for symbol, signal_type in legs:
            hedge_ratio = (
                self._beta
                if symbol == self.leg_x and signal_type in (LONG, SHORT)
                else None
            )
            events_queue.put(
                SignalEvent(timestamp, symbol, signal_type,
                            strength=1.0, delay=self.delay,
                            hedge_ratio=hedge_ratio)
            )
        self._state = target

    def calculate_signals(self, event, events_queue) -> None:
        # Multi-symbol synchronization guard: a missing bar on either leg
        # means the spread is untradeable at this timestamp; hold state.
        if event.bars.get(self.leg_y) is None or event.bars.get(self.leg_x) is None:
            return
        s = self._update_spread(event.timestamp)
        if s is None:
            return
        self._emit_transition(
            event.timestamp, events_queue,
            next_spread_state(self._state, s, self.entry_z, self.exit_z),
        )

    # ------------------------------------------------------- vectorized twin

    def vectorized_signals(self, data: dict) -> dict:
        """Full-history target positions per leg for the MC fast-track.

        Uses the identical per-window helpers as the event form; parity is
        exact on synchronized calendars (GLD/GDX). Output is reindexed to the
        union calendar with the state carried across untradeable timestamps.
        """
        closes_y = data[self.leg_y]["close"].astype(float)
        closes_x = data[self.leg_x]["close"].astype(float)
        joined = pd.concat(
            [closes_y.rename("y"), closes_x.rename("x")], axis=1, join="inner"
        ).dropna()
        log_y = np.log(joined["y"].to_numpy(dtype=float))
        log_x = np.log(joined["x"].to_numpy(dtype=float))

        states = np.zeros(len(joined))
        z_hist: list[float] = []
        state = FLAT
        for i in range(len(joined)):
            if i + 1 < self.ols_window:
                continue
            alpha, beta, z = ols_spread(
                log_y[i + 1 - self.ols_window: i + 1],
                log_x[i + 1 - self.ols_window: i + 1],
            )
            if not np.isfinite(z):
                continue
            z_hist.append(z)
            if len(z_hist) < self.zscore_window:
                continue
            s = zscore_of(np.asarray(z_hist[-self.zscore_window:]))
            if s is None:
                continue
            state = next_spread_state(state, s, self.entry_z, self.exit_z)
            states[i] = float(state)

        union = data[self.leg_y].index.union(data[self.leg_x].index)
        leg_y_target = pd.Series(states, index=joined.index).reindex(union).ffill().fillna(0.0)
        return {self.leg_y: leg_y_target, self.leg_x: -leg_y_target}

    @property
    def current_state(self) -> int:
        """Current spread state: FLAT(0) / LONG_SPREAD(+1) / SHORT_SPREAD(-1)."""
        return self._state

    def diagnostics(self) -> pd.DataFrame:
        """Per-bar diagnostic trail: alpha, beta, z, s, and pre-signal state."""
        if not self.history_:
            return pd.DataFrame(
                columns=["alpha", "beta", "z", "s", "state"]
            ).rename_axis("timestamp")
        return pd.DataFrame(self.history_).set_index("timestamp")
