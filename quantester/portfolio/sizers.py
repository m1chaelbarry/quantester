"""Event-loop position sizers (signal → target quantity).

Separated from ``PortfolioManager`` so sizing policy stays a single
responsibility, and from ``sizing.py`` which holds research math (Kelly,
Vince optimal-f, vol parity) rather than live signal callables.
"""

from __future__ import annotations

import numpy as np

from ..events import EXIT, LONG, SignalEvent


class FixedUnitSizer:
    """Target = +/- units * strength per signal (used for fast-track parity)."""

    def __init__(self, units: float = 100.0):
        units = float(units)
        if units <= 0:
            raise ValueError(
                f"FixedUnitSizer units must be positive (shares per signal). "
                f"Got {units!r}."
            )
        self.units = units

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        return sign * self.units * signal.strength


class PercentEquitySizer:
    """Target quantity worth pct * equity * strength at the reference price."""

    def __init__(self, pct: float = 0.5):
        pct = float(pct)
        if not 0.0 < pct <= 1.0:
            raise ValueError(
                f"PercentEquitySizer pct must be in (0, 1] — "
                f"e.g. 0.9 means 'use 90% of account equity'. Got {pct!r}."
            )
        self.pct = pct

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT or ref_price <= 0:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        dollar_target = portfolio.equity * self.pct * signal.strength
        return sign * dollar_target / ref_price


class FractionalRiskSizer:
    """Vince-style fractional bet: risk a fixed equity fraction to the stop.

    Target quantity q = ± (equity × risk_fraction) / stop_distance, where
    ``signal.stop_distance`` is the protective stop gap in price units
    (e.g. 2 × ATR_14). A full stop-out then loses approximately
    ``risk_fraction`` of account equity before friction.
    """

    def __init__(self, risk_fraction: float = 0.02):
        if not 0.0 < risk_fraction <= 1.0:
            raise ValueError("risk_fraction must lie in (0, 1]")
        self.risk_fraction = float(risk_fraction)

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT or ref_price <= 0:
            return 0.0
        distance = getattr(signal, "stop_distance", None)
        if distance is None or float(distance) <= 0.0:
            raise ValueError(
                "FractionalRiskSizer requires signal.stop_distance > 0 "
                "(price units from entry to the protective stop)."
            )
        sign = 1.0 if signal.signal_type == LONG else -1.0
        return sign * (portfolio.equity * self.risk_fraction) / float(distance)


class HedgeRatioSizer:
    """Cointegrating-residual pairs sizing: the hedge leg tracks the primary.

    The primary leg (``primary_symbol``) is sized by ``base_sizer`` as usual.
    Every other symbol is a HEDGE leg whose target quantity is

        q_X = sign(signal_type) * hedge_ratio * |q_primary|

    so a long-spread entry (LONG Y, SHORT X with hedge_ratio beta) books
    q_X = -beta * q_Y — the cointegrating (unit-ratio) hedge per Kaufman TSM
    / Chan pairs sizing (synthesis §1.13). The hedge leg is NOT sized off its
    own price: independent PercentEquity sizing of both legs breaks the
    cointegrating relationship the spread trades. ``base_sizer`` must be an
    equity-style sizer (PercentEquity/FixedUnit); a stop-distance sizer
    (FractionalRiskSizer) has no stop on the synthetic primary probe and
    will refuse.

    The hedge signal must carry ``hedge_ratio=beta_t`` (e.g. attached by
    PairsTradingStrategy from its rolling OLS fit). The hedge leg's strength
    scales the primary-leg sizing decision (legs are expected to agree).

    Verification status: not covered by the user's quant-literature notebook —
    implemented from Kaufman TSM hedge-ratio pairs sizing and Chan's
    cointegrating-spread construction (3rd-cross-reference synthesis §1.13).
    """

    def __init__(self, primary_symbol: str, base_sizer=None):
        self.primary_symbol = primary_symbol
        self.base_sizer = base_sizer or PercentEquitySizer(0.5)

    def _primary_ref_price(self, signal, portfolio) -> float | None:
        """Firewall-respecting reference price of the primary leg."""
        handler = portfolio.data_handler
        if getattr(signal, "delay", 1) == 0:
            price = handler.get_current_open(self.primary_symbol)
            return None if price is None else float(price)
        bars = handler.get_latest_bars(self.primary_symbol, 1)
        if bars.empty:
            return None  # primary untradeable at this timestamp
        return float(bars["close"].iloc[-1])

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT or ref_price <= 0:
            return 0.0
        if signal.symbol == self.primary_symbol:
            return self.base_sizer(signal, portfolio, ref_price)
        beta = getattr(signal, "hedge_ratio", None)
        if beta is None or not np.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError(
                f"HedgeRatioSizer: hedge leg {signal.symbol!r} requires a "
                "positive signal.hedge_ratio (beta_t from the primary-leg "
                "regression). Wire PairsTradingStrategy or attach beta_t "
                "explicitly."
            )
        primary_ref = self._primary_ref_price(signal, portfolio)
        if primary_ref is None or primary_ref <= 0:
            return 0.0
        primary_signal = SignalEvent(
            signal.timestamp, self.primary_symbol, LONG,
            strength=signal.strength, delay=getattr(signal, "delay", 1),
        )
        q_primary = abs(self.base_sizer(primary_signal, portfolio, primary_ref))
        sign = 1.0 if signal.signal_type == LONG else -1.0
        return sign * float(beta) * q_primary
