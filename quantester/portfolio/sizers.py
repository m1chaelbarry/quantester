"""Event-loop position sizers (signal → target quantity).

Separated from ``PortfolioManager`` so sizing policy stays a single
responsibility, and from ``sizing.py`` which holds research math (Kelly,
Vince optimal-f, vol parity) rather than live signal callables.
"""

from __future__ import annotations

from ..events import EXIT, LONG


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
    """Spread sizer: q_Y from percent-equity, q_X = -β q_Y.

    The dependent leg (Y) uses ``hedge_ratio=1`` and is sized as
    ``pct * equity / P_Y``. The explanatory leg (X) carries ``hedge_ratio=β``
    and ``hedge_ref_price=P_Y`` so ``q_X = sign_X * β * pct * equity / P_Y``.
    Independent per-leg percent-equity sizing is not dollar-neutral on a
    cointegrating residual (synthesis §1.13).

    Not covered by the notebook — implemented from Chan *Quantitative Trading*
    hedge-ratio sizing (q_X = -β q_Y).
    """

    def __init__(self, pct: float = 0.5):
        pct = float(pct)
        if not 0.0 < pct <= 1.0:
            raise ValueError(
                f"HedgeRatioSizer pct must be in (0, 1]. Got {pct!r}."
            )
        self.pct = pct

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT or ref_price <= 0:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        ratio = float(getattr(signal, "hedge_ratio", 1.0) or 1.0)
        px = getattr(signal, "hedge_ref_price", None)
        base_price = float(px) if px is not None and float(px) > 0 else float(ref_price)
        return sign * abs(ratio) * portfolio.equity * self.pct / base_price
