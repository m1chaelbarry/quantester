"""Event-loop position sizers (signal → target quantity).

Separated from ``PortfolioManager`` so sizing policy stays a single
responsibility, and from ``sizing.py`` which holds research math (Kelly,
Vince optimal-f, vol parity) rather than live signal callables.

Sizing base (ruling D10, ticket 26): the live sizers default to
``base="cash"`` — targets scale with available cash, not mark-to-market
equity (MTM sizing is procyclical: equity inflated by a run-up over-sizes
the next entry and the reversal unwinds it, synthesis §1.16). ``base="equity"``
is the explicit procyclical opt-in; ``cash_ewma_span`` optionally smooths the
cash base (Carver-style). Vol-target / Kelly / f* stay research libraries in
``sizing.py`` (D5 KEEP — nothing wires them into the event loop).
"""

from __future__ import annotations

import pandas as pd

from ..events import EXIT, LONG


class _CashOrEquityBase:
    """Shared sizing-base policy for the live sizers (D10).

    ``base="cash"`` (default): size off raw ``portfolio.cash``, or its EWMA
    over signal timestamps when ``cash_ewma_span`` is set. Non-positive cash
    yields a zero target — never a silent fall-back onto equity.
    """

    def __init__(self, base: str, cash_ewma_span: int | None):
        if base not in ("cash", "equity"):
            raise ValueError("base must be 'cash' or 'equity'")
        if cash_ewma_span is not None and cash_ewma_span < 2:
            raise ValueError("cash_ewma_span must be >= 2")
        self.base = base
        self.cash_ewma_span = cash_ewma_span
        self._cash_seen_ts = None
        self._cash_history: list[float] = []

    def _base_value(self, signal, portfolio) -> float:
        if self.base == "equity":
            return float(portfolio.equity)
        cash = float(portfolio.cash)
        if self.cash_ewma_span is None:
            return cash
        ts = getattr(signal, "timestamp", None)
        if (
            ts is not None
            and self._cash_seen_ts is not None
            and ts < self._cash_seen_ts
        ):
            # A reused sizer starting a new run: restart the smoothing window.
            self._cash_history = []
            self._cash_seen_ts = None
        if ts is None or ts != self._cash_seen_ts:
            # One cash observation per signal timestamp (a bar may carry
            # several signals; the EWMA must not see the same cash twice).
            self._cash_history.append(cash)
            self._cash_seen_ts = ts
        return float(
            pd.Series(self._cash_history).ewm(span=self.cash_ewma_span).mean().iloc[-1]
        )


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


class PercentEquitySizer(_CashOrEquityBase):
    """Target quantity worth pct * base * strength at the reference price.

    The default base is CASH (D10): ``pct * portfolio.cash * strength /
    ref_price`` — MTM equity only with the explicit ``base="equity"`` opt-in
    (``cash_ewma_span`` applies to the cash base only). The class name is
    kept for API stability; "percent of equity" is the legacy behavior, not
    the default.
    """

    def __init__(self, pct: float = 0.5, *, base: str = "cash",
                 cash_ewma_span: int | None = None):
        pct = float(pct)
        if not 0.0 < pct <= 1.0:
            raise ValueError(
                f"PercentEquitySizer pct must be in (0, 1] — "
                f"e.g. 0.9 means 'use 90% of the sizing base'. Got {pct!r}."
            )
        _CashOrEquityBase.__init__(self, base, cash_ewma_span)
        self.pct = pct

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT or ref_price <= 0:
            return 0.0
        base_value = self._base_value(signal, portfolio)
        if base_value <= 0:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        dollar_target = base_value * self.pct * signal.strength
        return sign * dollar_target / ref_price


class FractionalRiskSizer(_CashOrEquityBase):
    """Vince-style fractional bet: risk a fixed base fraction to the stop.

    Target quantity q = ± (base × risk_fraction) / stop_distance, where
    ``signal.stop_distance`` is the protective stop gap in price units
    (e.g. 2 × ATR_14). A full stop-out then loses approximately
    ``risk_fraction`` of the sizing base before friction. The default base is
    CASH (D10); ``base="equity"`` is the procyclical MTM opt-in.
    """

    def __init__(self, risk_fraction: float = 0.02, *, base: str = "cash",
                 cash_ewma_span: int | None = None):
        if not 0.0 < risk_fraction <= 1.0:
            raise ValueError("risk_fraction must lie in (0, 1]")
        _CashOrEquityBase.__init__(self, base, cash_ewma_span)
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
        base_value = self._base_value(signal, portfolio)
        if base_value <= 0:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        return sign * (base_value * self.risk_fraction) / float(distance)


class HedgeRatioSizer(_CashOrEquityBase):
    """Spread sizer: q_Y from percent-of-base, q_X = -β q_Y.

    The dependent leg (Y) uses ``hedge_ratio=1`` and is sized as
    ``pct * base / P_Y``. The explanatory leg (X) carries ``hedge_ratio=β``
    and ``hedge_ref_price=P_Y`` so ``q_X = sign_X * β * pct * base / P_Y``.
    Independent per-leg percent sizing is not dollar-neutral on a
    cointegrating residual (synthesis §1.13). The default base is CASH (D10).

    Not covered by the notebook — implemented from Chan *Quantitative Trading*
    hedge-ratio sizing (q_X = -β q_Y).
    """

    def __init__(self, pct: float = 0.5, *, base: str = "cash",
                 cash_ewma_span: int | None = None):
        pct = float(pct)
        if not 0.0 < pct <= 1.0:
            raise ValueError(
                f"HedgeRatioSizer pct must be in (0, 1]. Got {pct!r}."
            )
        _CashOrEquityBase.__init__(self, base, cash_ewma_span)
        self.pct = pct

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT or ref_price <= 0:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        raw = getattr(signal, "hedge_ratio", None)
        ratio = 1.0 if raw is None else float(raw)
        px = getattr(signal, "hedge_ref_price", None)
        base_price = float(px) if px is not None and float(px) > 0 else float(ref_price)
        base_value = self._base_value(signal, portfolio)
        if base_value <= 0:
            return 0.0
        return sign * abs(ratio) * base_value * self.pct / base_price
