"""Performance analytics with complete mathematical rigor (Report 1 section 2.5).

- Annualized Sharpe from SIMPLE returns (notebook D1, ruling ticket 05):
  SR = (mean(r) - Rf) / std(r) * sqrt(periods_per_year) with
  r_t = E_t/E_{t-1} - 1, so Carver cost drag stays linear in Sharpe units.
  ``log_returns`` remains available for Masters-style IID resampling
  (documented MCPT exception), but it is not the tearsheet default.
- Max drawdown and duration: continuous peak-to-trough capital decay + recovery time
- Calmar ratio: annualized return / |max drawdown|
- Carver transaction-cost drag (notebook-verified): drag in SR units =
  annual round-trip turnover x standardized instrument cost (in SR), with
  Carver's 0.08 SR/yr "speed limit" surfaced as a warning threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # fallback periods-per-year when the index cannot speak
SPEED_LIMIT_SR = 0.08  # Carver's annual cost-drag speed limit for allocators
_DAYS_PER_YEAR = 365.25


def measured_periods_per_year(index) -> float:
    """Measured bar frequency ``N_T = N / span_in_years`` (Chan ch. 3).

    ``N`` is the observation count; the span is ``t_N - t_0`` in units of
    365.25-day years. This is the series' own calendar frequency — not
    Carver's 256 convenience (a daily approximation so ``sqrt(P) == 16``),
    and not a median-Δt smear that stretches over irregular calendars
    (synthesis §4.2; ruling D2, ticket 20).

    Raises ``TypeError`` on non-datetime indexes and ``ValueError`` when fewer
    than two observations (or a zero span) leave nothing to measure.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            "measured_periods_per_year needs a DatetimeIndex — pass "
            "periods_per_year explicitly for non-datetime indexes."
        )
    n = len(index)
    if n < 2:
        raise ValueError("need >= 2 observations to measure periods per year")
    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    if span_days <= 0:
        raise ValueError("index span is zero — nothing to measure")
    return float(n / (span_days / _DAYS_PER_YEAR))


def _resolve_periods_per_year(index, periods_per_year) -> float:
    """Explicit override wins; datetime indexes are measured; else TRADING_DAYS."""
    if periods_per_year is not None:
        return float(periods_per_year)
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        try:
            return measured_periods_per_year(index)
        except ValueError:
            pass  # zero span: fall through to the fallback
    return float(TRADING_DAYS)


def log_returns(equity: pd.Series) -> pd.Series:
    """Log returns — Masters MCPT/resampling exception path, NOT the tearsheet
    default (D1: the canonical tearsheet Sharpe is simple-return based)."""
    equity = equity.dropna()
    return np.log(equity / equity.shift(1)).dropna()


def simple_returns(equity: pd.Series) -> pd.Series:
    """Simple returns ``E_t/E_{t-1} - 1`` — the canonical tearsheet path (D1)."""
    equity = equity.dropna()
    return equity.pct_change().dropna()


def annualized_sharpe(equity: pd.Series, risk_free_daily: float = 0.0,
                      periods_per_year: float | None = None) -> float:
    """Simple-return Sharpe annualized by ``sqrt(periods_per_year)`` (D1,
    notebook-verified: Carver *Systematic Trading* ch. 12/15 — cost drag is
    linear in simple-return Sharpe units).

    ``periods_per_year=None`` (default) MEASURES the bar calendar from the
    equity index (D2: ``measured_periods_per_year``) — hourly/24-7 crypto
    series are never silently annualized on 252. Pass an explicit value to
    override (e.g. 256 for Carver-style research); non-datetime indexes fall
    back to ``TRADING_DAYS``.
    """
    ppy = _resolve_periods_per_year(equity.index, periods_per_year)
    rets = simple_returns(equity)
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float((rets.mean() - risk_free_daily) / rets.std() * np.sqrt(ppy))


def max_drawdown(equity: pd.Series) -> dict:
    """Peak-to-trough capital decay, plus duration to recover the high watermark."""
    equity = equity.dropna()
    if equity.empty:
        return {"max_drawdown": 0.0, "duration": 0, "peak": None, "trough": None}
    hwm = equity.cummax()
    dd = equity / hwm - 1.0
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    after = equity.loc[trough:]
    recovered = after[after >= equity.loc[peak]]
    if len(recovered):
        duration = int((recovered.index[0] - peak).days)
    else:
        duration = int((equity.index[-1] - peak).days)
    return {
        "max_drawdown": float(dd.min()),
        "duration": duration,
        "peak": peak,
        "trough": trough,
    }


def drawdown_series(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    return equity / equity.cummax() - 1.0


def calmar_ratio(equity: pd.Series,
                 periods_per_year: float | None = None) -> float:
    """Annualized return / |max drawdown|; years = bars / periods_per_year.

    ``periods_per_year=None`` measures the bar calendar from the index (D2),
    matching ``annualized_sharpe``; explicit override wins.
    """
    ppy = _resolve_periods_per_year(equity.index, periods_per_year)
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    years = max(len(equity) / ppy, 1e-12)
    annualized = (float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(equity)["max_drawdown"])
    if mdd == 0:
        return np.inf if annualized > 0 else 0.0
    return float(annualized / mdd)


def carver_cost_drag_sr(annual_turnover: float, standardized_cost_sr: float) -> float:
    """Cost drag in Sharpe-ratio units = turnover (round trips/yr) x cost (SR)."""
    return float(annual_turnover * standardized_cost_sr)


def speed_limit_warning(drag_sr: float) -> str | None:
    if drag_sr > SPEED_LIMIT_SR:
        return (
            f"Cost drag {drag_sr:.3f} SR/yr exceeds Carver's speed limit "
            f"of {SPEED_LIMIT_SR} SR/yr: turnover is consuming the edge."
        )
    return None


def summarize(equity: pd.Series, risk_free_daily: float = 0.0,
              periods_per_year: float | None = None) -> dict:
    """Headline stats. ``periods_per_year=None`` measures N_T from a datetime
    index (D2); pass an explicit value to override."""
    equity = equity.dropna()
    ppy = _resolve_periods_per_year(equity.index, periods_per_year)
    mdd = max_drawdown(equity)
    total_return = (
        float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    )
    return {
        "total_return": total_return,
        "sharpe": annualized_sharpe(equity, risk_free_daily, ppy),
        "max_drawdown": mdd["max_drawdown"],
        "max_drawdown_duration_days": mdd["duration"],
        "calmar": calmar_ratio(equity, ppy),
    }
