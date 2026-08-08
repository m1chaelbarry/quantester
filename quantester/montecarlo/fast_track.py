"""Vectorized execution bypass for Monte Carlo (Cross-Ref section 3.2).

Running 10,000 permuted retraining loops through the pure-Python event queue is
computationally intractable (Cross-Ref section 2.C). The fast-track applies
strategy logic and cost models directly to NumPy/pandas arrays.

PARITY CONTRACT with the event engine (asserted by test):
- target positions are decided on bar T's close and executed at bar T+1's open
- fills use the SAME CostModel adverse adjustments as SimulatedExecutionHandler
- open fills evaluate Kaufman/range costs on the **prior** bar (or a zero-range
  open proxy on the first bar) — never the fill bar's own H/L
- cash_t  = cash_{t-1} - dQ_t * fill_price_t - commission(|dQ_t|)
- equity_t = cash_t + Q_t * close_t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..execution.costs import CostModel


@dataclass
class FastResult:
    equity: pd.Series
    positions: pd.Series
    cash: pd.Series
    daily_returns: pd.Series

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def sharpe(self) -> float:
        rets = self.daily_returns.dropna()
        if len(rets) < 2 or rets.std() == 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252))


def _open_cost_bar(i: int, open_, high, low, close, volume) -> dict:
    """Match SimulatedExecutionHandler open-phase cost proxy (no same-bar H/L)."""
    if i > 0:
        return {
            "open": float(open_[i]),
            "high": float(high[i - 1]),
            "low": float(low[i - 1]),
            "close": float(close[i - 1]),
            "volume": float(volume[i - 1]),
        }
    o = float(open_[i])
    return {
        "open": o,
        "high": o,
        "low": o,
        "close": o,
        "volume": float(volume[i]),
    }


def fast_backtest(ohlc: pd.DataFrame, target: pd.Series, cost_model: CostModel,
                  initial_capital: float = 100_000.0,
                  units: float = 100.0,
                  liquidity_policy: str = "partial") -> FastResult:
    """Vectorized T+1 backtest on one symbol.

    ohlc: DataFrame [open, high, low, close, volume]; target: position after
    each close in {-1, 0, +1} (executed at the NEXT bar's open); units: shares
    per unit of target.

    When ``cost_model`` exposes ``max_fill_quantity`` (e.g. RetailCostModel),
    per-bar fill size is clipped to that cap so MC paths cannot ignore the
    event-engine participation constraint. ``liquidity_policy`` mirrors the
    simulator: ``partial`` carries unfilled delta to later bars, ``reject``
    drops the excess, ``none`` disables the cap.
    """
    if liquidity_policy not in {"partial", "reject", "none"}:
        raise ValueError("liquidity_policy must be 'partial', 'reject', or 'none'")

    target = target.reindex(ohlc.index).fillna(0.0)

    desired = (target.shift(1).fillna(0.0) * units).to_numpy(dtype=float)
    open_ = ohlc["open"].to_numpy(dtype=float)
    high = ohlc["high"].to_numpy(dtype=float)
    low = ohlc["low"].to_numpy(dtype=float)
    close = ohlc["close"].to_numpy(dtype=float)
    volume = ohlc["volume"].to_numpy(dtype=float)

    q = np.zeros_like(desired)
    dq = np.zeros_like(desired)
    fill_price = np.empty_like(open_)
    commission = np.zeros_like(open_)
    position = 0.0
    residual = 0.0
    has_cap = (
        liquidity_policy != "none"
        and hasattr(cost_model, "max_fill_quantity")
    )

    for i in range(len(open_)):
        want = desired[i] + residual
        delta = want - position
        fill_qty = abs(delta)
        if has_cap and fill_qty > 0:
            cap = float(cost_model.max_fill_quantity(float(volume[i])))
            if fill_qty > cap + 1e-12:
                if liquidity_policy == "reject" or cap <= 0:
                    fill_qty = 0.0
                    residual = 0.0
                    delta = 0.0
                else:
                    fill_qty = cap
                    delta = np.sign(delta) * fill_qty
                    residual = want - (position + delta)
            else:
                residual = 0.0
        else:
            residual = 0.0

        dq[i] = delta
        if abs(delta) < 1e-12:
            fill_price[i] = open_[i]
            q[i] = position
            continue
        # Route through adverse_adjustment (not the individual terms) so any
        # CostModel override — e.g. ConservativeFrictionCostModel — applies
        # identically on both tracks. Open-fill cost bar matches the engine
        # temporal firewall (prior bar / zero-range proxy).
        cost_bar = _open_cost_bar(i, open_, high, low, close, volume)
        adj = cost_model.adverse_adjustment(open_[i], abs(delta), cost_bar)
        fill_price[i] = open_[i] + np.sign(delta) * adj
        # Pass the fill reference so notional-based cost models (e.g.
        # ConservativeFrictionCostModel) charge identical fees on both tracks.
        commission[i] = cost_model.commission(abs(delta), price=open_[i])
        position = position + delta
        q[i] = position

    cash = initial_capital + np.cumsum(-dq * fill_price - commission)
    equity = cash + q * close

    idx = ohlc.index
    equity_s = pd.Series(equity, index=idx, name="equity")
    return FastResult(
        equity=equity_s,
        positions=pd.Series(q, index=idx, name="position"),
        cash=pd.Series(cash, index=idx, name="cash"),
        daily_returns=equity_s.pct_change(),
    )
