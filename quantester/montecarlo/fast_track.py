"""Vectorized execution bypass for Monte Carlo (Cross-Ref section 3.2).

Running 10,000 permuted retraining loops through the pure-Python event queue is
computationally intractable (Cross-Ref section 2.C). The fast-track applies
strategy logic and cost models directly to NumPy/pandas arrays.

PARITY CONTRACT with the event engine (asserted by test):
- target positions are decided on bar T's close and executed at bar T+1's open
- fills use the SAME CostModel adverse adjustments as SimulatedExecutionHandler
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


def fast_backtest(ohlc: pd.DataFrame, target: pd.Series, cost_model: CostModel,
                  initial_capital: float = 100_000.0,
                  units: float = 100.0) -> FastResult:
    """Vectorized T+1 backtest on one symbol.

    ohlc: DataFrame [open, high, low, close, volume]; target: position after
    each close in {-1, 0, +1} (executed at the NEXT bar's open); units: shares
    per unit of target.
    """
    target = target.reindex(ohlc.index).fillna(0.0)

    q = (target.shift(1).fillna(0.0) * units).to_numpy()
    dq = np.diff(q, prepend=0.0)

    open_ = ohlc["open"].to_numpy()
    high = ohlc["high"].to_numpy()
    low = ohlc["low"].to_numpy()
    close = ohlc["close"].to_numpy()
    volume = ohlc["volume"].to_numpy()

    fill_price = np.empty_like(open_)
    commission = np.zeros_like(open_)
    for i in range(len(open_)):
        if dq[i] == 0.0:
            fill_price[i] = open_[i]
            continue
        # Route through adverse_adjustment (not the individual terms) so any
        # CostModel override — e.g. ConservativeFrictionCostModel — applies
        # identically on both tracks.
        adj = cost_model.adverse_adjustment(
            open_[i], abs(dq[i]),
            {"high": high[i], "low": low[i], "volume": volume[i]},
        )
        fill_price[i] = open_[i] + np.sign(dq[i]) * adj
        # Pass the fill reference so notional-based cost models (e.g.
        # ConservativeFrictionCostModel) charge identical fees on both tracks.
        commission[i] = cost_model.commission(abs(dq[i]), price=open_[i])

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
