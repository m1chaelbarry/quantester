"""de Prado's ETF Trick (AFML; notebook-verified exact recurrence).

Total-return index of $1 invested in a multi-product basket (e.g. a futures
spread), letting strategies treat complex spreads as simple cash instruments:

    K_t = K_{t-1} + sum_i h_{i,t-1} * phi_{i,t} * (delta_{i,t} + d_{i,t}),  K_0 = 1

with
    phi_{i,t} : USD value of one point of instrument i (incl. FX)
    d_{i,t}   : carry/dividend/coupon (may carry margin/funding costs)
    delta_{i,t} = p_t - o_t     if t-1 is a rebalance/roll bar
                = p_t - p_{t-1} otherwise
    h_{i,t}   = h_{i,t-1} off rebalance bars; on rebalance bars
                h_{i,t} = omega_{i,t} * K_t / (price * phi_{i,t} * sum_j|omega_j|)
                (roll bars price = o_{i,t+1}, else p_{i,t}; sum|omega| de-levers)

Rebalancing costs are computed but kept strictly EXTERNAL to K_t:

    c_t = sum_i (|h_{i,t-1}|*p_{i,t} + |h_{i,t}|*o_{i,t+1}) * phi_{i,t} * tau_i,  t in B

de Prado verbatim: "We do not embed c_t in K_t, or shorting the spread will
generate fictitious profits when the allocation is rebalanced. In your code, you
can treat {c_t} as a (negative) dividend." (Cross-Ref-2 section 3.A)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ETFTrick:
    def __init__(
        self,
        weights: pd.DataFrame,          # omega: T x I allocation vectors
        open_prices: pd.DataFrame,      # o: T x I raw opens
        close_prices: pd.DataFrame,     # p: T x I raw closes
        rebalance_times,                # B: set/Index of rebalance/roll timestamps
        point_values: pd.DataFrame | float = 1.0,   # phi
        dividends: pd.DataFrame | float = 0.0,      # d
        cost_rates: pd.Series | float = 0.0,        # tau_i per instrument
        roll_times=None,                # subset of B priced with o_{t+1}
        aum0: float = 1.0,
    ):
        self.w = weights.astype(float)
        self.o = open_prices.astype(float).reindex(self.w.index)
        self.p = close_prices.astype(float).reindex(self.w.index)
        self.B = pd.DatetimeIndex(rebalance_times)
        self.roll_times = pd.DatetimeIndex(roll_times) if roll_times is not None else pd.DatetimeIndex([])
        self.phi = self._broadcast(point_values, "point_values")
        self.d = self._broadcast(dividends, "dividends")
        if np.isscalar(cost_rates):
            self.tau = pd.Series(float(cost_rates), index=self.w.columns)
        else:
            self.tau = cost_rates.astype(float)
        self.aum0 = float(aum0)

    def _broadcast(self, value, name) -> pd.DataFrame:
        if np.isscalar(value):
            return pd.DataFrame(float(value), index=self.w.index, columns=self.w.columns)
        return value.astype(float).reindex(index=self.w.index, columns=self.w.columns)

    def compute(self) -> pd.DataFrame:
        """Returns DataFrame with K (cost-free index) and c (external rebalancing cost)."""
        idx = self.w.index
        cols = self.w.columns
        n = len(idx)

        K = np.zeros(n)
        c = np.zeros(n)
        h_prev = pd.Series(0.0, index=cols)
        K[0] = self.aum0
        if idx[0] in self.B:
            h_prev = self._rebalance_holdings(0, K[0])
            c[0] = self._rebalance_cost(0, pd.Series(0.0, index=cols), h_prev)

        prev_rebalanced = idx[0] in self.B
        for t in range(1, n):
            ts = idx[t]
            if prev_rebalanced:
                delta = self.p.iloc[t] - self.o.iloc[t]
            else:
                delta = self.p.iloc[t] - self.p.iloc[t - 1]
            K[t] = K[t - 1] + float((h_prev * self.phi.iloc[t] * (delta + self.d.iloc[t])).sum())

            if ts in self.B:
                h_new = self._rebalance_holdings(t, K[t])
                c[t] = self._rebalance_cost(t, h_prev, h_new)
                h_prev = h_new
                prev_rebalanced = True
            else:
                prev_rebalanced = False

        return pd.DataFrame({"K": K, "c": c}, index=idx)

    def _rebalance_holdings(self, t: int, k_value: float) -> pd.Series:
        ts = self.w.index[t]
        delever = float(self.w.iloc[t].abs().sum())
        if delever == 0.0:
            return pd.Series(0.0, index=self.w.columns)
        if ts in self.roll_times and t + 1 < len(self.w.index):
            price = self.o.iloc[t + 1]  # new contract price unknown at roll: use next open
        else:
            price = self.p.iloc[t]
        return self.w.iloc[t] * k_value / (price * self.phi.iloc[t] * delever)

    def _rebalance_cost(self, t: int, h_old: pd.Series, h_new: pd.Series) -> float:
        if t + 1 < len(self.w.index):
            next_open = self.o.iloc[t + 1]
        else:
            next_open = self.p.iloc[t]
        legs = (h_old.abs() * self.p.iloc[t] + h_new.abs() * next_open) * self.phi.iloc[t] * self.tau
        return float(legs.sum())
