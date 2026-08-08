"""Risk overlays (Report 1 section 2.3, Report 2 section 2).

Spectral risk attribution R_n = beta_n^2 * Lambda_nn / sigma^2 runs on a
STABILIZED covariance matrix: Ledoit-Wolf shrinkage is applied before any
eigendecomposition (Cross-Ref-2 section 3.C -- with N assets approaching or
exceeding the observation count, the raw sample covariance is singular or
ill-conditioned and raw eigenvalues are numerical noise).

MarginMonitor tracks leverage and triggers liquidation logic at breaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def stabilized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf-shrunk covariance of asset returns."""
    lw = LedoitWolf().fit(returns.to_numpy())
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


def spectral_risk_attribution(returns: pd.DataFrame,
                              weights: pd.Series | None = None) -> pd.DataFrame:
    """Attribute portfolio variance to orthogonal principal components.

    beta_n = w'v_n (loading of the portfolio on component n);
    R_n = beta_n^2 * Lambda_nn / sigma^2 with sigma^2 = w' Sigma w.
    Returns a DataFrame [eigenvalue, beta^2, risk_share] sorted by eigenvalue.
    """
    cov = stabilized_covariance(returns)
    if weights is None:
        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    weights = weights.reindex(returns.columns).fillna(0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov.to_numpy())
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    w = weights.to_numpy()
    total_var = float(w @ cov.to_numpy() @ w)
    beta = eigenvectors.T @ w
    component_var = beta**2 * eigenvalues
    if total_var > 0:
        risk_share = component_var / total_var
    else:
        risk_share = np.zeros_like(component_var)

    return pd.DataFrame(
        {
            "eigenvalue": eigenvalues,
            "beta_sq": beta**2,
            "risk_share": risk_share,
        },
        index=[f"PC{i + 1}" for i in range(len(eigenvalues))],
    )


class DailyDrawdownBreaker:
    """Intraday circuit breaker against the daily opening balance.

    Verification status: not covered by the notebook — implemented from the
    user's strategy specification (prop-evaluation safeguard: trip at a 4.5%
    intraday loss, a 0.5% cushion below the hard 5.0% daily-loss limit).

    Fires when (day_open_equity - equity) / day_open_equity >= max_intraday_dd
    on any close-phase valuation. While halted, the portfolio liquidates all
    positions, cancels every resting order, and drops new entry signals; the
    halt resets at the next trading-day rollover (timestamp date change).

    Daily opening balance convention: the last equity valuation of the previous
    trading day (exchange rollover carry). The first valuation of the backtest
    seeds the baseline from itself, so the breaker cannot trip on day one
    unless equity later falls below that same-day baseline.
    """

    def __init__(self, max_intraday_dd: float = 0.045):
        if not 0.0 < max_intraday_dd < 1.0:
            raise ValueError("max_intraday_dd must lie in (0, 1)")
        self.max_intraday_dd = float(max_intraday_dd)
        self.halted = False
        self.triggered_count = 0
        self._day = None
        self._day_open_equity: float | None = None
        self._last_equity: float | None = None

    def update(self, timestamp, equity: float) -> bool:
        """Roll the daily baseline and check the breach; returns True only on
        the valuation call that newly trips the breaker."""
        day = pd.Timestamp(timestamp).date()
        if day != self._day:
            self._day = day
            self.halted = False
            self._day_open_equity = (
                self._last_equity if self._last_equity is not None else equity
            )
        self._last_equity = equity
        if self.halted or not self._day_open_equity or self._day_open_equity <= 0:
            return False
        drawdown = (self._day_open_equity - equity) / self._day_open_equity
        if drawdown >= self.max_intraday_dd:
            self.halted = True
            self.triggered_count += 1
            return True
        return False


class MarginMonitor:
    """Tracks leverage = gross exposure / equity; breaches trigger liquidation.

    Preferred state machine after a margin breach:

        MARGIN_BREACH
            → cancel / block new entry risk
            → liquidate (shrink positions)
            → remain restricted until leverage recovers below max_leverage

    While ``restricted`` is True, the portfolio must not allow strategies to
    *increase* gross exposure. Liquidation and risk-reducing exits remain
    permitted. Restriction clears only when leverage falls back to
    ``max_leverage`` (explicit safe recovery), not merely because a
    liquidation order has been queued.
    """

    def __init__(self, max_leverage: float = 2.0, liquidation_fraction: float = 0.5):
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if not 0.0 < liquidation_fraction <= 1.0:
            raise ValueError("liquidation_fraction must lie in (0, 1]")
        self.max_leverage = max_leverage
        self.liquidation_fraction = liquidation_fraction
        self.restricted = False
        self.breach_count = 0

    def leverage(self, equity: float, gross_exposure: float) -> float:
        if equity <= 0:
            return np.inf
        return gross_exposure / equity

    def is_breach(self, equity: float, gross_exposure: float) -> bool:
        return self.leverage(equity, gross_exposure) > self.max_leverage

    def update(self, equity: float, gross_exposure: float) -> bool:
        """Update restriction state; return True whenever still breached.

        Returns True on every valuation call while leverage remains above
        ``max_leverage`` so the portfolio can re-issue shrink orders (one-shot
        liquidation is insufficient when fills are partial, capped, or when a
        single ``liquidation_fraction`` cut does not restore safe leverage).
        ``breach_count`` increments only on the transition into restriction.
        """
        if self.is_breach(equity, gross_exposure):
            newly = not self.restricted
            self.restricted = True
            if newly:
                self.breach_count += 1
            return True
        # Explicit safe recovery: leverage back at or below the limit.
        self.restricted = False
        return False

    def liquidation_targets(self, positions: dict) -> dict:
        """Target quantities after de-risking: shrink every position."""
        return {
            symbol: qty * (1.0 - self.liquidation_fraction)
            for symbol, qty in positions.items()
        }
