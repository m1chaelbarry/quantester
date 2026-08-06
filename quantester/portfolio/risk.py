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


class MarginMonitor:
    """Tracks leverage = gross exposure / equity; breaches trigger liquidation."""

    def __init__(self, max_leverage: float = 2.0, liquidation_fraction: float = 0.5):
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        self.max_leverage = max_leverage
        self.liquidation_fraction = liquidation_fraction

    def leverage(self, equity: float, gross_exposure: float) -> float:
        if equity <= 0:
            return np.inf
        return gross_exposure / equity

    def is_breach(self, equity: float, gross_exposure: float) -> bool:
        return self.leverage(equity, gross_exposure) > self.max_leverage

    def liquidation_targets(self, positions: dict) -> dict:
        """Target quantities after de-risking: shrink every position."""
        return {
            symbol: qty * (1.0 - self.liquidation_fraction)
            for symbol, qty in positions.items()
        }
