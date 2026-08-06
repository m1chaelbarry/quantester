"""Autocorrelation diagnostics gate (MC Report section 6, Autocorrelation Check).

Runs-test and Ljung-Box on backtest residuals. If serial correlation exists,
simple trade shuffling artificially smooths simulated paths and dangerously
underestimates downside risk (Kaufman's autocorrelation trap): the gate then
routes to block bootstrapping or O-U synthetic paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm


def runs_test(x) -> tuple[float, float]:
    """Wald-Wolfowitz runs test on sign(x - median); returns (z, p)."""
    x = np.asarray(x, dtype=float)
    signs = np.sign(x - np.median(x))
    signs = signs[signs != 0]
    n1 = int((signs > 0).sum())
    n2 = int((signs < 0).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 2:
        return 0.0, 1.0
    runs = 1 + int((signs[1:] != signs[:-1]).sum())
    mu = 2.0 * n1 * n2 / n + 1.0
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n**2 * (n - 1))
    if var <= 0:
        return 0.0, 1.0
    z = (runs - mu) / np.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(z), float(p)


def ljung_box(x, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q on raw autocorrelations; returns (Q, p)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= lags + 1:
        return 0.0, 1.0
    x = x - x.mean()
    denom = float((x**2).sum())
    if denom == 0:
        return 0.0, 1.0
    q = 0.0
    for k in range(1, lags + 1):
        rk = float((x[k:] * x[:-k]).sum()) / denom
        q += rk**2 / (n - k)
    q *= n * (n + 2)
    return float(q), float(1.0 - chi2.cdf(q, lags))


@dataclass
class DiagnosticsReport:
    serial_correlation: bool
    recommended_method: str
    runs_z: float
    runs_p: float
    ljung_box_q: float
    ljung_box_p: float


def autocorrelation_gate(returns, alpha: float = 0.05,
                         lags: int = 10) -> DiagnosticsReport:
    """If serial correlation is detected, iid resampling is INVALID."""
    z, runs_p = runs_test(returns)
    q, lb_p = ljung_box(returns, lags)
    serial = (runs_p < alpha) or (lb_p < alpha)
    return DiagnosticsReport(
        serial_correlation=serial,
        recommended_method="block_bootstrap_or_ou_paths" if serial else "iid_resampling",
        runs_z=z,
        runs_p=runs_p,
        ljung_box_q=q,
        ljung_box_p=lb_p,
    )
