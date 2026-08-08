"""Deflated Sharpe Ratio (Bailey & Lopez de Prado).

Primary defense against selection bias: deflates the observed Sharpe by the
expected maximum Sharpe under the null that all N trials had zero skill.

    E[max SR_N] = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N)
                             + gamma * Phi^-1(1 - 1/(N*e)))
    DSR = Phi( ((SR_hat - E[max SR_N]) * sqrt(T - 1))
               / sqrt(1 - skew*SR_hat + (kurt - 1)/4 * SR_hat^2) )

where V is the cross-trial variance of Sharpe ratios, gamma the
Euler-Mascheroni constant, T the number of return observations, and
skew/kurt the selected trial's return moments.

NOTE: the exact formula was NOT covered by the user's notebook; implemented from
the canonical Bailey-de Prado 2014 paper ("The Deflated Sharpe Ratio") and
flagged accordingly (plan, Source verification status).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """E[max_N] under the null: quantile blend with Euler-Mascheroni constant."""
    if n_trials <= 1 or trial_variance <= 0:
        return 0.0
    n = float(n_trials)
    return float(
        np.sqrt(trial_variance)
        * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n)
            + EULER_MASCHERONI * norm.ppf(1 - 1 / (n * np.e))
        )
    )


def probabilistic_sharpe_ratio(sr_hat: float, sr_benchmark: float, n_obs: int,
                               skew: float = 0.0, kurtosis: float = 3.0,
                               *, annualized: bool = False,
                               periods_per_year: float = 252.0) -> float:
    """PSR: probability that the true Sharpe exceeds the benchmark.

    ``sr_hat`` and ``sr_benchmark`` must be **per-period** Sharpes matching
    ``n_obs`` return observations, unless ``annualized=True`` in which case
    they are de-annualized by ``sqrt(periods_per_year)`` before the Bailey
    formula is applied. Mixing annualized SR with bar-count ``n_obs`` without
    this flag massively inflates PSR/DSR.
    """
    if n_obs <= 1:
        return 0.0
    if annualized:
        scale = np.sqrt(max(periods_per_year, 1e-12))
        sr_hat = sr_hat / scale
        sr_benchmark = sr_benchmark / scale
    denom = np.sqrt(max(1 - skew * sr_hat + (kurtosis - 1) / 4.0 * sr_hat**2, 1e-12))
    return float(norm.cdf((sr_hat - sr_benchmark) * np.sqrt(n_obs - 1) / denom))


def deflated_sharpe_ratio(sr_hat: float, n_trials: int, trial_variance: float,
                          n_obs: int, skew: float = 0.0,
                          kurtosis: float = 3.0, *,
                          annualized: bool = False,
                          periods_per_year: float = 252.0) -> float:
    """DSR: PSR against the expected maximum Sharpe of N null trials.

    See ``probabilistic_sharpe_ratio`` for the annualized vs per-period
    contract. ``kurtosis`` must be **Pearson** (normal = 3), not Fisher excess.
    """
    if annualized and trial_variance > 0:
        # Cross-trial variance of annualized Sharpes → per-period variance.
        trial_variance = trial_variance / max(periods_per_year, 1e-12)
    benchmark = expected_max_sharpe(n_trials, trial_variance)
    return probabilistic_sharpe_ratio(
        sr_hat, benchmark, n_obs, skew, kurtosis,
        annualized=annualized, periods_per_year=periods_per_year,
    )


def dsr_from_registry(registry, sr_hat: float, n_obs: int, skew: float = 0.0,
                      kurtosis: float = 3.0, *,
                      annualized: bool = False,
                      periods_per_year: float = 252.0) -> float:
    """Registry-driven DSR: N and sigma^2_SR reflect what was actually tried.

    Pass ``annualized=True`` when ``sr_hat`` and registry Sharpe values are
    annualized (typical for ``annualized_sharpe`` / tearsheet outputs).
    """
    trial_variance = registry.sharpe_variance()
    return deflated_sharpe_ratio(
        sr_hat=sr_hat,
        n_trials=max(registry.n_trials(), 1),
        trial_variance=trial_variance,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
        annualized=annualized,
        periods_per_year=periods_per_year,
    )
