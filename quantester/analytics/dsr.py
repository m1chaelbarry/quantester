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
                               skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """PSR: probability that the true Sharpe exceeds the benchmark."""
    if n_obs <= 1:
        return 0.0
    denom = np.sqrt(max(1 - skew * sr_hat + (kurtosis - 1) / 4.0 * sr_hat**2, 1e-12))
    return float(norm.cdf((sr_hat - sr_benchmark) * np.sqrt(n_obs - 1) / denom))


def deflated_sharpe_ratio(sr_hat: float, n_trials: int, trial_variance: float,
                          n_obs: int, skew: float = 0.0,
                          kurtosis: float = 3.0) -> float:
    """DSR: PSR against the expected maximum Sharpe of N null trials."""
    benchmark = expected_max_sharpe(n_trials, trial_variance)
    return probabilistic_sharpe_ratio(sr_hat, benchmark, n_obs, skew, kurtosis)


def dsr_from_registry(registry, sr_hat: float, n_obs: int, skew: float = 0.0,
                      kurtosis: float = 3.0) -> float:
    """Registry-driven DSR: N and sigma^2_SR reflect what was actually tried."""
    return deflated_sharpe_ratio(
        sr_hat=sr_hat,
        n_trials=max(registry.n_trials(), 1),
        trial_variance=registry.sharpe_variance(),
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )
