"""Assumption-aware Monte Carlo resampling.

Routes simple-return resampling through the autocorrelation diagnostic gate:
IID hat resampling is only used when residuals look serially independent;
otherwise a block bootstrap is selected (with an explicit warning). Block
bootstrap is a modelling assumption about dependence length, not a universal
truth for all financial series.

Verification status: not covered by the notebook — orchestration layer over
existing diagnostics + empirical_resample (MC Report §6).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .diagnostics import DiagnosticsReport, autocorrelation_gate
from .trade_resampling import ResampleResult, empirical_resample


@dataclass
class AdaptiveResampleResult:
    result: ResampleResult
    diagnostics: DiagnosticsReport
    method_used: str
    block_length: int | None
    warning: str | None = None


def suggest_block_length(returns, max_lag: int = 20) -> int:
    """Heuristic block length from the first lag where |ρ| drops below 2/√n.

    Falls back to ``max(2, round(n**(1/3)))`` when no lag crosses the threshold.
    Documented as a sensitivity starting point — not claimed optimal.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        return 2
    x = r - r.mean()
    denom = float((x ** 2).sum())
    if denom == 0:
        return 2
    thresh = 2.0 / np.sqrt(n)
    chosen = None
    for k in range(1, min(max_lag, n - 1) + 1):
        rk = abs(float((x[k:] * x[:-k]).sum()) / denom)
        if rk < thresh:
            chosen = k
            break
    if chosen is None:
        chosen = max(2, int(round(n ** (1.0 / 3.0))))
    return int(max(2, min(chosen, n // 2)))


def adaptive_empirical_resample(
    returns,
    horizon: int = 260,
    n_sims: int = 10_000,
    seed: int | None = None,
    *,
    alpha: float = 0.05,
    lags: int = 10,
    block_length: int | None = None,
    force_iid: bool = False,
) -> AdaptiveResampleResult:
    """Resample simple returns with diagnostic-aware method selection.

    If serial correlation is detected and ``force_iid`` is False, uses block
    bootstrap (``block_length`` or a heuristic). Forcing IID when dependence
    exists emits a warning — IID under dependence underestimates downside risk
    (Kaufman's autocorrelation trap).
    """
    gate = autocorrelation_gate(returns, alpha=alpha, lags=lags)
    warning = None
    used_block = None
    method = "iid_resampling"

    if gate.serial_correlation and not force_iid:
        used_block = block_length or suggest_block_length(returns)
        method = "block_bootstrap"
        warning = (
            "Serial correlation detected "
            f"(runs_p={gate.runs_p:.4g}, ljung_box_p={gate.ljung_box_p:.4g}); "
            f"using block bootstrap with block_length={used_block}. "
            "Block length is a modelling assumption — sensitivity-test it."
        )
    elif gate.serial_correlation and force_iid:
        method = "iid_resampling_forced"
        warning = (
            "Serial correlation detected but force_iid=True: IID resampling "
            "will likely underestimate path-dependent downside risk."
        )
        warnings.warn(warning, UserWarning, stacklevel=2)
    elif block_length is not None:
        used_block = block_length
        method = "block_bootstrap"

    if warning and method == "block_bootstrap":
        warnings.warn(warning, UserWarning, stacklevel=2)

    result = empirical_resample(
        returns,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed,
        block_length=used_block,
    )
    return AdaptiveResampleResult(
        result=result,
        diagnostics=gate,
        method_used=method,
        block_length=used_block,
        warning=warning,
    )
