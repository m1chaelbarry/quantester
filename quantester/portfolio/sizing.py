"""Position sizing engines (Report 1 section 2.3).

- Kelly criterion (discrete win/loss and continuous Gaussian forms)
- Volatility parity (equal risk contribution across assets)
- Ralph Vince's optimal-f (notebook-verified):
    HPR_i = 1 + f * (-Trade_i / WorstLoss)
    TWR(f) = prod_i HPR_i
    f* = argmax_f TWR(f)
  WorstLoss is gap-stressed below the nominal stop: stops do not guarantee fills
  through overnight gaps or limit-down halts (Cross-Ref-2 section 4.2), and
  unconstrained optimal-f is catastrophically sensitive to exceeding the
  historical max loss (Cross-Ref section 2.D). Vince's own mitigations:
  dilution flattens drawdowns arithmetically but cuts returns geometrically;
  dynamic fractional-f reallocates with equity shifts.
- Kakushadze effective-return adjustment applied BEFORE weight optimization:
    E_eff = sign(E) * max(|E| - tau, 0)
  so marginal edges smaller than linear trading costs are zeroed (Cross-Ref 3.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Kelly f* = p - q/b for binary outcomes (b = avg win / avg loss)."""
    p = win_rate
    q = 1.0 - p
    return p - q / win_loss_ratio


def kelly_gaussian(mean: float, variance: float) -> float:
    """Continuous Kelly fraction f = mu / sigma^2."""
    if variance <= 0:
        return 0.0
    return mean / variance


def volatility_parity_weights(cov: pd.DataFrame) -> pd.Series:
    """w_i proportional to 1/sigma_i, normalized to sum to 1."""
    vols = np.sqrt(np.diag(cov))
    inv = 1.0 / np.where(vols > 0, vols, np.inf)
    weights = pd.Series(inv, index=cov.index)
    total = weights.sum()
    if not np.isfinite(total) or total == 0:
        return pd.Series(1.0 / len(cov), index=cov.index)
    return weights / total


def hpr(trades: np.ndarray, f: float, worst_loss: float) -> np.ndarray:
    """Holding Period Returns: HPR_i = 1 + f * (-Trade_i / WorstLoss)."""
    return 1.0 + f * (-np.asarray(trades, dtype=float) / worst_loss)


def twr(trades: np.ndarray, f: float, worst_loss: float) -> float:
    """Terminal Wealth Ratio: product of HPRs (0 if any HPR <= 0: ruin)."""
    hprs = hpr(trades, f, worst_loss)
    if (hprs <= 0).any():
        return 0.0
    return float(hprs.prod())


def optimal_f(trades, worst_loss: float | None = None, gap_stress: float = 1.5,
              f_max: float = 1.0) -> float:
    """f* = argmax TWR(f) over [0, f_max].

    worst_loss defaults to the historical worst loss multiplied by `gap_stress`
    (>1 stresses it below the nominal stop for gap-through risk). If there are
    no losing trades, f_max is returned (no loss-bounded optimum exists).
    """
    trades = np.asarray(trades, dtype=float)
    if len(trades) == 0:
        return 0.0
    if worst_loss is None:
        historical_worst = float(trades.min())
        if historical_worst >= 0:
            return f_max
        worst_loss = historical_worst * gap_stress
    if worst_loss >= 0:
        raise ValueError("worst_loss must be negative")
    result = minimize_scalar(
        lambda f: -twr(trades, f, worst_loss),
        bounds=(0.0, f_max),
        method="bounded",
    )
    return float(result.x)


def kakushadze_effective_returns(expected: pd.Series, linear_costs: pd.Series) -> pd.Series:
    """E_eff = sign(E) * max(|E| - tau, 0): zeroes edges smaller than trading costs."""
    expected, linear_costs = expected.align(linear_costs, fill_value=0.0)
    magnitude = (expected.abs() - linear_costs.abs()).clip(lower=0.0)
    return np.sign(expected) * magnitude
