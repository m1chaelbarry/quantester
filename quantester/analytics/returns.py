"""Explicit return / wealth representation helpers.

Quantester distinguishes four quantities that must never be silently mixed:

- **simple returns** ``r_t = P_t / P_{t-1} - 1`` — compound geometrically
- **log returns** ``ℓ_t = log(P_t / P_{t-1})`` — aggregate additively
- **P&L** — additive currency amounts (not returns)
- **equity / wealth** — capital level series

Verification status: not covered by the notebook as a standalone module —
implemented from elementary return algebra (canonical identities).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns_from_prices(prices) -> np.ndarray:
    """``P_t / P_{t-1} - 1``; drops the leading NaN."""
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return np.asarray([], dtype=float)
    return p[1:] / p[:-1] - 1.0


def log_returns_from_prices(prices) -> np.ndarray:
    """``log(P_t / P_{t-1})``; drops the leading NaN."""
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return np.asarray([], dtype=float)
    return np.log(p[1:] / p[:-1])


def simple_to_log(simple_returns) -> np.ndarray:
    """``log(1 + r)`` for valid simple returns ``r > -1``."""
    r = np.asarray(simple_returns, dtype=float)
    if np.any(~np.isfinite(r)):
        raise ValueError("simple_returns contain NaN/inf")
    if np.any(r <= -1.0):
        raise ValueError("simple returns must be > -1 for log conversion")
    return np.log1p(r)


def log_to_simple(log_returns) -> np.ndarray:
    """``exp(ℓ) - 1``."""
    ell = np.asarray(log_returns, dtype=float)
    if np.any(~np.isfinite(ell)):
        raise ValueError("log_returns contain NaN/inf")
    return np.expm1(ell)


def wealth_from_simple_returns(simple_returns, initial: float = 1.0) -> np.ndarray:
    """``W_t = W_0 * cumprod(1 + r)`` — length ``len(r) + 1`` including ``W_0``."""
    r = np.asarray(simple_returns, dtype=float)
    if r.size == 0:
        return np.asarray([initial], dtype=float)
    if np.any(~np.isfinite(r)):
        raise ValueError("simple_returns contain NaN/inf")
    if np.any(r <= -1.0):
        raise ValueError("simple returns must be > -1 for wealth construction")
    return np.concatenate([[initial], initial * np.cumprod(1.0 + r)])


def wealth_from_log_returns(log_returns, initial: float = 1.0) -> np.ndarray:
    """``W_t = W_0 * exp(cumsum(ℓ))`` — length ``len(ℓ) + 1`` including ``W_0``."""
    ell = np.asarray(log_returns, dtype=float)
    if ell.size == 0:
        return np.asarray([initial], dtype=float)
    if np.any(~np.isfinite(ell)):
        raise ValueError("log_returns contain NaN/inf")
    return np.concatenate([[initial], initial * np.exp(np.cumsum(ell))])


def simple_returns_from_equity(equity) -> pd.Series | np.ndarray:
    """Period simple returns from an equity / wealth level series."""
    if isinstance(equity, pd.Series):
        return equity.pct_change().dropna()
    e = np.asarray(equity, dtype=float)
    if e.size < 2:
        return np.asarray([], dtype=float)
    return e[1:] / e[:-1] - 1.0
