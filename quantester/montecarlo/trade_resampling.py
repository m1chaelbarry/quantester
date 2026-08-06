"""Trade-level resampling and equity-curve modeling (MC Report section 2).

- Ehlers' parametric equity randomization: the system is stripped to win-rate p
  and profit factor PF; each trade draws u ~ U(0,1), wins if u <= p, paying
  avg_loss * PF, else loses avg_loss. M = 10,000 simulated paths give the
  distribution of final returns.
- Empirical "hat" resampling: draw historical returns WITH REPLACEMENT to build
  synthetic paths (e.g. 260-day years), preserving the exact empirical return
  distribution; block-bootstrap variant preserves serial correlation (Kaufman's
  autocorrelation trap, section 2.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResampleResult:
    paths: np.ndarray          # (n_sims, horizon + 1) cumulative equity, starts at 1.0
    terminal_returns: np.ndarray

    def quantiles(self, qs=(0.05, 0.5, 0.95)) -> dict:
        return {q: float(np.quantile(self.terminal_returns, q)) for q in qs}


def ehlers_randomized_equity(win_rate: float, profit_factor: float,
                             avg_loss: float, n_trades: int,
                             n_sims: int = 10_000, e0: float = 1.0,
                             seed: int | None = None) -> np.ndarray:
    """Parametric randomization; returns (n_sims, n_trades + 1) equity paths."""
    if not (0.0 <= win_rate <= 1.0):
        raise ValueError("win_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    u = rng.random((n_sims, n_trades))
    pnl = np.where(u <= win_rate, abs(avg_loss) * profit_factor, -abs(avg_loss))
    equity = np.concatenate([np.full((n_sims, 1), e0), e0 + np.cumsum(pnl, axis=1)], axis=1)
    return equity


def _iid_draws(rng, pool: np.ndarray, n_sims: int, horizon: int) -> np.ndarray:
    idx = rng.integers(0, len(pool), size=(n_sims, horizon))
    return pool[idx]


def _block_draws(rng, pool: np.ndarray, n_sims: int, horizon: int,
                 block_length: int) -> np.ndarray:
    out = np.empty((n_sims, horizon))
    n_blocks = int(np.ceil(horizon / block_length))
    max_start = len(pool) - block_length
    if max_start < 1:
        raise ValueError("block_length too large for the return pool")
    for s in range(n_sims):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        seq = np.concatenate([pool[st : st + block_length] for st in starts])
        out[s] = seq[:horizon]
    return out


def empirical_resample(returns, horizon: int = 260, n_sims: int = 10_000,
                       seed: int | None = None,
                       block_length: int | None = None) -> ResampleResult:
    """"Hat" resampling with replacement of net returns.

    block_length=None -> iid draws; block_length=L -> stationary block
    bootstrap preserving serial correlation within blocks.
    """
    pool = np.asarray(returns, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0:
        raise ValueError("empty return pool")
    rng = np.random.default_rng(seed)
    if block_length is None:
        draws = _iid_draws(rng, pool, n_sims, horizon)
    else:
        draws = _block_draws(rng, pool, n_sims, horizon, block_length)
    paths = np.concatenate(
        [np.ones((n_sims, 1)), 1.0 + np.cumsum(draws, axis=1)], axis=1
    )
    return ResampleResult(paths=paths, terminal_returns=paths[:, -1] - 1.0)
