"""Probability of Backtest Overfitting via Bailey-de Prado CSCV
(notebook-verified exact algorithm).

1. Matrix M (T x N): each column n is the synchronous PnL series of trial n.
2. Partition row-wise into an even number S of disjoint submatrices.
3. Form all C(S, S/2) combinations; for each combination c:
   - train J = S/2 blocks joined; test J^c = complement
   - n* = argmax_n R_n (train performance); R^c = test performance vector
   - relative rank omega_bar_c = Rank(R^c_{n*}) / (N + 1) in (0, 1)
   - logit lambda_c = log(omega_bar_c / (1 - omega_bar_c))
4. PBO = integral_{-inf}^0 f(lambda) d lambda, i.e. P(lambda < 0).

Hard gate at PBO < 0.10 before any model is approved for paper trading
(Report 1 section 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PBO_GATE = 0.10


@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_combinations: int
    n_trials: int

    @property
    def passes_gate(self) -> bool:
        return self.pbo < PBO_GATE


def _default_performance(block: np.ndarray) -> np.ndarray:
    """Per-trial Sharpe of a (t x N) PnL block."""
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(std > 0, mean / std, 0.0)
    return sharpe


def pbo_cscv(pnl: pd.DataFrame, n_blocks: int = 16,
             performance_fn=_default_performance) -> PBOResult:
    """pnl: (T x N) DataFrame of synchronous PnL series for N trials."""
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")
    values = pnl.to_numpy(dtype=float)
    t_total, n_trials = values.shape
    if t_total < n_blocks:
        raise ValueError("more blocks than observations")

    blocks = np.array_split(values, n_blocks, axis=0)
    logits = []
    for combo in combinations(range(n_blocks), n_blocks // 2):
        train = np.vstack([blocks[i] for i in combo])
        test = np.vstack([blocks[i] for i in range(n_blocks) if i not in combo])
        r_train = performance_fn(train)
        r_test = performance_fn(test)
        n_star = int(np.argmax(r_train))
        ranks = rankdata(r_test, method="average")
        omega = ranks[n_star] / (n_trials + 1.0)
        omega = min(max(omega, 1e-12), 1 - 1e-12)
        logits.append(float(np.log(omega / (1 - omega))))

    logits = np.array(logits)
    pbo = float((logits < 0).mean())
    return PBOResult(
        pbo=pbo,
        logits=logits,
        n_combinations=len(logits),
        n_trials=n_trials,
    )
