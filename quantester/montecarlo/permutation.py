"""Monte Carlo Permutation Testing (MCPT) engine (MC Report section 3).

Shuffling log price changes destroys the chronological patterns a strategy
exploits while preserving the data's exact statistical moments. If the strategy
is capturing genuine patterns, performance on the original series must
significantly exceed the permuted distribution.

- Permutation p-value per Masters' exact algorithm: count starts at 1,
  incremented whenever a permuted run's optimized performance >= the original;
  p = count / n_reps. Gate at p < 0.05.
- Masters' return partition (notebook-verified):
    Bias       = R_perm - B_perm
    R_unbiased = R_orig - Bias = Skill + Trend
    Skill      = R_unbiased - B_orig        (Trend = B_orig)
  Benchmarks are recomputed on permuted paths, not assumed zero.
- Protocol I: offset-synchronized multi-market permutation -- identical shuffle
  across all assets preserves cross-sectional correlations.
- Protocol II: intra-bar (H/O, L/O, C/O jointly) vs inter-bar (O/prev-C gaps)
  split permutation with physically valid OHLC reconstruction (MC Report
  section 3.3; not covered by the notebook, implemented per the report).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def permute_log_changes(prices: pd.Series, rng: np.random.Generator,
                        offset: int = 0) -> pd.Series:
    """Shuffle log price changes after `offset`; rebuild from the base price."""
    log_p = np.log(prices.to_numpy(dtype=float))
    changes = np.diff(log_p)
    head, tail = changes[:offset], changes[offset:].copy()
    rng.shuffle(tail)
    rebuilt_log = log_p[0] + np.concatenate([[0.0], np.cumsum(np.concatenate([head, tail]))])
    return pd.Series(np.exp(rebuilt_log), index=prices.index)


def multi_market_permutation(prices: pd.DataFrame, rng: np.random.Generator,
                             offset: int = 0) -> pd.DataFrame:
    """Protocol I: identical shuffle indices applied to every market's log changes."""
    log_p = np.log(prices.to_numpy(dtype=float))
    changes = np.diff(log_p, axis=0)
    n_changes = changes.shape[0]
    active = changes[offset:]
    perm_idx = rng.permutation(active.shape[0])
    shuffled_active = active[perm_idx]
    rebuilt_changes = np.vstack([changes[:offset], shuffled_active])
    out = np.exp(np.vstack([log_p[:1], log_p[:1] + np.cumsum(rebuilt_changes, axis=0)]))
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def intra_inter_bar_permutation(ohlc: pd.DataFrame,
                                rng: np.random.Generator) -> pd.DataFrame:
    """Protocol II: split intra-bar moves from inter-bar gaps; shuffle the two
    groups independently; reconstruct physically valid OHLC bars.

    Intra-bar records (H/O, L/O, C/O) are shuffled JOINTLY so each reconstructed
    bar keeps a coherent high/low/close geometry; inter-bar gaps (O_t / C_{t-1})
    are shuffled independently of the intra records.
    """
    o = ohlc["open"].to_numpy(dtype=float)
    h = ohlc["high"].to_numpy(dtype=float)
    l = ohlc["low"].to_numpy(dtype=float)
    c = ohlc["close"].to_numpy(dtype=float)
    n = len(ohlc)

    intra = np.column_stack([np.log(h / o), np.log(l / o), np.log(c / o)])
    gaps = np.log(o[1:] / c[:-1])

    intra_perm = intra[rng.permutation(n)]
    gaps_perm = gaps[rng.permutation(n - 1)] if n > 1 else gaps

    out_o = np.empty(n)
    out_h = np.empty(n)
    out_l = np.empty(n)
    out_c = np.empty(n)
    out_c[0] = c[0]  # anchor at the original base close
    for t in range(n):
        if t == 0:
            out_o[t] = o[0]
        else:
            out_o[t] = out_c[t - 1] * np.exp(gaps_perm[t - 1])
        rh, rl, rc = intra_perm[t]
        out_h[t] = out_o[t] * np.exp(max(rh, rc, 0.0))
        out_l[t] = out_o[t] * np.exp(min(rl, rc, 0.0))
        out_c[t] = out_o[t] * np.exp(rc)

    return pd.DataFrame(
        {"open": out_o, "high": out_h, "low": out_l, "close": out_c,
         "volume": ohlc["volume"].to_numpy() if "volume" in ohlc else 0.0},
        index=ohlc.index,
    )


def masters_p_value(original_performance: float, permuted_performances) -> float:
    """count starts at 1; +1 whenever permuted >= original; p = count / n_reps."""
    permuted = np.asarray(permuted_performances, dtype=float)
    count = 1 + int((permuted >= original_performance).sum())
    return count / (len(permuted) + 1)


@dataclass
class MCPTResult:
    p_value: float
    original_performance: float
    permuted_performances: np.ndarray
    n_reps: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def permutation_test(data, optimizer, n_reps: int = 1000, seed: int | None = None,
                     permute_fn=None) -> MCPTResult:
    """TRAIN PERMUTED: retrain the optimizer from scratch on each permuted path.

    optimizer(data) -> float performance metric (higher is better). Drives the
    vectorized fast-track, never 10,000 event-loop re-runs (Cross-Ref 3.2).
    permute_fn(data, rng) defaults to permute_log_changes for a price Series.

    ``n_reps`` is Masters' total trial count **including the original**: the
    engine runs ``n_reps - 1`` permutations and forms
    ``p = (1 + #{perm >= orig}) / n_reps``. Requesting ``n_reps`` shuffles
    would inflate the denominator by one and break the exact p-value.
    """
    if n_reps < 2:
        raise ValueError("n_reps must be >= 2 (original + at least one permutation)")
    if permute_fn is None:
        permute_fn = permute_log_changes
    rng = np.random.default_rng(seed)
    original = float(optimizer(data))
    permuted = np.array(
        [float(optimizer(permute_fn(data, rng))) for _ in range(n_reps - 1)]
    )
    return MCPTResult(
        p_value=masters_p_value(original, permuted),
        original_performance=original,
        permuted_performances=permuted,
        n_reps=n_reps,
    )


def trend_bias_skill(r_orig: float, b_orig: float, r_perm: float,
                     b_perm: float) -> dict:
    """Masters' partition of total return (notebook-verified)."""
    bias = r_perm - b_perm
    unbiased = r_orig - bias
    return {
        "trend": b_orig,
        "training_bias": bias,
        "unbiased_return": unbiased,
        "skill": unbiased - b_orig,
    }
