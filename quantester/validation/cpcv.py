"""Combinatorial Purged Cross-Validation (AFML ch.7/12).

Purging is defined by LABEL-INTERVAL OVERLAP, not fixed bar counts
(notebook-verified geometry). A train label [t_i0, t_i1] is dropped against
a test interval [t_j0, t_j1] if any of:
  1. t_j0 <= t_i0 <= t_j1   (train starts inside test)
  2. t_j0 <= t_i1 <= t_j1   (train ends inside test)
  3. t_i0 <= t_j0 <= t_j1 <= t_i1 (train envelops test)

Embargo LENGTH (ruling D8, ticket 24 — NOT covered by the notebook;
implemented from Masters *Assessing* ch. 1 / TTMTS): an integer bar window
of ``min(lookback, lookahead) - 1`` positions strictly after the test
block's label end. Embargo is counted in INDEX POSITIONS of X — a calendar
gap never stretches it (the old ``pct_embargo * T`` x median-Δt smear is
removed). ``pct_embargo`` survives only as an explicit override for de
Prado ~0.01T research (floored to integer bars); with no knobs given, the
embargo defaults to 0 bars rather than a silent 0.01T.

CPCV: T observations partitioned into N groups without shuffling; test sets of
k groups yield C(N, N-k) splits and phi[N,k] = (k/N) * C(N, N-k) unique paths.

(The F-1 / B+F-1 fixed-bar windows from Cross-Ref section 1.A are retained only
as a derived guard for non-labeled walk-forward engine runs.)
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def _resolve_embargo_bars(embargo_bars, lookback, lookahead, pct_embargo,
                          n: int) -> int:
    """Embargo length in integer bar positions (D8, ticket 24).

    Priority: explicit ``embargo_bars`` → ``min(lookback, lookahead) - 1``
    (single horizon: that value minus 1) → explicit ``pct_embargo`` floored
    to bars (de Prado ~0.01T research override) → 0 (documented product
    default: no silent 0.01T).
    """
    if embargo_bars is not None:
        bars = int(embargo_bars)
        if bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        return bars
    if lookback is not None or lookahead is not None:
        horizons = [int(v) for v in (lookback, lookahead) if v is not None]
        if any(v < 1 for v in horizons):
            raise ValueError("lookback/lookahead must be >= 1 bar")
        return max(min(horizons) - 1, 0)
    if pct_embargo is not None:
        if not 0.0 <= pct_embargo < 1.0:
            raise ValueError("pct_embargo must lie in [0, 1)")
        return int(pct_embargo * n)
    return 0


def _embargo_ref_pos(X: pd.DataFrame, t1_test) -> int:
    """Position of the last index entry at or before ``t1_test`` (the test
    block's label end); embargo covers the positions strictly after it."""
    return int(X.index.searchsorted(t1_test, side="right")) - 1


class PurgedKFold:
    """K-fold CV with label-overlap purging and post-test embargo.

    Embargo knobs (D8): ``embargo_bars`` (direct integer positions), or
    ``lookback``/``lookahead`` (embargo = min(B, F) - 1 bars; single-sided
    uses the given one minus 1), or explicit ``pct_embargo`` (legacy de
    Prado fraction of T, floored to bars). Default: 0 bars.
    """

    def __init__(self, n_splits: int = 3, t1: pd.Series | None = None,
                 pct_embargo: float | None = None, *,
                 embargo_bars: int | None = None,
                 lookback: int | None = None,
                 lookahead: int | None = None):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.t1 = t1  # label end times, aligned with X.index
        self.pct_embargo = pct_embargo
        self.embargo_bars = embargo_bars
        self.lookback = lookback
        self.lookahead = lookahead

    def split(self, X: pd.DataFrame):
        n = len(X)
        indices = np.arange(n)
        folds = np.array_split(indices, self.n_splits)
        embargo = _resolve_embargo_bars(
            self.embargo_bars, self.lookback, self.lookahead,
            self.pct_embargo, n,
        )

        for test_idx in folds:
            test_start_pos = test_idx[0]
            test_end_pos = test_idx[-1]
            test_set = set(test_idx)
            t0 = X.index[test_start_pos]
            t1_test = (
                self.t1.iloc[test_end_pos]
                if self.t1 is not None
                else X.index[test_end_pos]
            )
            ref_pos = _embargo_ref_pos(X, t1_test) if embargo else None

            train_idx = []
            for i in indices:
                if i in test_set:
                    continue
                start_i = X.index[i]
                end_i = self.t1.iloc[i] if self.t1 is not None else start_i

                # Label-overlap purge (three de Prado conditions).
                overlap = (
                    (t0 <= start_i <= t1_test)
                    or (t0 <= end_i <= t1_test)
                    or (start_i <= t0 and end_i >= t1_test)
                )
                # Embargo: integer positions strictly after the label end.
                embargoed = (
                    ref_pos is not None and ref_pos < i <= ref_pos + embargo
                )
                if not overlap and not embargoed:
                    train_idx.append(i)
            yield np.array(train_idx), test_idx


class CombinatorialPurgedKFold:
    """CPCV: N groups, all C(N, N-k) train/test combinations with purging.

    Embargo knobs identical to ``PurgedKFold`` (D8): ``embargo_bars`` /
    ``lookback`` + ``lookahead`` / explicit ``pct_embargo``; default 0 bars.
    """

    def __init__(self, n_groups: int = 6, k_test: int = 2,
                 t1: pd.Series | None = None, pct_embargo: float | None = None,
                 *,
                 embargo_bars: int | None = None,
                 lookback: int | None = None,
                 lookahead: int | None = None):
        if not (1 <= k_test < n_groups):
            raise ValueError("require 1 <= k_test < n_groups")
        self.n_groups = n_groups
        self.k_test = k_test
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.embargo_bars = embargo_bars
        self.lookback = lookback
        self.lookahead = lookahead

    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_groups - self.k_test)

    @property
    def n_paths(self) -> int:
        """phi[N,k] = C(N-1, k-1) = (k/N) * C(N, N-k) unique backtest paths.

        The binomial identity is exact. ``int((k/N) * n_splits)`` truncates
        when the float product sits just below an integer (e.g. N=11, k=6).
        """
        return comb(self.n_groups - 1, self.k_test - 1)

    def split(self, X: pd.DataFrame):
        n = len(X)
        indices = np.arange(n)
        groups = np.array_split(indices, self.n_groups)
        embargo = _resolve_embargo_bars(
            self.embargo_bars, self.lookback, self.lookahead,
            self.pct_embargo, n,
        )

        for test_group_ids in combinations(range(self.n_groups), self.k_test):
            test_idx = np.concatenate([groups[g] for g in test_group_ids])
            train_pool = np.concatenate(
                [groups[g] for g in range(self.n_groups) if g not in test_group_ids]
            )

            train_idx = []
            for i in train_pool:
                start_i = X.index[i]
                end_i = self.t1.iloc[i] if self.t1 is not None else start_i
                drop = False
                for g in test_group_ids:
                    grp = groups[g]
                    t0 = X.index[grp[0]]
                    t1_test = (
                        self.t1.iloc[grp[-1]]
                        if self.t1 is not None
                        else X.index[grp[-1]]
                    )
                    overlap = (
                        (t0 <= start_i <= t1_test)
                        or (t0 <= end_i <= t1_test)
                        or (start_i <= t0 and end_i >= t1_test)
                    )
                    ref_pos = _embargo_ref_pos(X, t1_test) if embargo else None
                    embargoed = (
                        ref_pos is not None and ref_pos < i <= ref_pos + embargo
                    )
                    if overlap or embargoed:
                        drop = True
                        break
                if not drop:
                    train_idx.append(i)
            yield np.array(sorted(train_idx)), np.sort(test_idx)
