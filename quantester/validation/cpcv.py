"""Combinatorial Purged Cross-Validation (AFML ch.7/12; notebook-verified).

Purging is defined by LABEL-INTERVAL OVERLAP, not fixed bar counts. A train
label [t_i0, t_i1] is dropped against a test interval [t_j0, t_j1] if any of:
  1. t_j0 <= t_i0 <= t_j1   (train starts inside test)
  2. t_j0 <= t_i1 <= t_j1   (train ends inside test)
  3. t_i0 <= t_j0 <= t_j1 <= t_i1 (train envelops test)
Embargo drops post-test train labels with t_j1 <= t_i0 <= t_j1 + h,
h = pct_embargo * T (de Prado recommends ~0.01T).

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


class PurgedKFold:
    """K-fold CV with label-overlap purging and post-test embargo."""

    def __init__(self, n_splits: int = 3, t1: pd.Series | None = None,
                 pct_embargo: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.t1 = t1  # label end times, aligned with X.index
        self.pct_embargo = pct_embargo

    def split(self, X: pd.DataFrame):
        n = len(X)
        indices = np.arange(n)
        folds = np.array_split(indices, self.n_splits)
        embargo = int(self.pct_embargo * n)

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
                # Embargo on training observations following the test set.
                embargoed = t1_test <= start_i <= t1_test + _as_offset(embargo, X)
                if not overlap and not embargoed:
                    train_idx.append(i)
            yield np.array(train_idx), test_idx


def _as_offset(embargo: int, X: pd.DataFrame):
    """Convert a bar-count embargo into the index's time units."""
    if isinstance(X.index, pd.DatetimeIndex):
        if len(X.index) > 1:
            median_delta = np.median(np.diff(X.index.values))
            return pd.Timedelta(median_delta) * embargo
        return pd.Timedelta(days=embargo)
    return embargo


class CombinatorialPurgedKFold:
    """CPCV: N groups, all C(N, N-k) train/test combinations with purging."""

    def __init__(self, n_groups: int = 6, k_test: int = 2,
                 t1: pd.Series | None = None, pct_embargo: float = 0.01):
        if not (1 <= k_test < n_groups):
            raise ValueError("require 1 <= k_test < n_groups")
        self.n_groups = n_groups
        self.k_test = k_test
        self.t1 = t1
        self.pct_embargo = pct_embargo

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
        embargo = int(self.pct_embargo * n)

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
                    embargoed = t1_test <= start_i <= t1_test + _as_offset(embargo, X)
                    if overlap or embargoed:
                        drop = True
                        break
                if not drop:
                    train_idx.append(i)
            yield np.array(sorted(train_idx)), np.sort(test_idx)
