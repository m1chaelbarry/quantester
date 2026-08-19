"""CPCV purging/embargo/path counts and CSCV PBO correctness."""

import numpy as np
import pandas as pd
import pytest

from quantester.validation.cpcv import CombinatorialPurgedKFold, PurgedKFold
from quantester.validation.pbo import pbo_cscv


def _X(n=100):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)


def test_purged_kfold_label_overlap():
    X = _X(30)
    horizon = pd.Timedelta(days=5)
    t1 = pd.Series(X.index + horizon, index=X.index)
    cv = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.0)
    for train_idx, test_idx in cv.split(X):
        t0 = X.index[test_idx[0]]
        t1_test = t1.iloc[test_idx[-1]]
        assert len(set(train_idx) & set(test_idx)) == 0
        for i in train_idx:
            start_i = X.index[i]
            end_i = t1.iloc[i]
            overlap = (
                (t0 <= start_i <= t1_test)
                or (t0 <= end_i <= t1_test)
                or (start_i <= t0 and end_i >= t1_test)
            )
            assert not overlap  # purged: no train label may overlap the test interval


def test_purged_kfold_embargo():
    X = _X(30)
    t1 = pd.Series(X.index, index=X.index)  # zero-horizon labels
    cv = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.1)  # h = 3 bars
    for train_idx, test_idx in cv.split(X):
        t1_test = X.index[test_idx[-1]]
        bar = pd.Timedelta(days=1)
        for i in train_idx:
            start_i = X.index[i]
            assert not (t1_test <= start_i <= t1_test + 3 * bar)


def test_purged_kfold_fixed_horizon_consistency():
    """With labels of fixed horizon F bars on a daily index, the F-1 preceding
    bars are purged (Cross-Ref 1.A's rule emerges as a special case)."""
    idx = pd.date_range("2024-01-01", periods=60, freq="D")  # daily: days == bars
    X = pd.DataFrame({"f": np.arange(60, dtype=float)}, index=idx)
    F = 6
    t1 = pd.Series(X.index + pd.Timedelta(days=F - 1), index=X.index)
    cv = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.0)
    for train_idx, test_idx in cv.split(X):
        before = train_idx[train_idx < test_idx[0]]
        if len(before):
            # Bars [test_start-F+1, test_start-1] are purged: F-1 preceding bars.
            assert before.max() == test_idx[0] - F


def test_cpcv_combinatorics_and_disjointness():
    X = _X(40)
    cpcv = CombinatorialPurgedKFold(n_groups=4, k_test=2, pct_embargo=0.0)
    assert cpcv.n_splits == 6
    assert cpcv.n_paths == 3  # phi[4,2] = (2/4) * C(4,2)
    splits = list(cpcv.split(X))
    assert len(splits) == 6
    for train_idx, test_idx in splits:
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(test_idx) == 20  # 2 groups of 10


def test_cpcv_n_paths_matches_binomial_identity():
    """phi[N,k] = C(N-1, k-1), not int((k/N) * C(N, N-k)) which truncates."""
    from math import comb

    cases = ((4, 2), (7, 3), (11, 6), (15, 11))
    for n_groups, k_test in cases:
        cpcv = CombinatorialPurgedKFold(n_groups=n_groups, k_test=k_test)
        assert cpcv.n_paths == comb(n_groups - 1, k_test - 1)
        # The float product is the known truncation failure at (11, 6).
        if (n_groups, k_test) == (11, 6):
            assert int(k_test / n_groups * cpcv.n_splits) != cpcv.n_paths


def _constructed_pnl():
    """Trial 0 shines in the first half and collapses in the second; the other
    trials are uniformly mediocre -> best in-sample trial fails out-of-sample."""
    n = 200
    cols = {}
    up = np.tile([0.010, 0.011], n // 4)
    down = np.tile([-0.010, -0.011], n // 4)
    cols["trial0"] = np.concatenate([up, down])
    for j in (1, 2, 3):
        cols[f"trial{j}"] = np.tile([-0.0001, -0.0002], n // 2)
    return pd.DataFrame(cols, index=pd.bdate_range("2024-01-01", periods=n))


def test_pbo_cscv_overfit_detection():
    result = pbo_cscv(_constructed_pnl(), n_blocks=4)
    assert result.n_combinations == 6
    assert np.isfinite(result.logits).all()
    # First combo trains on blocks {0,1}: trial0 best IS, worst OOS -> logit < 0.
    assert result.logits[0] < 0
    assert result.pbo >= 1 / 6
    assert not result.passes_gate  # overfit trials are rejected at the 0.10 gate
    assert 0.0 <= result.pbo <= 1.0


def test_pbo_requires_even_blocks():
    with pytest.raises(ValueError):
        pbo_cscv(_constructed_pnl(), n_blocks=5)
