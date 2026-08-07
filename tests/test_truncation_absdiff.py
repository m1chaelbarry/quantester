"""Regression tests for truncation absolute-difference diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantester.validation.truncation import run_truncation_test


def _make_run_fn(full: pd.DataFrame, truncated: pd.DataFrame):
    def run_fn(truncate_last):
        return full if truncate_last is None else truncated

    return run_fn


def test_truncation_passes_on_equality():
    idx = pd.bdate_range("2024-01-01", periods=10, tz="UTC")
    full = pd.DataFrame({"AAA": np.arange(10, dtype=float), "BBB": 1.0}, index=idx)
    truncated = full.iloc[:-3]
    result = run_truncation_test(_make_run_fn(full, truncated), n_truncated=3)
    assert result.passed
    assert result.rows_compared == 7
    assert result.mismatches == []


def test_truncation_catches_full_greater_than_truncated():
    idx = pd.bdate_range("2024-01-01", periods=8, tz="UTC")
    full = pd.DataFrame({"AAA": [0, 1, 2, 3, 4, 5, 6, 7]}, index=idx, dtype=float)
    truncated = full.iloc[:-2].copy()
    truncated.iloc[2, 0] = 1.0  # full=2 > truncated=1
    result = run_truncation_test(_make_run_fn(full, truncated), n_truncated=2, atol=1e-9)
    assert not result.passed
    assert result.mismatches[0]["full"] == 2.0
    assert result.mismatches[0]["truncated"] == 1.0


def test_truncation_catches_full_less_than_truncated():
    """The pre-fix directional bug: (full - truncated) > atol misses this."""
    idx = pd.bdate_range("2024-01-01", periods=8, tz="UTC")
    full = pd.DataFrame({"AAA": [100.0] * 8}, index=idx)
    truncated = full.iloc[:-2].copy()
    truncated.iloc[3, 0] = 101.0  # full=100 < truncated=101
    result = run_truncation_test(_make_run_fn(full, truncated), n_truncated=2, atol=1e-9)
    assert not result.passed
    m = result.mismatches[0]
    assert m["full"] == 100.0
    assert m["truncated"] == 101.0
    assert m["abs_diff"] == pytest.approx(1.0)


def test_truncation_tolerance_boundary():
    idx = pd.bdate_range("2024-01-01", periods=6, tz="UTC")
    full = pd.DataFrame({"AAA": [1.0] * 6}, index=idx)
    trunc_within = full.iloc[:-1].copy()
    trunc_within.iloc[0, 0] = 1.0 + 1e-10
    assert run_truncation_test(
        _make_run_fn(full, trunc_within), n_truncated=1, atol=1e-9
    ).passed

    trunc_outside = full.iloc[:-1].copy()
    trunc_outside.iloc[0, 0] = 1.0 + 1e-8
    assert not run_truncation_test(
        _make_run_fn(full, trunc_outside), n_truncated=1, atol=1e-9
    ).passed


def test_truncation_multi_asset_multi_timestamp():
    idx = pd.bdate_range("2024-01-01", periods=12, tz="UTC")
    full = pd.DataFrame(
        {
            "AAA": np.linspace(0, 1, 12),
            "BBB": np.linspace(1, 0, 12),
            "CCC": np.ones(12),
        },
        index=idx,
    )
    truncated = full.iloc[:-4].copy()
    truncated.loc[idx[1], "BBB"] = -0.5
    truncated.loc[idx[5], "CCC"] = 2.0
    result = run_truncation_test(_make_run_fn(full, truncated), n_truncated=4)
    assert not result.passed
    symbols = {m["symbol"] for m in result.mismatches}
    assert "BBB" in symbols and "CCC" in symbols
    assert len({m["timestamp"] for m in result.mismatches}) >= 2
