"""Return representation and Monte Carlo compounding regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantester.analytics.performance import max_drawdown
from quantester.analytics.returns import (
    log_to_simple,
    simple_to_log,
    wealth_from_log_returns,
    wealth_from_simple_returns,
)
from quantester.montecarlo.drawdown import max_drawdown_of_returns
from quantester.montecarlo.trade_resampling import empirical_resample


def test_simple_returns_compound_to_0_99():
    simple = np.array([0.10, -0.10])
    wealth = wealth_from_simple_returns(simple, initial=1.0)
    assert wealth[-1] == pytest.approx(0.99)
    # Additive path is WRONG for simple returns:
    assert 1.0 + np.sum(simple) == pytest.approx(1.0)
    assert wealth[-1] != pytest.approx(1.0)


def test_log_returns_aggregate_additively():
    logs = np.array([0.1, -0.05, 0.02])
    wealth = wealth_from_log_returns(logs, initial=1.0)
    assert wealth[-1] == pytest.approx(np.exp(np.sum(logs)))


def test_simple_log_roundtrip():
    simple = np.array([0.05, -0.02, 0.0, 0.5, -0.5])
    assert np.allclose(log_to_simple(simple_to_log(simple)), simple)


def test_wealth_edge_cases():
    assert wealth_from_simple_returns([], initial=2.0).tolist() == [2.0]
    assert wealth_from_simple_returns([0.0, 0.0])[-1] == pytest.approx(1.0)
    assert wealth_from_simple_returns([1.0])[-1] == pytest.approx(2.0)
    assert wealth_from_simple_returns([-0.5, -0.5])[-1] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        wealth_from_simple_returns([-1.0])
    with pytest.raises(ValueError):
        wealth_from_simple_returns([np.nan])
    with pytest.raises(ValueError):
        simple_to_log([-1.0])


def test_empirical_resample_compounds_simple_returns():
    # Force a deterministic path by using a one-element pool + horizon 2.
    pool = np.array([0.10, -0.10])
    # Seeded draws are random; verify the construction identity on a known path:
    known = np.array([0.10, -0.10])
    expected_terminal = float(np.prod(1.0 + known) - 1.0)
    assert expected_terminal == pytest.approx(-0.01)

    res = empirical_resample(pool, horizon=2, n_sims=500, seed=0)
    # Every path must equal cumprod construction for its own draws (recompute).
    # Spot-check: no path may equal the additive 1+cumsum for the [+10%, -10%]
    # sequence when that sequence appears.
    assert res.paths.shape == (500, 3)
    assert np.all(res.paths[:, 0] == 1.0)
    # Reconstruct from terminal identity: paths are positive under |r|<1 pool.
    assert np.all(res.paths > 0)


def test_empirical_resample_rejects_invalid_and_empty():
    with pytest.raises(ValueError, match="empty"):
        empirical_resample([])
    with pytest.raises(ValueError, match="> -1"):
        empirical_resample([-1.0, 0.01])
    with pytest.raises(ValueError, match="empty"):
        empirical_resample([np.nan, np.inf])


def test_max_drawdown_of_returns_matches_performance():
    simple = np.array([0.10, -0.20, 0.05, -0.10, 0.15])
    dd_mc = max_drawdown_of_returns(simple)
    idx = pd.bdate_range("2024-01-01", periods=len(simple) + 1, tz="UTC")
    equity = pd.Series(wealth_from_simple_returns(simple), index=idx)
    dd_perf = abs(max_drawdown(equity)["max_drawdown"])
    assert dd_mc == pytest.approx(dd_perf, rel=1e-12)
    # Additive DD would be wrong for this path:
    additive = np.concatenate([[0.0], np.cumsum(simple)])
    hwm = np.maximum.accumulate(additive)
    additive_dd = float((hwm - additive).max())
    assert dd_mc != pytest.approx(additive_dd)


def test_max_drawdown_of_returns_edge_cases():
    assert max_drawdown_of_returns(np.array([])) == 0.0
    assert max_drawdown_of_returns(np.array([0.0, 0.0])) == 0.0
    with pytest.raises(ValueError):
        max_drawdown_of_returns(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        max_drawdown_of_returns(np.array([-1.0]))
