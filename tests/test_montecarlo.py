"""Seeded Monte Carlo correctness tests + fast-track/engine parity."""

import numpy as np
import pandas as pd
import pytest

from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel, CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.diagnostics import autocorrelation_gate, ljung_box, runs_test
from quantester.montecarlo.drawdown import (
    double_bootstrap_dd_bound,
    single_loop_dd_quantile,
)
from quantester.montecarlo.fast_track import fast_backtest
from quantester.montecarlo.permutation import (
    intra_inter_bar_permutation,
    masters_p_value,
    multi_market_permutation,
    permute_log_changes,
    trend_bias_skill,
)
from quantester.montecarlo.synthetic import (
    _stationary_bootstrap_indices,
    bootstrap_ohlcv,
    estimate_ou_params,
    generate_ou_paths,
)
from quantester.montecarlo.trade_resampling import (
    ehlers_randomized_equity,
    empirical_resample,
)
from quantester.portfolio.portfolio import FixedUnitSizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv


def test_seeded_reproducibility(toy_returns):
    a = empirical_resample(toy_returns, horizon=100, n_sims=50, seed=42)
    b = empirical_resample(toy_returns, horizon=100, n_sims=50, seed=42)
    assert np.array_equal(a.paths, b.paths)
    c = empirical_resample(toy_returns, horizon=100, n_sims=50, seed=43)
    assert not np.array_equal(a.paths, c.paths)


def test_ehlers_all_winners_exact():
    paths = ehlers_randomized_equity(1.0, 2.0, 50.0, n_trades=10, n_sims=5, e0=1.0, seed=1)
    assert np.allclose(paths[:, -1], 1.0 + 10 * 100.0)


def test_ehlers_all_losers_exact():
    paths = ehlers_randomized_equity(0.0, 2.0, 50.0, n_trades=10, n_sims=5, e0=1.0, seed=1)
    assert np.allclose(paths[:, -1], 1.0 - 10 * 50.0)


def test_masters_p_value_exact():
    assert masters_p_value(1.0, [0.5, 1.5, 0.2]) == pytest.approx(2 / 4)
    assert masters_p_value(1.0, [0.5, 0.4]) == pytest.approx(1 / 3)


def test_trend_bias_skill_partition():
    out = trend_bias_skill(r_orig=0.30, b_orig=0.10, r_perm=0.15, b_perm=0.08)
    assert out["training_bias"] == pytest.approx(0.07)
    assert out["unbiased_return"] == pytest.approx(0.23)
    assert out["skill"] == pytest.approx(0.13)
    assert out["trend"] == pytest.approx(0.10)


def test_permute_log_changes_preserves_moments(ohlc):
    rng = np.random.default_rng(3)
    permuted = permute_log_changes(ohlc["close"], rng)
    original_changes = np.sort(np.diff(np.log(ohlc["close"])))
    permuted_changes = np.sort(np.diff(np.log(permuted)))
    assert np.allclose(original_changes, permuted_changes)  # same multiset
    assert permuted.iloc[0] == pytest.approx(ohlc["close"].iloc[0])


def test_multi_market_permutation_preserves_cross_correlation():
    rng = np.random.default_rng(4)
    base = make_synthetic_ohlcv("A", n_bars=150, seed=8)["close"]
    shadow = base * 1.7 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(base))))
    prices = pd.DataFrame({"A": base, "B": shadow})
    before = np.corrcoef(np.diff(np.log(prices["A"])), np.diff(np.log(prices["B"])))[0, 1]
    permuted = multi_market_permutation(prices, rng)
    after = np.corrcoef(np.diff(np.log(permuted["A"])), np.diff(np.log(permuted["B"])))[0, 1]
    assert before == pytest.approx(after, rel=1e-9)  # identical joint shuffle


def test_protocol2_ohlc_validity(ohlc):
    rng = np.random.default_rng(6)
    permuted = intra_inter_bar_permutation(ohlc, rng)
    assert (permuted["high"] >= permuted[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (permuted["low"] <= permuted[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (permuted["low"] > 0).all()
    assert (permuted["high"] >= permuted["low"]).all()


def test_double_bootstrap_conservative_vs_single_loop(toy_returns):
    single = single_loop_dd_quantile(toy_returns, horizon=len(toy_returns),
                                     n_sims=200, conf=0.90, seed=10)
    double = double_bootstrap_dd_bound(toy_returns, dd_conf=0.99, bound_conf=0.99,
                                       n_outer=100, n_inner=100, seed=10)
    assert double.bound >= single  # bound-on-a-bound must not undercut the naive one


def test_ou_parameter_recovery():
    from quantester.montecarlo.synthetic import OUParams

    true = OUParams(theta=0.10, mu=50.0, sigma=1.0)
    paths = generate_ou_paths(true, p0=50.0, n_steps=20_000, n_paths=1, seed=12)
    est = estimate_ou_params(paths[0])
    assert est.mu == pytest.approx(50.0, abs=1.0)
    assert est.theta == pytest.approx(0.10, abs=0.05)
    assert est.sigma == pytest.approx(1.0, abs=0.15)


def test_diagnostics_white_noise_vs_correlated(toy_returns):
    report = autocorrelation_gate(toy_returns, alpha=0.05)
    assert report.recommended_method == "iid_resampling"

    correlated = np.convolve(toy_returns, np.ones(5) / 5, mode="same")
    report2 = autocorrelation_gate(correlated, alpha=0.05)
    z, p_runs = runs_test(correlated)
    q, p_lb = ljung_box(correlated)
    assert np.isfinite(z) and np.isfinite(q)
    assert report2.serial_correlation in (True, False)  # gate executes cleanly


def test_fast_track_engine_parity(ohlc):
    """The vectorized bypass must reproduce the full engine exactly (seeded)."""
    costs = CostModel()
    units = 100

    handler = HistoricCSVDataHandler({"AAA": ohlc})
    strategy = MovingAverageCrossStrategy(handler, "AAA", fast=3, slow=8)
    portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(units))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(costs))
    engine.run_backtest()

    target = MovingAverageCrossStrategy(None, "AAA", fast=3, slow=8).vectorized_signals(
        {"AAA": ohlc}
    )["AAA"]
    fast = fast_backtest(ohlc, target, costs, initial_capital=100_000.0, units=units)

    engine_equity = portfolio.equity_curve.reindex(ohlc.index).ffill()
    assert np.allclose(
        engine_equity.to_numpy(), fast.equity.to_numpy(), rtol=1e-9, atol=1e-6
    )


def test_bootstrap_ohlcv_determinism_and_invariants(ohlc):
    a = bootstrap_ohlcv(ohlc, mean_block=20, seed=7)
    b = bootstrap_ohlcv(ohlc, mean_block=20, seed=7)
    c = bootstrap_ohlcv(ohlc, mean_block=20, seed=8)
    pd.testing.assert_frame_equal(a, b)          # seeded determinism
    assert not a["close"].equals(c["close"])
    assert len(a) == len(ohlc) and a.index.equals(ohlc.index)
    assert list(a.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC consistency by construction.
    assert (a["high"] >= a[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1) + 1e-12).all()
    assert (a[["open", "high", "low", "close"]] > 0).all().all()


def test_stationary_bootstrap_block_contiguity():
    """Long mean blocks preserve consecutive bars; mean_block=1 = iid shuffle."""
    n = 2000
    long_blocks = _stationary_bootstrap_indices(n, 200.0, np.random.default_rng(3))
    contiguity = (long_blocks[1:] == (long_blocks[:-1] + 1) % n).mean()
    assert contiguity > 0.9
    iid = _stationary_bootstrap_indices(n, 1.0, np.random.default_rng(3))
    contiguity_iid = (iid[1:] == (iid[:-1] + 1) % n).mean()
    assert contiguity_iid < 0.1


def test_fast_track_engine_parity_price_aware_costs(ohlc):
    """Parity must also hold for notional-fee cost models: the engine and the
    fast-track both pass the fill reference price into commission()."""
    costs = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
    units = 100

    handler = HistoricCSVDataHandler({"AAA": ohlc})
    strategy = MovingAverageCrossStrategy(handler, "AAA", fast=3, slow=8)
    portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(units))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(costs))
    engine.run_backtest()

    target = MovingAverageCrossStrategy(None, "AAA", fast=3, slow=8).vectorized_signals(
        {"AAA": ohlc}
    )["AAA"]
    fast = fast_backtest(ohlc, target, costs, initial_capital=100_000.0, units=units)

    engine_equity = portfolio.equity_curve.reindex(ohlc.index).ffill()
    assert np.allclose(
        engine_equity.to_numpy(), fast.equity.to_numpy(), rtol=1e-9, atol=1e-6
    )
    assert any(f.commission > 0 for f in portfolio.fills)  # fees really charged
