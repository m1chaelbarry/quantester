"""Ledger accounting, sizing math, spectral risk, margin monitor."""

import numpy as np
import pandas as pd
import pytest

from quantester.events import BUY, SELL, FillEvent
from quantester.portfolio.portfolio import (
    FixedUnitSizer,
    PercentEquitySizer,
    PortfolioManager,
)
from quantester.portfolio.risk import (
    MarginMonitor,
    spectral_risk_attribution,
    stabilized_covariance,
)
from quantester.portfolio.sizing import (
    hpr,
    kakushadze_effective_returns,
    kelly_fraction,
    kelly_gaussian,
    optimal_f,
    twr,
    volatility_parity_weights,
)

TS = pd.Timestamp("2024-01-02")


def _portfolio():
    class _Handler:
        symbols = ["AAA"]

    return PortfolioManager(_Handler(), 100_000.0)


def test_ledger_accounting_round_trip():
    portfolio = _portfolio()
    portfolio.update_from_fill(FillEvent(TS, "AAA", 100, BUY, 10.0, 0.0, 0.0))
    assert portfolio.cash == pytest.approx(100_000 - 1_000)
    assert portfolio.positions["AAA"] == 100

    portfolio.update_from_fill(FillEvent(TS, "AAA", 100, SELL, 12.0, 0.0, 0.0))
    assert portfolio.cash == pytest.approx(100_000 + 200)
    assert "AAA" not in portfolio.positions
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0]["pnl"] == pytest.approx(200.0)


def test_ledger_costs_deducted():
    portfolio = _portfolio()
    fill = FillEvent(TS, "AAA", 100, BUY, 10.0, 1.5, 0.5)
    portfolio.update_from_fill(fill)
    # Commission hits cash; slippage is embedded in fill_price, recorded only.
    assert portfolio.cash == pytest.approx(100_000 - 1_000 - 1.5)
    assert fill.slippage_cost == 0.5  # phi_t still reported for cost analytics


def test_sizers():
    class _Signal:
        signal_type = "LONG"
        strength = 0.5

    assert FixedUnitSizer(200)(_Signal(), None, 10.0) == 100.0
    portfolio = _portfolio()
    portfolio.last_prices["AAA"] = 10.0
    target = PercentEquitySizer(0.5)(_Signal(), portfolio, 10.0)
    assert target == pytest.approx(100_000 * 0.5 * 0.5 / 10.0)


def test_kelly():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)
    assert kelly_gaussian(0.001, 0.0004) == pytest.approx(2.5)
    assert kelly_gaussian(0.001, 0.0) == 0.0


def test_optimal_f_twr_maximizes():
    trades = np.array([2.0, -1.0, 3.0, -1.0, 1.5, -0.5])
    worst = trades.min() * 1.5  # gap-stressed below nominal stop
    f_star = optimal_f(trades, worst_loss=worst)
    assert 0.0 < f_star <= 1.0
    assert twr(trades, f_star, worst) >= twr(trades, f_star + 0.05, worst) - 1e-9
    assert twr(trades, f_star, worst) >= twr(trades, max(f_star - 0.05, 0.0), worst) - 1e-9
    # HPR form: HPR_i = 1 + f(-Trade_i / WorstLoss)
    assert hpr(np.array([2.0]), 0.5, -1.5)[0] == pytest.approx(1 + 0.5 * 2 / 1.5)
    # Gap-stress bound: the EFFECTIVE bet fraction f*/|WorstLoss| never grows
    # when the worst loss is stressed below the nominal stop.
    f_stressed = optimal_f(trades, worst_loss=trades.min() * 1.5)
    f_plain = optimal_f(trades, worst_loss=trades.min() * 1.0)
    assert f_stressed / abs(trades.min() * 1.5) <= f_plain / abs(trades.min()) + 1e-9


def test_kakushadze_effective_returns():
    expected = pd.Series([0.05, -0.02, 0.01], index=["A", "B", "C"])
    costs = pd.Series([0.03, 0.01, 0.02], index=["A", "B", "C"])
    eff = kakushadze_effective_returns(expected, costs)
    assert eff["A"] == pytest.approx(0.02)
    assert eff["B"] == pytest.approx(-0.01)
    assert eff["C"] == 0.0  # marginal edge fully consumed by costs


def test_volatility_parity():
    cov = pd.DataFrame(
        np.diag([0.04, 0.01, 0.09]), index=["A", "B", "C"], columns=["A", "B", "C"]
    )
    w = volatility_parity_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert w["B"] > w["A"] > w["C"]  # lowest vol gets most weight


def test_spectral_risk_on_singular_covariance():
    rng = np.random.default_rng(9)
    n_obs, n_assets = 15, 6  # assets near/exceeding observations -> raw cov singular
    base = rng.normal(0, 0.01, size=(n_obs, 4))
    returns = pd.DataFrame(
        np.column_stack([base, base[:, 0] + base[:, 1], base[:, 2] - base[:, 3]]),
        columns=list("ABCDEF"),
    )
    raw_rank = np.linalg.matrix_rank(np.cov(returns.to_numpy(), rowvar=False))
    assert raw_rank < n_assets  # genuinely singular

    attribution = spectral_risk_attribution(returns)
    assert np.isfinite(attribution["eigenvalue"]).all()
    assert np.isfinite(attribution["risk_share"]).all()
    assert attribution["risk_share"].sum() == pytest.approx(1.0, rel=1e-6)

    shrunk = stabilized_covariance(returns)
    assert np.linalg.matrix_rank(shrunk.to_numpy()) == n_assets  # stabilized


def test_margin_monitor_liquidation():
    monitor = MarginMonitor(max_leverage=2.0, liquidation_fraction=0.5)
    assert monitor.is_breach(equity=100_000, gross_exposure=250_000)
    assert not monitor.is_breach(equity=100_000, gross_exposure=150_000)
    targets = monitor.liquidation_targets({"AAA": 100, "BBB": -50})
    assert targets == {"AAA": 50.0, "BBB": -25.0}
