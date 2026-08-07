"""Ledger accounting, sizing math, spectral risk, margin monitor, breaker."""

import numpy as np
import pandas as pd
import pytest

from quantester.events import (
    BUY,
    CANCEL_ORDER,
    EXIT,
    LONG,
    MARKET_ORDER,
    MOC_ORDER,
    SELL,
    FillEvent,
    MarketEvent,
    SignalEvent,
)
from quantester.portfolio.portfolio import (
    FixedUnitSizer,
    PercentEquitySizer,
    PortfolioManager,
)
from quantester.portfolio.risk import (
    DailyDrawdownBreaker,
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


# ------------------------------------------------------- daily drawdown breaker

D1, D2, D3 = (
    pd.Timestamp("2024-01-02"),
    pd.Timestamp("2024-01-03"),
    pd.Timestamp("2024-01-04"),
)


class _Queue(list):
    def put(self, item):
        self.append(item)


def _valuation(ts, close):
    bar = pd.Series({"open": close, "high": close, "low": close,
                     "close": close, "volume": 1e6})
    return MarketEvent(ts, bars={"AAA": bar}, phase="close")


def test_breaker_threshold_and_rollover_reset():
    breaker = DailyDrawdownBreaker(max_intraday_dd=0.045)
    assert not breaker.update(D1, 100_000.0)      # seeds the baseline
    assert not breaker.update(D1, 95_600.0)       # 4.4% < 4.5%: holds
    assert breaker.update(D1, 95_500.0)           # 4.5%: trips (>= boundary)
    assert breaker.halted and breaker.triggered_count == 1
    assert not breaker.update(D1, 94_000.0)       # already halted: no re-fire
    assert breaker.triggered_count == 1
    assert not breaker.update(D2, 94_000.0)       # rollover resets the halt
    assert not breaker.halted
    # New day baseline is the carried 94k: another 4.5%+ slide re-trips.
    assert breaker.update(D2, 89_700.0)
    assert breaker.triggered_count == 2


def test_breaker_liquidates_cancels_and_blocks_entries():
    from quantester.data.streaming import StreamingDataHandler

    idx = pd.bdate_range("2024-01-02", periods=4)
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
         "volume": 1e6},
        index=idx,
    )
    handler = StreamingDataHandler({"AAA": df})
    breaker = DailyDrawdownBreaker(max_intraday_dd=0.045)
    portfolio = PortfolioManager(handler, 100_000.0, drawdown_breaker=breaker)
    portfolio.update_from_fill(FillEvent(D1, "AAA", 100, BUY, 100.0, 0.0, 0.0))
    # cash 90k, long 100 @ 100

    queue = _Queue()
    portfolio.update_portfolio_valuation(_valuation(D1, 100.0), queue)
    assert not breaker.halted and len(queue) == 0

    # Day 2: close 50 -> equity 95k -> 5% intraday drawdown trips the breaker.
    portfolio.update_portfolio_valuation(_valuation(D2, 50.0), queue)
    assert breaker.halted
    cancels = [o for o in queue if o.order_type == CANCEL_ORDER]
    liquidations = [o for o in queue if o.order_type == MARKET_ORDER]
    assert len(cancels) == 1 and cancels[0].symbol == "AAA"
    assert len(liquidations) == 1
    assert liquidations[0].direction == SELL
    assert liquidations[0].quantity == pytest.approx(100.0)
    assert liquidations[0].earliest_fill_time == idx[2]  # next bar's open

    # While halted ALL signal flow is suspended (entries and exits alike);
    # the breaker's own parked liquidation order handles the flattening.
    n = len(queue)
    portfolio.update_from_signal(SignalEvent(D2, "AAA", LONG, strength=1.0), queue)
    portfolio.update_from_signal(
        SignalEvent(D2, "AAA", EXIT, cancel_orders=True), queue
    )
    assert len(queue) == n

    # Day 3 rollover: halt clears, entries resume (reference price available).
    portfolio.update_portfolio_valuation(_valuation(D3, 50.0), queue)
    assert not breaker.halted
    handler.set_phase("close", D3)
    n = len(queue)
    portfolio.update_from_signal(SignalEvent(D3, "AAA", LONG, strength=0.5), queue)
    assert len(queue) > n


# ------------------------------------------------------------- MOC routing

def test_moc_signal_routing_and_delay_guard():
    from quantester.data.streaming import StreamingDataHandler

    idx = pd.bdate_range("2024-01-02", periods=2)
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
         "volume": 1e6},
        index=idx,
    )
    handler = StreamingDataHandler({"AAA": df})
    handler.set_phase("close", D1)
    portfolio = PortfolioManager(handler, 100_000.0)
    portfolio.update_from_fill(FillEvent(D1, "AAA", 100, BUY, 100.0, 0.0, 0.0))

    queue = _Queue()
    portfolio.update_from_signal(
        SignalEvent(D1, "AAA", EXIT, delay=1, fill_at="close"), queue
    )
    moc = [o for o in queue if o.order_type == MOC_ORDER]
    assert len(moc) == 1
    assert moc[0].earliest_fill_time == D1  # this bar's close auction
    assert moc[0].direction == SELL and moc[0].quantity == pytest.approx(100.0)

    # A delay=0 strategy requesting a close fill would trade a print that does
    # not exist yet at decision time: rejected loudly.
    with pytest.raises(ValueError):
        portfolio.update_from_signal(
            SignalEvent(D1, "AAA", EXIT, delay=0, fill_at="close"), _Queue()
        )


# ------------------------------------------------------------ cash yield

def test_idle_cash_yield_accrual():
    """Kaufman/Carver (notebook-verified): idle cash earns rate x fraction."""
    portfolio = _portfolio()
    portfolio.cash_yield_rate = 0.04
    portfolio.idle_cash_fraction = 0.5      # Kaufman: half the T-bill rate
    portfolio.update_portfolio_valuation(_valuation(D1, 100.0))   # baseline ts
    portfolio.update_portfolio_valuation(_valuation(D2, 100.0))   # +1 day
    expected = 100_000.0 * (1 + 0.04 * 0.5 / 365.0)
    assert portfolio.cash == pytest.approx(expected)
    # Calendar gaps accrue elapsed days, not bars (Jan 3 -> Jan 8 = 5 days).
    portfolio.update_portfolio_valuation(_valuation(pd.Timestamp("2024-01-08"), 100.0))
    assert portfolio.cash == pytest.approx(
        expected * (1 + 0.04 * 0.5 * 5 / 365.0)
    )


def test_idle_cash_yield_off_by_default_and_no_negative_accrual():
    portfolio = _portfolio()  # defaults: yield disabled
    portfolio.update_portfolio_valuation(_valuation(D1, 100.0))
    portfolio.update_portfolio_valuation(_valuation(D2, 100.0))
    assert portfolio.cash == pytest.approx(100_000.0)

    portfolio2 = _portfolio()
    portfolio2.cash_yield_rate = 0.04
    portfolio2.cash = -1_000.0  # borrowed cash accrues nothing
    portfolio2.update_portfolio_valuation(_valuation(D1, 100.0))
    portfolio2.update_portfolio_valuation(_valuation(D2, 100.0))
    assert portfolio2.cash == pytest.approx(-1_000.0)
