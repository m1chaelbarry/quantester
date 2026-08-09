"""Tests for the trader-facing run_backtest facade and input guards."""

from __future__ import annotations

import pandas as pd
import pytest

from quantester import (
    BacktestEngine,
    CostModel,
    MovingAverageCrossStrategy,
    PercentEquitySizer,
    run_backtest,
)
from quantester.events import LONG, SignalEvent
from quantester.utils.synthetic import make_synthetic_ohlcv


@pytest.fixture
def aaa():
    return make_synthetic_ohlcv("AAA", n_bars=120, seed=1)


def test_run_backtest_one_dataframe(aaa):
    result = run_backtest(
        aaa,
        MovingAverageCrossStrategy,
        symbol="AAA",
        fast=5,
        slow=20,
        capital=50_000.0,
        equity_pct=0.5,
    )
    assert result.sharpe == result.stats["sharpe"]
    assert len(result.equity) > 0
    text = result.summary()
    assert "Total return" in text
    assert "Sharpe" in text


def test_run_backtest_rejects_built_instance(aaa):
    with pytest.raises(TypeError, match="already-built"):
        run_backtest(
            aaa,
            MovingAverageCrossStrategy(None, "AAA", 5, 20),
            symbol="AAA",
        )


def test_run_backtest_factory(aaa):
    result = run_backtest(
        {"AAA": aaa},
        lambda h: MovingAverageCrossStrategy(h, "AAA", fast=5, slow=20),
    )
    assert isinstance(result.stats["total_return"], float)


def test_run_backtest_missing_symbol(aaa):
    with pytest.raises(ValueError, match="symbol="):
        run_backtest(aaa, MovingAverageCrossStrategy, fast=5, slow=20)


def test_run_backtest_bad_equity_pct(aaa):
    with pytest.raises(ValueError, match="equity_pct"):
        run_backtest(
            aaa, MovingAverageCrossStrategy, symbol="AAA",
            fast=5, slow=20, equity_pct=90,
        )


def test_percent_equity_sizer_rejects_bad_pct():
    with pytest.raises(ValueError, match="\\(0, 1\\]"):
        PercentEquitySizer(0.0)


def test_cost_model_rejects_percent_as_fraction():
    with pytest.raises(ValueError, match="fraction"):
        CostModel(spread_pct=0.05)


def test_signal_event_rejects_bad_type():
    ts = pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="LONG, SHORT, or EXIT"):
        SignalEvent(ts, "AAA", "BUY")


def test_signal_event_rejects_non_positive_strength():
    ts = pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="strength"):
        SignalEvent(ts, "AAA", LONG, strength=0.0)


def test_engine_rejects_wrong_order_components(aaa):
    from quantester.data.csv_handler import HistoricCSVDataHandler
    from quantester.execution.simulator import SimulatedExecutionHandler
    from quantester.portfolio.portfolio import PortfolioManager

    handler = HistoricCSVDataHandler({"AAA": aaa})
    strategy = MovingAverageCrossStrategy(handler, "AAA", 5, 20)
    portfolio = PortfolioManager(handler, 10_000.0)
    execution = SimulatedExecutionHandler(CostModel())
    with pytest.raises(TypeError, match="wrong order"):
        # portfolio where strategy should be
        BacktestEngine(handler, portfolio, strategy, execution)


def test_ma_cross_friendly_window_error():
    with pytest.raises(ValueError, match="fast window"):
        MovingAverageCrossStrategy(None, "AAA", fast=40, slow=10)
