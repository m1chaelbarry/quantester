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


def test_run_backtest_delay0_needs_same_print_opt_in(aaa):
    """D4: the facade forwards allow_same_print_fills to the engine."""
    from quantester.strategy.base import Strategy

    class Delay0Strategy(Strategy):
        delay = 0

        def __init__(self, data_handler):
            self.data_handler = data_handler

        def calculate_signals(self, event, events_queue):
            pass  # never trades; the gate fires before signals matter

    with pytest.raises(ValueError, match="allow_same_print_fills"):
        run_backtest(aaa, Delay0Strategy, symbol="AAA")
    result = run_backtest(
        aaa, Delay0Strategy, symbol="AAA", allow_same_print_fills=True
    )
    assert len(result.equity) > 0


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


def test_check_lookahead_passes(aaa):
    result = run_backtest(
        aaa, MovingAverageCrossStrategy, symbol="AAA", fast=5, slow=20,
    )
    truncation = result.check_lookahead(n_truncate=20)
    assert truncation.passed


def test_public_exports_include_data_helpers():
    import quantester

    assert callable(quantester.load_yahoo)
    assert callable(quantester.load_crypto)
    assert callable(quantester.make_synthetic_ohlcv)
    assert callable(quantester.generate_tearsheet)


def test_streaming_calendar_helpers(aaa):
    from quantester.data.csv_handler import HistoricCSVDataHandler

    handler = HistoricCSVDataHandler({"AAA": aaa})
    assert handler.n_bars == len(aaa)
    assert handler.first_timestamp == aaa.index[0]
    assert handler.last_timestamp == aaa.index[-1]
    frame = handler.source_ohlcv("AAA")
    assert list(frame.columns)[:4] == ["open", "high", "low", "close"]
