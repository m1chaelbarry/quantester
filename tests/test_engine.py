"""Temporal-firewall enforcement and no-look-ahead regression."""

import numpy as np
import pandas as pd
import pytest

from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.events import LONG, SignalEvent
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import FixedUnitSizer, PortfolioManager
from quantester.strategy.base import Strategy
from quantester.strategy.examples import MovingAverageCrossStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv
from quantester.validation.truncation import run_truncation_test


def _run(df, strategy, zero_costs, capital=100_000.0, allow_same_print_fills=False):
    handler = HistoricCSVDataHandler({"AAA": df})
    strategy.data_handler = handler
    portfolio = PortfolioManager(handler, capital, sizer=FixedUnitSizer(100))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(zero_costs),
                            allow_same_print_fills=allow_same_print_fills)
    engine.run_backtest()
    return handler, portfolio, engine


def test_delay1_fills_at_next_bar_open(ohlc, zero_costs):
    """Signal at close of bar T must fill at the open of bar T+1 (zero costs)."""
    handler, portfolio, _ = _run(
        ohlc, MovingAverageCrossStrategy(None, "AAA", fast=3, slow=8), zero_costs
    )
    positions = portfolio.positions_history["AAA"]
    first_active = positions[positions != 0].index[0]

    from quantester.strategy.examples import crossover_positions

    targets = crossover_positions(ohlc["close"], 3, 8)
    first_signal = targets[targets != 0].index[0]
    assert first_active == handler.timestamp_at_offset(first_signal, 1)

    fill = portfolio.fills[0]
    assert fill.timestamp == first_active
    assert fill.fill_price == pytest.approx(
        float(ohlc.loc[first_active, "open"]), rel=1e-12
    )


def test_delay0_intra_bar_guard(ohlc, zero_costs):
    """delay=0: strategy fills at bar T's open but cannot see bar T's close."""
    seen = {}

    class CloseToOpenStrategy(Strategy):
        delay = 0
        symbol = "AAA"
        done = False

        def calculate_signals(self, event, events_queue):
            bars = self.data_handler.get_latest_bars(self.symbol, 10**6)
            seen[event.timestamp] = (
                bars.index.max() if len(bars) else None,
                self.data_handler.get_current_open(self.symbol),
            )
            if len(bars) >= 1 and event.bars.get(self.symbol) is not None:
                prev_close = float(bars["close"].iloc[-1])
                if event.bars[self.symbol]["open"] < prev_close * 0.995 and not self.done:
                    events_queue.put(
                        SignalEvent(event.timestamp, self.symbol, LONG, delay=0)
                    )
                    self.done = True

    strategy = CloseToOpenStrategy()
    _, portfolio, _ = _run(ohlc, strategy, zero_costs, allow_same_print_fills=True)

    for ts, (last_visible, current_open) in seen.items():
        if last_visible is not None:
            assert last_visible < ts  # never the current bar's close
        assert current_open == pytest.approx(float(ohlc.loc[ts, "open"]))

    if portfolio.fills:  # a mean-reversion entry, if triggered, fills same-bar open
        fill = portfolio.fills[0]
        assert fill.fill_price == pytest.approx(
            float(ohlc.loc[fill.timestamp, "open"]), rel=1e-12
        )


class _Delay0OpenStrategy(Strategy):
    """Minimal delay=0 strategy: buys at the current bar's open once."""

    delay = 0
    done = False

    def calculate_signals(self, event, events_queue):
        if event.bars.get("AAA") is not None and not self.done:
            events_queue.put(
                SignalEvent(event.timestamp, "AAA", LONG, delay=0)
            )
            self.done = True


def test_delay0_forbidden_without_opt_in(ohlc, zero_costs):
    """D4 (ticket 22): same-print fills are unphysical by default — the engine
    refuses delay=0 strategies unless allow_same_print_fills=True (Harris
    latency). The firewall path stays available behind the opt-in."""
    handler = HistoricCSVDataHandler({"AAA": ohlc})
    portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(100))
    with pytest.raises(ValueError, match="allow_same_print_fills"):
        BacktestEngine(handler, _Delay0OpenStrategy(), portfolio,
                       SimulatedExecutionHandler(zero_costs))
    assert portfolio.fills == []  # refused before any fill


def test_delay0_opt_in_fills_at_same_bar_open(ohlc, zero_costs):
    """With the explicit opt-in, the delay-0 firewall path still works."""
    _, portfolio, _ = _run(
        ohlc, _Delay0OpenStrategy(), zero_costs, allow_same_print_fills=True
    )
    assert portfolio.fills  # delay-0 filled
    fill = portfolio.fills[0]
    assert fill.fill_price == pytest.approx(
        float(ohlc.loc[fill.timestamp, "open"]), rel=1e-12
    )


def test_delay0_signal_from_delay1_strategy_rejected_at_runtime(ohlc, zero_costs):
    """A delay>=1 strategy hand-emitting a delay=0 SignalEvent is refused at
    dispatch time, not silently filled on the observed print."""

    class SneakyStrategy(Strategy):
        delay = 1
        done = False

        def calculate_signals(self, event, events_queue):
            if event.bars.get("AAA") is not None and not self.done:
                events_queue.put(
                    SignalEvent(event.timestamp, "AAA", LONG, delay=0)
                )
                self.done = True

    handler = HistoricCSVDataHandler({"AAA": ohlc})
    portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(100))
    engine = BacktestEngine(handler, SneakyStrategy(), portfolio,
                            SimulatedExecutionHandler(zero_costs))
    with pytest.raises(ValueError, match="allow_same_print_fills"):
        engine.run_backtest()


def test_delay1_strategies_unaffected_without_flag(ohlc, zero_costs):
    """Default delay-1 strategies run fine with the flag off (default)."""
    _, portfolio, _ = _run(
        ohlc, MovingAverageCrossStrategy(None, "AAA", fast=3, slow=8), zero_costs
    )
    assert portfolio.fills  # normal delay-1 fills proceed


def test_no_lookahead_truncation_regression(zero_costs):
    """Chan truncation check: positions on truncated data must be identical."""
    df = make_synthetic_ohlcv("AAA", n_bars=150, seed=31)

    def run_fn(truncate_last):
        data = df if truncate_last is None else df.iloc[:-truncate_last]
        handler = HistoricCSVDataHandler({"AAA": data})
        strategy = MovingAverageCrossStrategy(handler, "AAA", fast=3, slow=8)
        portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(100))
        engine = BacktestEngine(handler, strategy, portfolio,
                                SimulatedExecutionHandler(zero_costs))
        engine.run_backtest()
        return portfolio.positions_history

    result = run_truncation_test(run_fn, n_truncated=25)
    assert result.passed, result.mismatches[:3]


def test_source_ohlcv_sealed_during_calculate_signals(ohlc, zero_costs):
    """Research frames must not be readable from calculate_signals."""

    class LeakyStrategy(Strategy):
        delay = 1

        def calculate_signals(self, event, events_queue):
            self.data_handler.source_ohlcv("AAA")

    strategy = LeakyStrategy()
    with pytest.raises(PermissionError, match="source_ohlcv"):
        _run(ohlc, strategy, zero_costs)


def test_source_ohlcv_available_after_backtest(ohlc, zero_costs):
    handler, _, _ = _run(
        ohlc, MovingAverageCrossStrategy(None, "AAA", fast=3, slow=8), zero_costs
    )
    frame = handler.source_ohlcv("AAA")
    assert len(frame) == len(ohlc)
    assert list(frame.columns)[:4] == ["open", "high", "low", "close"]
