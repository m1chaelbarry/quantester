"""LETF dual-EMA + Kakushadze Δ: regime, protective stop, leverage sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.events import BUY, EXIT, LONG
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.portfolio.sizing import letf_equity_fraction
from quantester.strategy.letf_dual_ema import (
    LetfDualEmaDeltaStrategy,
    dual_ema_delta_positions,
)
from quantester.validation.truncation import run_truncation_test

ZERO_COSTS = CostModel(
    fixed_commission=0.0, per_share_commission=0.0, spread_pct=0.0,
    slippage_vol_coef=0.0, impact_coef=0.0,
)
CAPITAL = 100_000.0


def _frame(closes, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=len(closes), tz="UTC")
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.05,
            "low": np.minimum(opens, closes) - 0.05,
            "close": closes,
            "volume": np.full(len(closes), 1e6),
        },
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


def test_letf_equity_fraction_halves_for_x2():
    assert letf_equity_fraction(1.0, leverage=2.0) == pytest.approx(0.5)
    assert letf_equity_fraction(0.8, leverage=2.0) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        letf_equity_fraction(1.0, leverage=0.0)


def test_dual_ema_delta_positions_trend_and_stop():
    """Warm-up arm → long → -3% Δ stop → flat while still EMA-bullish."""
    climb = list(np.linspace(100.0, 120.0, 40))
    shock_day = climb[-1] * 0.97
    recovery = [shock_day * 1.005, shock_day * 1.01, shock_day * 1.015]
    close = pd.Series(
        climb + [shock_day] + recovery,
        index=pd.bdate_range("2024-01-01", periods=len(climb) + 1 + len(recovery)),
        dtype=float,
    )
    pos = dual_ema_delta_positions(close, fast=5, slow=15, delta=0.02)

    assert (pos.iloc[:15] == 0.0).all()
    assert pos.iloc[35] == 1.0
    shock_i = len(climb)
    assert pos.iloc[shock_i] == 0.0
    assert (pos.iloc[shock_i:] == 0.0).all()


def test_delta_blocks_arm_on_first_ready_shock_bar():
    """First ready bar that is Δ-hit must not arm long."""
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 95.0] + [95.0] * 10
    close = pd.Series(
        closes, index=pd.bdate_range("2024-01-01", periods=len(closes)), dtype=float
    )
    pos = dual_ema_delta_positions(close, fast=2, slow=5, delta=0.02)
    assert pos.iloc[5] == 0.0


def test_fresh_golden_cross_required_after_delta():
    """After Δ stop, a new golden cross re-opens the long."""
    up = list(np.linspace(100.0, 130.0, 40))
    crash = up[-1] * 0.94
    grind = list(np.linspace(crash, crash * 0.85, 25))
    resume = list(np.linspace(grind[-1], grind[-1] * 1.35, 35))
    close = pd.Series(
        up + [crash] + grind + resume,
        index=pd.bdate_range(
            "2024-01-01", periods=len(up) + 1 + len(grind) + len(resume)
        ),
        dtype=float,
    )
    pos = dual_ema_delta_positions(close, fast=5, slow=15, delta=0.02)
    crash_i = len(up)
    assert pos.iloc[crash_i - 1] == 1.0
    assert pos.iloc[crash_i] == 0.0
    assert pos.iloc[-1] == 1.0
    assert (pos.iloc[crash_i: crash_i + 10] == 0.0).all()


def test_event_form_matches_vectorized_twin():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.001, 0.02, size=180)
    closes = 50.0 * np.cumprod(1.0 + rets)
    df = _frame(closes)
    handler = HistoricCSVDataHandler({"LETF": df})
    strategy = LetfDualEmaDeltaStrategy(handler, "LETF", fast=10, slow=30, delta=0.02)
    handler.prime_data()
    emitted = []

    class _Q:
        def put(self, item):
            emitted.append(item)

    while handler.continue_backtest:
        ts, bars = handler.advance()
        handler.set_phase("close", ts)
        if bars["LETF"] is not None:
            strategy.calculate_signals(
                type("E", (), {"timestamp": ts, "bars": bars})(), _Q()
            )

    twin = strategy.vectorized_signals({"LETF": df})["LETF"]
    for signal in emitted:
        expected = twin.loc[: signal.timestamp].iloc[-1]
        if signal.signal_type == LONG:
            assert expected == 1.0
        else:
            assert signal.signal_type == EXIT
            assert expected == 0.0


def test_delay1_fill_and_half_sizing():
    """Entry at close T fills at open T+1, sized to 50% equity (x2 Kaufman)."""
    closes = list(np.linspace(100.0, 130.0, 60))
    df = _frame(closes)
    handler = HistoricCSVDataHandler({"LETF": df})
    strategy = LetfDualEmaDeltaStrategy(handler, "LETF", fast=5, slow=15, delta=0.02)
    pct = letf_equity_fraction(1.0, leverage=2.0)
    portfolio = PortfolioManager(handler, CAPITAL, sizer=PercentEquitySizer(pct))
    BacktestEngine(
        handler, strategy, portfolio, SimulatedExecutionHandler(ZERO_COSTS),
    ).run_backtest()

    buys = [f for f in portfolio.fills if f.direction == BUY]
    assert len(buys) >= 1
    fill = buys[0]
    assert fill.quantity == pytest.approx(CAPITAL * pct / fill.fill_price, rel=1e-9)


def test_delta_stop_emits_exit_before_ema_death_cross():
    """A -5% close while still EMA-bullish must EXIT (Δ), not wait for death cross."""
    climb = list(np.linspace(100.0, 125.0, 50))
    crash = climb[-1] * 0.95
    after = [crash * 1.01, crash * 1.02, crash * 1.03]
    df = _frame(climb + [crash] + after)
    strategy = LetfDualEmaDeltaStrategy(None, "LETF", fast=5, slow=15, delta=0.02)
    twin = strategy.vectorized_signals({"LETF": df})["LETF"]
    crash_ts = df.index[len(climb)]
    assert twin.loc[df.index[len(climb) - 1]] == 1.0
    assert twin.loc[crash_ts] == 0.0


def test_truncation_no_lookahead():
    rng = np.random.default_rng(3)
    closes = 80.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.018, size=200))
    df = _frame(closes)

    def run(truncate_last):
        data = df if truncate_last is None else df.iloc[:-truncate_last]
        handler = HistoricCSVDataHandler({"LETF": data})
        strategy = LetfDualEmaDeltaStrategy(handler, "LETF")
        portfolio = PortfolioManager(
            handler, CAPITAL,
            sizer=PercentEquitySizer(letf_equity_fraction(1.0, 2.0)),
        )
        BacktestEngine(
            handler, strategy, portfolio, SimulatedExecutionHandler(ZERO_COSTS),
        ).run_backtest()
        return portfolio.positions_history

    result = run_truncation_test(run, n_truncated=25)
    assert result.passed, result
