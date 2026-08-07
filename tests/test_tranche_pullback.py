"""Tranche pullback ladder: regime gate, latching, tranche fills, exits.

All frames are handcrafted bar-by-bar so every threshold, fill price and
quantity is exactly computable. Zero-cost execution makes fill prices equal
to the resting limit levels (min(open, limit)).
"""

import numpy as np
import pandas as pd
import pytest

from quantester.data.streaming import StreamingDataHandler
from quantester.engine import BacktestEngine
from quantester.events import BUY, SELL
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.visualization.indicators import atr as wilder_atr

ZERO_COSTS = CostModel(
    fixed_commission=0.0, per_share_commission=0.0, spread_pct=0.0,
    slippage_vol_coef=0.0, impact_coef=0.0,
)
CAPITAL = 100_000.0


def _frame(bars, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=len(bars))
    return pd.DataFrame(
        {
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": [1e6] * len(bars),
        },
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


def _ramp(n=200, start=90.0, end=100.0):
    """Zero-range uptrend drift; first possible latch bar is index 199."""
    closes = np.linspace(start, end, n)
    bars, prev = [], closes[0]
    for c in closes:
        bars.append((prev, max(prev, c), min(prev, c), c))
        prev = c
    return bars


def _run(bars):
    handler = StreamingDataHandler({"BTC": _frame(bars)})
    strategy = TranchePullbackStrategy(handler, "BTC")
    portfolio = PortfolioManager(handler, CAPITAL, sizer=PercentEquitySizer(1.0))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(ZERO_COSTS))
    engine.run_backtest()
    return strategy, portfolio


def _latched_levels(bars):
    """Thresholds as computable at the latch bar (index 199) from then-visible
    data — the proof of latching is that fills match THESE, not later values."""
    df = _frame(bars[:200])
    close = df["close"]
    peak = close.rolling(20).max().iloc[-1]
    atr = wilder_atr(df["high"], df["low"], close, 14).iloc[-1]
    t = [peak - k * 1.5 * atr for k in (1, 2, 3)]
    return peak, atr, t, peak - 5.0 * atr


def test_latch_and_tranche_fills_exact_capital_mapping():
    bars = _ramp()
    bars += [
        (100.0, 100.0, 99.80, 99.83),   # 200: dips through T1 and T2
        (99.83, 99.85, 99.70, 99.76),   # 201: dips through T3
        (99.76, 100.05, 99.75, 100.02), # 202: close back above SMA5 -> exit
        (100.04, 100.06, 99.90, 100.05),# 203: exit fills at this open
    ] + [(100.05, 100.05, 100.05, 100.05)] * 5  # flat: re-latch, no fills
    strategy, portfolio = _run(bars)

    _, _, (t1, t2, t3), _ = _latched_levels(bars)
    fills = portfolio.fills
    buys = [f for f in fills if f.direction == BUY]
    sells = [f for f in fills if f.direction == SELL]

    # Three tranche buys at exactly the latched limit levels (latching proof:
    # ATR/peak drifted during the volatile dip bars, so drifting levels would
    # fill at different prices).
    assert [f.reference_price for f in buys] == pytest.approx([t1, t2, t3])
    # q_k = A_t * fraction_k / T_k with A_t = 100k equity at the latch bar.
    assert [f.quantity for f in buys] == pytest.approx(
        [CAPITAL * 0.25 / t1, CAPITAL * 0.35 / t2, CAPITAL * 0.40 / t3]
    )
    # Exact capital mapping: each tranche deploys its equity fraction.
    assert [f.quantity * f.fill_price for f in buys] == pytest.approx(
        [25_000.0, 35_000.0, 40_000.0]
    )
    # One market exit for the full stacked position at bar 203's open.
    assert len(sells) == 1
    assert sells[0].quantity == pytest.approx(sum(f.quantity for f in buys))
    assert sells[0].fill_price == pytest.approx(100.04)
    assert portfolio.positions == {}

    # Single profitable round-trip at the volume-weighted average entry.
    assert len(portfolio.trades) == 1
    trade = portfolio.trades[0]
    assert trade["entry_price"] == pytest.approx(
        np.average([t1, t2, t3], weights=[b.quantity for b in buys])
    )
    assert trade["exit_price"] == pytest.approx(100.04)
    assert trade["pnl"] > 0
    assert portfolio.equity_curve.iloc[-1] == pytest.approx(
        CAPITAL + trade["pnl"]
    )


def test_regime_filter_blocks_entries_below_sma200():
    # Monotonic decline: close < SMA200 at every warmup-complete bar.
    bars = _ramp(n=220, start=100.0, end=80.0)
    bars += [(80.0, 80.0, 70.0, 71.0)]  # deep dip: would fill any live ladder
    _, portfolio = _run(bars)
    assert portfolio.fills == []
    assert portfolio.trades == []


def test_close_based_stop_flattens_position():
    bars = _ramp()
    bars += [
        (100.0, 100.0, 99.80, 99.83),  # 200: fills T1, T2
        (99.83, 99.85, 99.60, 99.70),  # 201: fills T3; close <= latched stop
        (99.68, 99.70, 99.60, 99.65),  # 202: stop exit fills at this open
        # 203+: still bullish, so the machine re-latches at 202's close; these
        # lows stay above the fresh T1 (~99.886), so no new tranche fills.
    ] + [(99.95, 99.95, 99.95, 99.95)] * 3
    _, portfolio = _run(bars)

    _, _, thresholds, stop = _latched_levels(bars)
    assert 99.70 <= stop  # the stop trigger was the close, not the intra-bar low
    buys = [f for f in portfolio.fills if f.direction == BUY]
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert len(buys) == 3  # gap bar took out all three resting limits
    assert [f.reference_price for f in buys] == pytest.approx(thresholds)
    assert len(sells) == 1 and sells[0].fill_price == pytest.approx(99.68)
    assert portfolio.positions == {}
    assert len(portfolio.trades) == 1 and portfolio.trades[0]["pnl"] < 0


def test_exit_purges_unfilled_tranche_limits():
    bars = _ramp()
    bars += [
        (100.0, 100.0, 99.85, 99.87),    # 200: fills T1 only
        (99.87, 100.10, 99.86, 100.08),  # 201: close >= SMA5 -> exit
        # 202: exit fills at the open; the deep low would touch T2/T3 if the
        # book purge on EXIT had not canceled them.
        (100.09, 100.09, 99.70, 100.09),
    ] + [(100.09, 100.09, 100.09, 100.09)] * 3
    strategy, portfolio = _run(bars)

    _, _, (t1, _, _), _ = _latched_levels(bars)
    fills = portfolio.fills
    assert len(fills) == 2  # T1 buy + exit sell only: T2/T3 were purged
    assert fills[0].reference_price == pytest.approx(t1)
    assert fills[1].direction == SELL
    assert fills[1].quantity == pytest.approx(fills[0].quantity)
    assert portfolio.positions == {}
    # The machine re-armed on the post-exit bar (still bullish) and rests a
    # fresh ladder — level values moved with the new peak, proving re-latch.
    assert strategy._state == "active"
    assert strategy._peak == pytest.approx(100.09)
