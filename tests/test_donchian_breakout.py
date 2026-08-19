"""Donchian breakout: regime gate, ADX filter, entries, exits, fractional sizing.

Handcrafted frames keep every threshold and quantity exactly computable.
Zero-cost execution makes fill prices equal to the bar open (delay=1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantester.data.streaming import StreamingDataHandler
from quantester.engine import BacktestEngine
from quantester.events import BUY, EXIT, LONG, SELL, SHORT, SignalEvent
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import FractionalRiskSizer, PortfolioManager
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.indicators import adx as wilder_adx
from quantester.indicators import atr as wilder_atr
from quantester.indicators import donchian

ZERO_COSTS = CostModel(
    fixed_commission=0.0, per_share_commission=0.0, spread_pct=0.0,
    slippage_vol_coef=0.0, impact_coef=0.0,
)
CAPITAL = 100_000.0
RISK = 0.02
STOP_MULT = 2.0


def _frame(bars, start="2024-01-01", freq="h"):
    idx = pd.date_range(start=start, periods=len(bars), freq=freq, tz="UTC")
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


def _trend(n=220, start=100.0, end=160.0, half_range=0.8):
    """Zero-noise uptrend; ADX saturates, close tracks above SMA200."""
    closes = np.linspace(start, end, n)
    bars, prev = [], closes[0]
    for c in closes:
        bars.append((prev, c + half_range, c - half_range, c))
        prev = c
    return bars


def _run(bars, **strategy_kwargs):
    defaults = dict(
        regime_window=200, entry_window=20, trail_window=10, exit_window=20,
        atr_window=14, adx_window=14, adx_threshold=25.0,
        stop_atr_mult=STOP_MULT, risk_fraction=RISK,
    )
    defaults.update(strategy_kwargs)
    handler = StreamingDataHandler({"BTC": _frame(bars)})
    strategy = DonchianBreakoutStrategy(handler, "BTC", **defaults)
    portfolio = PortfolioManager(
        handler, CAPITAL, sizer=FractionalRiskSizer(RISK),
    )
    engine = BacktestEngine(
        handler, strategy, portfolio, SimulatedExecutionHandler(ZERO_COSTS),
    )
    engine.run_backtest()
    return strategy, portfolio, handler


def test_adx_and_donchian_helpers():
    bars = _trend(80)
    df = _frame(bars)
    channel = donchian(df["high"], df["low"], window=20, shift=1)
    # Prior-20 max high excludes the current bar.
    assert channel["upper"].iloc[40] == pytest.approx(
        df["high"].iloc[20:40].max()
    )
    assert channel["lower"].iloc[40] == pytest.approx(
        df["low"].iloc[20:40].min()
    )
    a = wilder_adx(df["high"], df["low"], df["close"], 14)
    assert a["adx"].dropna().iloc[-1] > 25.0
    assert (a["plus_di"].dropna() >= 0).all()
    assert (a["minus_di"].dropna() >= 0).all()


def test_fractional_risk_sizer():
    class _Handler:
        symbols = ["BTC"]

    portfolio = PortfolioManager(_Handler(), CAPITAL)
    portfolio.last_prices["BTC"] = 100.0
    long = SignalEvent(
        pd.Timestamp("2024-01-01"), "BTC", LONG, stop_distance=10.0,
    )
    short = SignalEvent(
        pd.Timestamp("2024-01-01"), "BTC", SHORT, stop_distance=10.0,
    )
    exit_sig = SignalEvent(pd.Timestamp("2024-01-01"), "BTC", EXIT)
    sizer = FractionalRiskSizer(0.02)
    assert sizer(long, portfolio, 100.0) == pytest.approx(CAPITAL * 0.02 / 10.0)
    assert sizer(short, portfolio, 100.0) == pytest.approx(-CAPITAL * 0.02 / 10.0)
    assert sizer(exit_sig, portfolio, 100.0) == 0.0
    with pytest.raises(ValueError):
        sizer(
            SignalEvent(pd.Timestamp("2024-01-01"), "BTC", LONG),
            portfolio, 100.0,
        )


def test_long_entry_sizing_and_delay1_fill():
    """Breakout at bar 219 close → fill at bar 220 open, sized to 2% / 2ATR."""
    bars = _trend(220)
    # Force a clean breakout on the last warmup bar: inflate close above the
    # prior-20 Donchian high while keeping the regime bullish.
    df_pre = _frame(bars)
    prior_high = df_pre["high"].iloc[-21:-1].max()
    breakout = prior_high + 1.0
    o, h, l, c = bars[-1]
    bars[-1] = (o, max(h, breakout + 0.5), l, breakout)

    # Fill bar: open = breakout + 0.25 (entry), then drift up (no immediate exit).
    entry_open = breakout + 0.25
    bars += [
        (entry_open, entry_open + 2.0, entry_open - 0.1, entry_open + 1.5),
        (entry_open + 1.5, entry_open + 3.0, entry_open + 1.0, entry_open + 2.5),
        (entry_open + 2.5, entry_open + 4.0, entry_open + 2.0, entry_open + 3.5),
    ]

    strategy, portfolio, _ = _run(bars)
    buys = [f for f in portfolio.fills if f.direction == BUY]
    assert len(buys) >= 1
    fill = buys[0]
    assert fill.reference_price == pytest.approx(entry_open)
    # stop_distance from the SIGNAL bar's ATR (bar index 219, post-rewrite).
    sig_df = _frame(bars[:220])
    atr_at_signal = float(
        wilder_atr(sig_df["high"], sig_df["low"], sig_df["close"], 14).iloc[-1]
    )
    stop_distance = STOP_MULT * atr_at_signal
    assert fill.quantity == pytest.approx(CAPITAL * RISK / stop_distance)
    assert strategy._protective_stop == pytest.approx(
        entry_open - STOP_MULT * atr_at_signal
    )


def test_adx_filter_blocks_chop():
    """With a sky-high ADX threshold, a breakout must not enter."""
    bars = _trend(230)
    df_pre = _frame(bars[:220])
    prior_high = df_pre["high"].iloc[-21:-1].max()
    breakout = prior_high + 1.0
    o, h, l, _ = bars[219]
    bars[219] = (o, max(h, breakout + 0.5), l, breakout)
    bars = bars[:220] + [
        (breakout, breakout + 1, breakout - 0.1, breakout + 0.5),
    ] * 5
    _, portfolio, _ = _run(bars, adx_threshold=101.0)
    assert portfolio.fills == []


def test_mean_reversion_exit_delay1():
    """After entry, close below SMA20 exits at the next open.

    Protective stop is widened so only the SMA20 (mean-reversion) exit fires.
    """
    bars = _trend(220)
    df_pre = _frame(bars)
    prior_high = df_pre["high"].iloc[-21:-1].max()
    breakout = float(prior_high + 1.0)
    o, h, l, _ = bars[-1]
    bars[-1] = (o, max(h, breakout + 0.5), l, breakout)

    entry_open = breakout + 0.25
    # Bar 220: fill at open, close still elevated (stay long).
    bars.append((entry_open, entry_open + 2.0, entry_open - 0.05, entry_open + 1.0))
    # Bar 221: close clearly below SMA20; low stays above the wide ATR floor.
    pullback = entry_open - 12.0
    bars.append((entry_open + 1.0, entry_open + 1.1, pullback - 0.1, pullback))
    exit_open = pullback - 0.25
    bars.append((exit_open, exit_open + 0.5, exit_open - 0.5, exit_open))
    bars += [(exit_open, exit_open + 0.2, exit_open - 0.2, exit_open)] * 3

    _, portfolio, _ = _run(bars, stop_atr_mult=50.0)
    buys = [f for f in portfolio.fills if f.direction == BUY]
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert len(buys) == 1
    assert len(sells) >= 1
    assert sells[0].reference_price == pytest.approx(exit_open)


def test_protective_atr_stop_next_open():
    """Low through entry − 2×ATR at close → exit fills at the next bar's open."""
    bars = _trend(220)
    df_pre = _frame(bars)
    prior_high = df_pre["high"].iloc[-21:-1].max()
    breakout = float(prior_high + 1.0)
    atr_sig = float(
        wilder_atr(df_pre["high"], df_pre["low"], df_pre["close"], 14).iloc[-1]
    )
    o, h, l, _ = bars[-1]
    bars[-1] = (o, max(h, breakout + 0.5), l, breakout)

    entry_open = breakout + 0.25
    protective = entry_open - STOP_MULT * atr_sig
    # Fill bar: stay above protective.
    bars.append((entry_open, entry_open + 1.0, entry_open - 0.05, entry_open + 0.5))
    # Stop bar: low gaps through protective (detected at close).
    stop_close = protective - 0.5
    bars.append(
        (entry_open + 0.5, entry_open + 0.6, protective - 1.0, stop_close)
    )
    # Exit fills at this open (delay=1); not same-bar MOC.
    exit_open = stop_close + 0.1
    bars.append((exit_open, exit_open + 0.2, exit_open - 0.2, exit_open))
    bars += [(exit_open, exit_open + 0.2, exit_open - 0.2, exit_open)] * 2

    _, portfolio, _ = _run(bars)
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert len(sells) >= 1
    assert sells[0].reference_price == pytest.approx(exit_open)


def test_short_entry_symmetric():
    """Bearish regime + downside Donchian breakout enters short."""
    # Strong downtrend so SMA200 is above price and ADX is elevated.
    closes = np.linspace(200.0, 100.0, 220)
    bars, prev = [], closes[0]
    for c in closes:
        bars.append((prev, c + 0.8, c - 0.8, c))
        prev = c
    df_pre = _frame(bars)
    prior_low = df_pre["low"].iloc[-21:-1].min()
    breakdown = float(prior_low - 1.0)
    o, h, l, _ = bars[-1]
    bars[-1] = (o, h, min(l, breakdown - 0.5), breakdown)

    entry_open = breakdown - 0.25
    bars += [
        (entry_open, entry_open + 0.1, entry_open - 2.0, entry_open - 1.0),
        (entry_open - 1.0, entry_open - 0.5, entry_open - 3.0, entry_open - 2.0),
    ]

    _, portfolio, _ = _run(bars)
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert len(sells) >= 1
    assert sells[0].reference_price == pytest.approx(entry_open)


# --------------------------------------------------------------------------
# Resting STOP_ORDER opt-in (synthesis 5.5): intra-bar stop protection on the
# execution ledger instead of close-observed delay-1 exits.
# --------------------------------------------------------------------------


def test_resting_stop_fills_at_stop_level_same_bar():
    """resting_stops=True: the protective stop rests on the ledger from the
    latch bar; a later bar whose low breaches it fills AT min(stop, open) on
    that bar's close phase — one bar earlier than the delay-1 exit and at the
    stop level rather than the next open (gap-through honored, never the
    guaranteed stop)."""
    bars = _trend(220)
    df_pre = _frame(bars)
    prior_high = df_pre["high"].iloc[-21:-1].max()
    breakout = float(prior_high + 1.0)
    o, h, l, _ = bars[-1]
    bars[-1] = (o, max(h, breakout + 0.5), l, breakout)
    # The strategy latches ATR from the SIGNAL bar (post-rewrite frame).
    atr_sig = float(
        wilder_atr(
            _frame(bars[:220])["high"], _frame(bars[:220])["low"],
            _frame(bars[:220])["close"], 14,
        ).iloc[-1]
    )

    entry_open = breakout + 0.25
    protective = entry_open - STOP_MULT * atr_sig
    bars.append((entry_open, entry_open + 1.0, entry_open - 0.05, entry_open + 0.5))
    # Stop bar: low gaps through protective; open still above it.
    bars.append(
        (entry_open + 0.5, entry_open + 0.6, protective - 1.0, protective - 0.5)
    )
    # Gap-down continuation: the legacy delay-1 exit would fill HERE, lower.
    exit_open = protective - 0.4
    bars.append((exit_open, exit_open + 0.2, exit_open - 0.2, exit_open))
    bars += [(exit_open, exit_open + 0.2, exit_open - 0.2, exit_open)] * 2

    strategy, portfolio, _ = _run(bars, resting_stops=True)
    frame = _frame(bars)
    buys = [f for f in portfolio.fills if f.direction == BUY]
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert len(buys) == 1
    assert len(sells) == 1
    # Stop fired on the touch bar itself, at the stop level (open above it).
    assert sells[0].timestamp == frame.index[221]
    assert sells[0].reference_price == pytest.approx(protective)
    assert sells[0].quantity == pytest.approx(buys[0].quantity)
    assert portfolio.positions == {}
    assert strategy._state == "flat" or strategy._state == "exiting"


def test_resting_stop_survives_until_exit_and_is_purged():
    """Mean-reversion exit with a live resting stop: the EXIT must purge the
    stop, or it would fire later and sell an already-flat book short."""
    bars = _trend(220)
    df_pre = _frame(bars)
    prior_high = df_pre["high"].iloc[-21:-1].max()
    breakout = float(prior_high + 1.0)
    o, h, l, _ = bars[-1]
    bars[-1] = (o, max(h, breakout + 0.5), l, breakout)
    atr_sig = float(
        wilder_atr(
            _frame(bars[:220])["high"], _frame(bars[:220])["low"],
            _frame(bars[:220])["close"], 14,
        ).iloc[-1]
    )

    entry_open = breakout + 0.25
    # 6x ATR protective floor (~10 below entry): the pullback bar's low must
    # stay ABOVE it (only the SMA20 exit fires), while the post-exit bar's
    # deep low dives THROUGH it (an unpurged stop would fire and short).
    stop_mult = 6.0
    protective = entry_open - stop_mult * atr_sig
    bars.append((entry_open, entry_open + 2.0, entry_open - 0.05, entry_open + 1.0))
    pullback = entry_open - 5.0
    assert pullback - 0.1 > protective  # stop untouched on the exit signal bar
    bars.append((entry_open + 1.0, entry_open + 1.1, pullback - 0.1, pullback))
    exit_open = pullback - 0.25
    bars.append((exit_open, exit_open + 0.5, entry_open - 30.0, exit_open))
    assert entry_open - 30.0 < protective  # would fire a surviving stop
    bars += [(exit_open, exit_open + 0.2, exit_open - 0.2, exit_open)] * 2

    handler = StreamingDataHandler({"BTC": _frame(bars)})
    strategy = DonchianBreakoutStrategy(
        handler, "BTC", stop_atr_mult=stop_mult, resting_stops=True,
    )
    portfolio = PortfolioManager(
        handler, CAPITAL, sizer=FractionalRiskSizer(RISK),
    )
    execution = SimulatedExecutionHandler(ZERO_COSTS)
    BacktestEngine(handler, strategy, portfolio, execution).run_backtest()

    buys = [f for f in portfolio.fills if f.direction == BUY]
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert len(buys) == 1
    assert len(sells) == 1  # the market exit only; the stop was purged
    assert sells[0].reference_price == pytest.approx(exit_open)
    assert portfolio.positions == {}
    assert execution._pending == []  # nothing rests after the exit
