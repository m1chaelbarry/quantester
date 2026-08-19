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
from quantester.indicators import atr as wilder_atr

ZERO_COSTS = CostModel(
    fixed_commission=0.0, per_share_commission=0.0, spread_pct=0.0,
    slippage_vol_coef=0.0, impact_coef=0.0,
)
CAPITAL = 100_000.0


def _frame(bars, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=len(bars), tz="UTC")
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
        (99.83, 99.85, 99.76, 99.78),   # 201: dips through T3 (low > stop)
        (99.78, 100.05, 99.77, 100.02), # 202: close back above SMA5 -> exit
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


def test_hard_stop_triggers_on_low_and_exits_next_open():
    """Intra-bar LOW triggers the stop at close; exit fills at the next open
    (delay=1). Same-bar MOC after observing the low is not live-tradable on
    OHLC without an intrabar trigger event."""
    bars = _ramp()
    bars += [
        (100.0, 100.0, 99.80, 99.83),  # 200: fills T1, T2 (low above stop)
        # 201: low breaches the latched stop (T3 fills at the open first);
        # the close stays ABOVE the stop — a close-triggered stop would not
        # fire, proving the low trigger.
        (99.83, 99.85, 99.60, 99.77),
        # 202: exit fills at this open (delay=1 after stop detection).
        (99.70, 99.72, 99.68, 99.71),
    ] + [(99.92, 99.92, 99.92, 99.92)] * 3
    _, portfolio = _run(bars)

    _, _, thresholds, stop = _latched_levels(bars)
    frame = _frame(bars)
    assert bars[201][2] <= stop < bars[201][3]  # low breached, close did not
    buys = [f for f in portfolio.fills if f.direction == BUY]
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert len(buys) == 3  # gap bar took out all three resting limits
    assert [f.reference_price for f in buys] == pytest.approx(thresholds)
    assert len(sells) == 1
    assert sells[0].timestamp == frame.index[202]   # next-bar open fill
    assert sells[0].fill_price == pytest.approx(99.70)  # open of exit bar
    assert sells[0].quantity == pytest.approx(sum(b.quantity for b in buys))
    assert portfolio.positions == {}
    assert len(portfolio.trades) == 1 and portfolio.trades[0]["pnl"] < 0


def test_unfilled_ladder_reanchors_until_first_fill():
    """While nothing is filled, the anchor must refresh to the current
    peak/ATR every bar (cancel + replace) instead of pinning to the first
    latch — a runaway market must not strand the ladder at a stale peak."""
    bars = _ramp()  # bullish from bar 199 onward, zero-range: no dip, no fill
    bars += [(100.0 + 0.1 * i, 100.0 + 0.1 * i, 100.0 + 0.1 * i,
              100.0 + 0.1 * i) for i in range(1, 31)]  # slow grind to 103
    strategy, portfolio = _run(bars)
    assert portfolio.fills == []  # never dipped 1.5x ATR below the peak
    assert strategy._state == "active"
    # Anchor tracked the market to the very last bar (not the first latch).
    last20 = [b[3] for b in bars[-20:]]
    assert strategy._peak == pytest.approx(max(last20))
    assert strategy._latched_at == _frame(bars).index[-1]


def test_regime_loss_purges_unfilled_ladder():
    """A resting (unfilled) ladder is a latent entry: losing the bull regime
    pulls it. Contrast run proves the same ladder fills when bullish."""
    # Upside-range bars: ATR lifts toward ~7 while the low pins at 99.99,
    # never touching the ladder (T1 sinks from ~99.92 toward ~90).
    wide = [(100.0, 107.0, 99.99, 100.0)] * 40
    # Bearish close with a shallow low: above T1 (~90), so nothing fills
    # at the open, then the close below SMA200 purges the ladder.
    purge_bar = [(100.0, 100.5, 95.5, 96.0)]
    deep_dip = [(95.5, 95.6, 80.0, 85.0)]     # would take out every tranche
    tail = [(85.0, 85.0, 85.0, 85.0)] * 3

    bars_purge = _ramp() + wide + purge_bar + deep_dip + tail
    strategy, portfolio = _run(bars_purge)
    assert portfolio.fills == []               # purged before the deep bar
    assert strategy._state == "flat"           # regime lost: back to flat

    bullish_bar = [(100.0, 100.5, 95.5, 97.5)]  # close stays above SMA200
    bars_live = _ramp() + wide + bullish_bar + deep_dip + tail
    _, portfolio_live = _run(bars_live)
    assert len(portfolio_live.fills) > 0       # same ladder, still armed: fills


def test_reanchor_every_skips_intervening_bars():
    """reanchor_every=N refreshes the ladder only every N bars while hunting;
    fills are still marked on intervening bars."""
    bars = _ramp()
    # 5 flat bullish bars after warmup: with reanchor_every=3 the anchor
    # timestamps should be bars 199 (first arm), 202, 205 — not every bar.
    bars += [(100.0 + 0.01 * i, 100.0 + 0.01 * i, 100.0 + 0.01 * i,
              100.0 + 0.01 * i) for i in range(1, 7)]
    handler = StreamingDataHandler({"BTC": _frame(bars)})
    strategy = TranchePullbackStrategy(handler, "BTC", reanchor_every=3)
    portfolio = PortfolioManager(handler, CAPITAL, sizer=PercentEquitySizer(1.0))
    BacktestEngine(handler, strategy, portfolio,
                   SimulatedExecutionHandler(ZERO_COSTS)).run_backtest()
    assert portfolio.fills == []
    assert strategy._state == "active"
    # Last refresh on the final bar (199+6=205): 199, then +3 -> 202, +3 -> 205.
    assert strategy._latched_at == _frame(bars).index[-1]
    assert strategy._bars_since_anchor == 0


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


# --------------------------------------------------------------------------
# Resting STOP_ORDER opt-in (synthesis 5.5)
# --------------------------------------------------------------------------


def _run_resting(bars, **kwargs):
    handler = StreamingDataHandler({"BTC": _frame(bars)})
    strategy = TranchePullbackStrategy(handler, "BTC", resting_stops=True, **kwargs)
    portfolio = PortfolioManager(handler, CAPITAL, sizer=PercentEquitySizer(1.0))
    execution = SimulatedExecutionHandler(ZERO_COSTS)
    BacktestEngine(handler, strategy, portfolio, execution).run_backtest()
    return strategy, portfolio, execution


def test_resting_stop_fills_at_stop_level_and_cleans_residual():
    """resting_stops=True: after the first tranche fills, the catastrophic
    stop rests on the execution ledger (sized to the open position). A bar
    whose low breaches it fills AT min(stop, open) on that bar's close phase —
    one bar earlier than the legacy delay-1 exit. A tranche that fills on the
    same bar (after the stop was placed) is flattened by the mirrored exit at
    the next open."""
    bars = _ramp()
    bars += [
        (100.0, 100.0, 99.80, 99.83),  # 200: fills T1, T2 (low above stop)
        # 201: T3 fills AND the low breaches the latched stop; the stop fires
        # at the stop level this close (legacy: next open).
        (99.83, 99.85, 99.60, 99.77),
        # 202: mirrored exit flattens the T3 residual at this open.
        (99.70, 99.72, 99.68, 99.71),
    ] + [(99.92, 99.92, 99.92, 99.92)] * 3
    strategy, portfolio, _ = _run_resting(bars)

    _, _, (t1, t2, t3), stop = _latched_levels(bars)
    frame = _frame(bars)
    buys = [f for f in portfolio.fills if f.direction == BUY]
    sells = [f for f in portfolio.fills if f.direction == SELL]
    assert [f.reference_price for f in buys] == pytest.approx([t1, t2, t3])
    q1, q2, q3 = (CAPITAL * f / t for f, t in zip((0.25, 0.35, 0.40), (t1, t2, t3)))

    # The resting stop covered the position known at placement (T1 + T2).
    assert sells[0].timestamp == frame.index[201]     # touch bar, not next open
    assert sells[0].reference_price == pytest.approx(stop)  # stop level
    assert sells[0].quantity == pytest.approx(q1 + q2)
    # The mirrored exit cleans up the later-filled T3 residual at the open.
    assert sells[1].timestamp == frame.index[202]
    assert sells[1].quantity == pytest.approx(q3)
    assert portfolio.positions == {}


def test_resting_stop_replacement_preserves_ladder():
    """Re-placing the stop as more tranches fill must purge ONLY stops: the
    unfilled ladder limits keep working (scoped cancel)."""
    bars = _ramp()
    bars += [
        (100.0, 100.0, 99.89, 99.89),  # 200: fills T1 only; stop rests (q1)
        (99.89, 99.90, 99.84, 99.85),  # 201: fills T2; stop replaced (q1+q2)
        (99.85, 99.86, 99.77, 99.80),  # 202: fills T3 — ladder survived
        (99.80, 99.95, 99.79, 99.90),  # 203: close >= SMA5 -> exit
        (99.92, 99.93, 99.91, 99.92),  # 204: exit fills at this open
    ] + [(99.92, 99.92, 99.92, 99.92)] * 3
    strategy, portfolio, _ = _run_resting(bars)

    _, _, (t1, t2, t3), stop = _latched_levels(bars)
    buys = [f for f in portfolio.fills if f.direction == BUY]
    sells = [f for f in portfolio.fills if f.direction == SELL]
    # All three tranches filled AT their levels: the stop replacement never
    # canceled the ladder, and no stop fired (every low stayed above it).
    assert [f.reference_price for f in buys] == pytest.approx([t1, t2, t3])
    assert all(b[2] > stop for b in bars[200:204])
    assert len(sells) == 1
    assert sells[0].quantity == pytest.approx(sum(f.quantity for f in buys))
    assert portfolio.positions == {}
