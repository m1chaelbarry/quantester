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
    STOP_ORDER,
    FillEvent,
    MarketEvent,
    SignalEvent,
)
from quantester.portfolio.portfolio import (
    FixedUnitSizer,
    FractionalRiskSizer,
    HedgeRatioSizer,
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


def test_trade_pnl_includes_entry_commission():
    portfolio = _portfolio()
    portfolio.update_from_fill(FillEvent(TS, "AAA", 100, BUY, 10.0, 2.0, 0.0))
    portfolio.update_from_fill(FillEvent(TS, "AAA", 100, SELL, 12.0, 1.5, 0.0))
    trade = portfolio.trades[0]
    assert trade["pnl"] == pytest.approx(200.0 - 2.0 - 1.5)
    assert trade["entry_commission"] == pytest.approx(2.0)
    assert trade["exit_commission"] == pytest.approx(1.5)


def test_margin_reliquidates_while_breached():
    """A single liquidation_fraction cut that leaves leverage above the limit
    must re-fire on the next valuation (not one-shot)."""
    monitor = MarginMonitor(max_leverage=1.5, liquidation_fraction=0.1)
    assert monitor.update(equity=100_000, gross_exposure=300_000) is True
    assert monitor.breach_count == 1
    assert monitor.restricted
    # Still breached after a tiny shrink: must return True again.
    assert monitor.update(equity=100_000, gross_exposure=270_000) is True
    assert monitor.breach_count == 1  # count only on first trip
    # Recover to the limit.
    assert monitor.update(equity=100_000, gross_exposure=150_000) is False
    assert not monitor.restricted


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
        stop_distance = 10.0

    assert FixedUnitSizer(200)(_Signal(), None, 10.0) == 100.0
    portfolio = _portfolio()
    portfolio.last_prices["AAA"] = 10.0
    target = PercentEquitySizer(0.5)(_Signal(), portfolio, 10.0)
    assert target == pytest.approx(100_000 * 0.5 * 0.5 / 10.0)
    risk_target = FractionalRiskSizer(0.02)(_Signal(), portfolio, 10.0)
    assert risk_target == pytest.approx(100_000 * 0.02 / 10.0)


def test_hedge_ratio_sizer_scales_x_leg_off_y_notional():
    """q_X = -β q_Y: X uses Y's price, not its own, so the spread is β-hedged."""
    class _Y:
        signal_type = LONG
        strength = 1.0
        hedge_ratio = 1.0
        hedge_ref_price = 50.0

    class _X:
        signal_type = "SHORT"
        strength = 1.0
        hedge_ratio = 1.4
        hedge_ref_price = 50.0

    portfolio = _portfolio()
    sizer = HedgeRatioSizer(0.5)
    q_y = sizer(_Y(), portfolio, 50.0)
    q_x = sizer(_X(), portfolio, 25.0)  # X's own price must not drive size
    assert q_y == pytest.approx(100_000 * 0.5 / 50.0)  # 1000
    assert q_x == pytest.approx(-1.4 * q_y)


def test_hedge_ratio_zero_does_not_collapse_to_one():
    class _X:
        signal_type = "SHORT"
        hedge_ratio = 0.0
        hedge_ref_price = 50.0

    assert HedgeRatioSizer(0.5)(_X(), _portfolio(), 25.0) == 0.0


# --------------------------------------------------------------------------
# D10 (ticket 26): live sizers size on cash, not mark-to-market equity
# --------------------------------------------------------------------------


class _LongSignal:
    signal_type = "LONG"
    strength = 1.0
    stop_distance = 10.0

    def __init__(self, timestamp=None):
        self.timestamp = timestamp


def _inflated_book():
    """Equity 100k but cash only 20k (80k marked-to-market position)."""
    portfolio = _portfolio()
    portfolio.cash = 20_000.0
    portfolio.positions["AAA"] = 8_000.0
    portfolio.last_prices["AAA"] = 10.0
    return portfolio


def test_sizers_default_base_is_cash_not_mtm_equity():
    """D10: rising MTM equity with flat cash must NOT grow target quantity —
    that is the procyclical unwind Penfold warns about (synthesis 1.16)."""
    book = _inflated_book()
    assert book.equity == pytest.approx(100_000.0)  # sanity: MTM-inflated

    pct = PercentEquitySizer(0.5)  # default base="cash"
    assert pct(_LongSignal(), book, 10.0) == pytest.approx(20_000 * 0.5 / 10.0)
    # Explicit procyclical opt-in reproduces the legacy MTM number.
    pct_eq = PercentEquitySizer(0.5, base="equity")
    assert pct_eq(_LongSignal(), book, 10.0) == pytest.approx(100_000 * 0.5 / 10.0)

    risk = FractionalRiskSizer(0.02)
    assert risk(_LongSignal(), book, 10.0) == pytest.approx(20_000 * 0.02 / 10.0)
    risk_eq = FractionalRiskSizer(0.02, base="equity")
    assert risk_eq(_LongSignal(), book, 10.0) == pytest.approx(100_000 * 0.02 / 10.0)


def test_sizers_zero_or_negative_cash_target_zero():
    """No silent fall-back onto equity when cash is gone."""
    book = _portfolio()
    book.cash = -500.0

    class _X:
        signal_type = "SHORT"
        strength = 1.0
        hedge_ratio = 1.4
        hedge_ref_price = 50.0
        timestamp = None

    assert PercentEquitySizer(0.5)(_LongSignal(), book, 10.0) == 0.0
    assert FractionalRiskSizer(0.02)(_LongSignal(), book, 10.0) == 0.0
    assert HedgeRatioSizer(0.5)(_X(), book, 25.0) == 0.0


def test_cash_ewma_span_smooths_step_change():
    """EWMA cash base reacts slower than raw cash to a step change."""
    idx = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    portfolio = _portfolio()
    raw = PercentEquitySizer(0.5)
    smooth = PercentEquitySizer(0.5, cash_ewma_span=4)
    raw(_LongSignal(idx[0]), portfolio, 10.0)
    smooth(_LongSignal(idx[0]), portfolio, 10.0)
    portfolio.cash = 200_000.0  # step change
    raw(_LongSignal(idx[1]), portfolio, 10.0)
    smooth(_LongSignal(idx[1]), portfolio, 10.0)
    t_raw = raw(_LongSignal(idx[2]), portfolio, 10.0)
    t_smooth = smooth(_LongSignal(idx[2]), portfolio, 10.0)
    assert t_raw == pytest.approx(200_000 * 0.5 / 10.0)  # fully caught up
    assert 100_000 * 0.5 / 10.0 < t_smooth < t_raw      # still smoothing


def test_sizer_base_and_span_validation():
    with pytest.raises(ValueError, match="base"):
        PercentEquitySizer(0.5, base="net-liquidation")
    with pytest.raises(ValueError, match="cash_ewma_span"):
        PercentEquitySizer(0.5, cash_ewma_span=1)


def test_hedge_ratio_sizer_cash_base_keeps_beta_relation():
    """q_X = -beta * q_Y still holds on the cash base (D10 + 1.13)."""
    class _Y:
        signal_type = LONG
        strength = 1.0
        hedge_ratio = 1.0
        hedge_ref_price = 50.0
        timestamp = None

    class _X:
        signal_type = "SHORT"
        strength = 1.0
        hedge_ratio = 1.4
        hedge_ref_price = 50.0
        timestamp = None

    book = _inflated_book()  # cash 20k, equity 100k
    sizer = HedgeRatioSizer(0.5)
    q_y = sizer(_Y(), book, 50.0)
    q_x = sizer(_X(), book, 25.0)
    assert q_y == pytest.approx(20_000 * 0.5 / 50.0)  # cash, not equity
    assert q_x == pytest.approx(-1.4 * q_y)


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


def test_optimal_f_default_is_raw_biggest_loss():
    """D3 (ticket 21): the default W is the raw historical BiggestLoss
    (gap_stress 1.0); the 1.5x stress is an opt-in, not the default."""
    trades = np.array([2.0, -1.0, 3.0, -1.0, 1.5, -0.5])
    default = optimal_f(trades)
    assert default == pytest.approx(optimal_f(trades, gap_stress=1.0))
    assert default == pytest.approx(
        optimal_f(trades, worst_loss=float(trades.min()))
    )
    # The opt-in stress still engages and never raises the effective fraction.
    stressed = optimal_f(trades, gap_stress=1.5)
    assert stressed / abs(trades.min() * 1.5) <= default / abs(trades.min()) + 1e-9


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
    pd.Timestamp("2024-01-02", tz="UTC"),
    pd.Timestamp("2024-01-03", tz="UTC"),
    pd.Timestamp("2024-01-04", tz="UTC"),
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


def test_breaker_rolls_on_session_close_not_utc_midnight():
    """D11 (ticket 27): the breaker's day boundary is the configured session
    roll (default 16:00 America/New_York), never a naive UTC date change."""
    breaker = DailyDrawdownBreaker()
    # 14:00 UTC = 09:00 ET (before the 16:00 roll): session of Jan 2.
    t0 = pd.Timestamp("2024-01-02 14:00", tz="UTC")
    assert not breaker.update(t0, 100_000.0)  # seeds the baseline
    # 23:00 UTC = 18:00 ET Jan 2 (after the roll): session Jan 3 — baseline
    # carries 100k; a 5% drop trips the 4.5% breaker.
    t1 = pd.Timestamp("2024-01-02 23:00", tz="UTC")
    assert breaker.update(t1, 95_000.0)
    assert breaker.halted
    # 01:00 UTC Jan 3 = 20:00 ET Jan 2: SAME NY session as t1. The UTC date
    # change at 00:00 must NOT reset the halt or the baseline.
    t2 = pd.Timestamp("2024-01-03 01:00", tz="UTC")
    assert not breaker.update(t2, 94_000.0)
    assert breaker.halted  # no midnight reset
    # 21:30 UTC = 16:30 ET Jan 3 (after the roll): new session baseline; the
    # halt clears and the baseline carries the last valuation (94k).
    t3 = pd.Timestamp("2024-01-03 21:30", tz="UTC")
    assert not breaker.update(t3, 94_000.0)
    assert not breaker.halted
    # A fresh 4.5% slide off the carried 94k re-trips in the new session.
    assert breaker.update(t3, 89_700.0)
    assert breaker.triggered_count == 2


def test_breaker_custom_session_roll():
    """A crypto-style midnight-UTC session roll keeps date-change behavior."""
    import datetime as _dt

    breaker = DailyDrawdownBreaker(
        day_roll_time=_dt.time(0, 0), tz="UTC",
    )
    d1 = pd.Timestamp("2024-01-02 12:00", tz="UTC")
    d2 = pd.Timestamp("2024-01-03 12:00", tz="UTC")
    assert not breaker.update(d1, 100_000.0)
    assert breaker.update(d1, 95_000.0)  # trips same-session
    assert breaker.halted
    assert not breaker.update(d2, 94_000.0)  # rolled at 00:00 UTC: halt clears
    assert not breaker.halted


def test_breaker_liquidates_cancels_and_blocks_entries():
    from quantester.data.streaming import StreamingDataHandler

    idx = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
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

    idx = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
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


# -------------------------------------------------------- resting stops

def test_signal_stop_price_emits_resting_flatten_stop():
    """Opt-in intra-bar stop: PortfolioManager rests STOP_ORDER with the entry."""
    from quantester.data.streaming import StreamingDataHandler

    idx = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": 1e6},
        index=idx,
    )
    handler = StreamingDataHandler({"AAA": df})
    handler.set_phase("close", idx[0])
    portfolio = PortfolioManager(handler, 100_000.0, sizer=PercentEquitySizer(1.0))
    queue = _Queue()
    portfolio.update_from_signal(
        SignalEvent(
            idx[0], "AAA", LONG, strength=0.1, delay=1, stop_price=95.0,
        ),
        queue,
    )
    markets = [o for o in queue if o.order_type == MARKET_ORDER]
    stops = [o for o in queue if o.order_type == STOP_ORDER]
    assert len(markets) == 1 and len(stops) == 1
    assert stops[0].direction == SELL
    assert stops[0].stop_price == pytest.approx(95.0)
    assert stops[0].quantity == pytest.approx(markets[0].quantity)
    assert stops[0].earliest_fill_time == markets[0].earliest_fill_time


def test_stop_only_signal_rests_stop_without_resizing():
    """Tranche freeze: attach a protective stop to the filled book, no delta."""
    from quantester.data.streaming import StreamingDataHandler

    idx = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": 1e6},
        index=idx,
    )
    handler = StreamingDataHandler({"AAA": df})
    handler.set_phase("close", idx[0])
    portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(40.0))
    portfolio.update_from_fill(FillEvent(idx[0], "AAA", 40.0, BUY, 100.0, 0.0, 0.0))
    queue = _Queue()
    portfolio.update_from_signal(
        SignalEvent(
            idx[0], "AAA", LONG, delay=1, stop_price=90.0, stop_only=True,
        ),
        queue,
    )
    assert [o.order_type for o in queue] == [STOP_ORDER]
    assert queue[0].quantity == pytest.approx(40.0)
    assert queue[0].direction == SELL
    assert queue[0].stop_price == pytest.approx(90.0)
    assert queue[0].earliest_fill_time == idx[1]


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
    portfolio.update_portfolio_valuation(_valuation(pd.Timestamp("2024-01-08", tz="UTC"), 100.0))
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
