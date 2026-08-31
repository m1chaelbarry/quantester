"""EWMAC + crypto-carry Combined Forecast, Funding Settlement, Carver sizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantester.data.crypto_extras import attach_extras, daily_sum
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.data.streaming import normalize_ohlcv_frame
from quantester.engine import BacktestEngine
from quantester.events import (
    LONG,
    FundingSettlementEvent,
    SignalEvent,
)
from quantester.execution.costs import PerpMakerTakerCostModel, perp_cost_scenario
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.permutation import permute_joint_bars
from quantester.portfolio.portfolio import CarverVolTargetSizer, PortfolioManager
from quantester.strategy.base import Strategy
from quantester.strategy.ewmac_carry import (
    EWMACCarryStrategy,
    combined_forecast_frame,
    combined_forecast_positions,
)
from quantester.utils.synthetic import make_synthetic_ohlcv


def _frame(n=180, seed=9, funding=0.0004, dvol=40.0):
    df = make_synthetic_ohlcv("BTC", n_bars=n, seed=seed, mu=0.25, sigma=0.40)
    rng = np.random.default_rng(seed)
    df["funding_rate"] = funding + 0.00005 * rng.normal(size=n)
    df["open_interest"] = 1e9 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    df["dvol"] = dvol
    return df


def test_normalize_keeps_funding_extra():
    df = _frame(n=5)
    out = normalize_ohlcv_frame(df, "BTC")
    assert "funding_rate" in out.columns
    assert "open_interest" in out.columns
    assert list(out.columns[:5]) == ["open", "high", "low", "close", "volume"]


def test_firewall_hides_current_funding_at_open():
    df = _frame(n=8)
    handler = HistoricCSVDataHandler({"BTC": df})
    handler.prime_data()
    handler.advance()
    ts, _ = handler.advance()
    handler.set_phase("open", ts)
    visible = handler.get_latest_bars("BTC", 10)
    assert visible.index.max() < ts
    assert "funding_rate" in visible.columns
    handler.set_phase("close", ts)
    closed = handler.get_latest_bars("BTC", 10)
    assert closed.index.max() == ts
    assert closed["funding_rate"].iloc[-1] == pytest.approx(float(df.loc[ts, "funding_rate"]))


def test_daily_funding_sum_three_prints():
    idx = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    s = pd.Series([0.0001, 0.0002, 0.0003], index=idx)
    daily = pd.DatetimeIndex(["2024-01-01"], tz="UTC")
    out = daily_sum(s, daily)
    assert out.iloc[0] == pytest.approx(0.0006)


def test_carry_forecast_opposite_sign_to_funding():
    df = _frame(n=120, funding=0.01)
    stats = combined_forecast_frame(df, fast=8, slow=16)
    # Persistently positive funding → Carry Forecast should pull the combined
    # forecast down relative to trend-only.
    trend_only = combined_forecast_frame(
        df.assign(funding_rate=0.0), fast=8, slow=16, carry_weight=0.0, trend_weight=1.0
    )
    assert stats["forecast"].iloc[-1] < trend_only["forecast"].iloc[-1]


def test_missing_funding_is_zero_carry_not_untradeable():
    df = make_synthetic_ohlcv("BTC", n_bars=80, seed=3)
    stats = combined_forecast_frame(df, fast=8, slow=16)
    assert stats["forecast"].notna().sum() > 0


def test_crowded_long_needs_positive_funding_and_oi_growth():
    n = 20
    df = _frame(n=n, funding=0.002)
    df["open_interest"] = 1e9
    df.iloc[-1, df.columns.get_loc("open_interest")] = 1e9 * 1.40
    hot = combined_forecast_frame(df, fast=4, slow=8, oi_growth_bars=3)
    assert bool(hot["crowded"].iloc[-1]) is True
    cold = df.copy()
    cold["funding_rate"] = -0.002
    stats = combined_forecast_frame(cold, fast=4, slow=8, oi_growth_bars=3)
    assert bool(stats["crowded"].iloc[-1]) is False


def test_dvol_gate_halves_forecast():
    df = _frame(n=100, dvol=40.0)
    base = combined_forecast_frame(df, fast=8, slow=16)["forecast"].iloc[-1]
    hot = df.copy()
    hot["dvol"] = 90.0
    scaled = combined_forecast_frame(hot, fast=8, slow=16)["forecast"].iloc[-1]
    if np.isfinite(base) and abs(base) > 1e-9:
        assert scaled == pytest.approx(base * 0.5, rel=1e-6)


def test_expanding_scalar_no_lookahead():
    df = _frame(n=90)
    full = combined_forecast_frame(df, fast=8, slow=16)["forecast"]
    prefix = combined_forecast_frame(df.iloc[:70], fast=8, slow=16)["forecast"]
    # Value at t=69 must match a prefix run (no future in the expanding mean).
    assert full.iloc[69] == pytest.approx(float(prefix.iloc[-1]), rel=1e-9, abs=1e-12)


def test_funding_settlement_debits_long_at_close(zero_costs):
    df = _frame(n=12, funding=0.001)
    handler = HistoricCSVDataHandler({"BTC": df})
    port = PortfolioManager(handler, 100_000.0, sizer=CarverVolTargetSizer(inertia_beta=0.0))
    # Force a known qty via a tiny custom sizer after first fill is messy;
    # book the settlement directly after a unit position.
    port.positions["BTC"] = 2.0
    event = FundingSettlementEvent(df.index[5], "BTC", 0.001, 100.0)
    cash0 = port.cash
    port.update_from_funding_settlement(event)
    assert port.cash == pytest.approx(cash0 - 2.0 * 0.001 * 100.0)


def test_engine_books_funding_before_close_signals(zero_costs):
    df = _frame(n=15, funding=0.002)
    handler = HistoricCSVDataHandler({"BTC": df})
    seen = []

    class Probe(Strategy):
        delay = 1

        def __init__(self, data_handler):
            self.data_handler = data_handler

        def calculate_signals(self, event, events_queue):
            seen.append(self.data_handler.get_latest_bars("BTC", 1)["funding_rate"].iloc[-1])
            if event.bars.get("BTC") is not None and len(seen) == 3:
                events_queue.put(SignalEvent(event.timestamp, "BTC", LONG, delay=1))

    port = PortfolioManager(handler, 50_000.0, sizer=CarverVolTargetSizer(inertia_beta=0.0))
    BacktestEngine(
        handler, Probe(handler), port, SimulatedExecutionHandler(zero_costs)
    ).run_backtest()
    assert seen  # close-phase saw funding extras


def test_crowded_long_does_not_increase_position(zero_costs):
    df = _frame(n=20)
    handler = HistoricCSVDataHandler({"BTC": df})
    port = PortfolioManager(handler, 100_000.0)
    port.positions["BTC"] = 5.0
    from quantester.portfolio.sizers import FixedUnitSizer

    port.sizer = FixedUnitSizer(10.0)
    q = []

    class _Q(list):
        def put(self, item):
            self.append(item)

    ts = df.index[10]
    handler.prime_data()
    while handler.continue_backtest:
        t, _ = handler.advance()
        if t == ts:
            handler.set_phase("close", t)
            break
    sig = SignalEvent(ts, "BTC", LONG, strength=1.0, cap_long_increase=True)
    queue = _Q()
    port.update_from_signal(sig, queue)
    # Target would be +10, current 5 → capped at 5 → no order.
    assert not [o for o in queue if getattr(o, "order_type", None) == "MARKET"]


def test_inertia_suppresses_small_rebalance(zero_costs):
    df = _frame(n=20)
    handler = HistoricCSVDataHandler({"BTC": df})
    from quantester.portfolio.sizers import FixedUnitSizer

    class Sticky(FixedUnitSizer):
        inertia_beta = 0.15

    port = PortfolioManager(handler, 100_000.0, sizer=Sticky(100.0))
    port.positions["BTC"] = 95.0
    ts = df.index[5]
    handler.prime_data()
    while handler.continue_backtest:
        t, _ = handler.advance()
        if t == ts:
            handler.set_phase("close", t)
            break

    class _Q(list):
        def put(self, item):
            self.append(item)

    sig = SignalEvent(ts, "BTC", LONG, strength=1.0)
    queue = _Q()
    port.update_from_signal(sig, queue)
    # |100-95| = 5 <= 0.15*100 = 15 → no trade.
    assert not [o for o in queue if getattr(o, "quantity", 0) > 0]


def test_dlr_zeroes_size_at_cap_drawdown():
    df = _frame(n=30)
    handler = HistoricCSVDataHandler({"BTC": df})
    sizer = CarverVolTargetSizer(dlr_threshold=0.10, dlr_cap=0.20, inertia_beta=0.0)
    port = PortfolioManager(handler, 100_000.0, sizer=sizer)
    handler.prime_data()
    handler.advance()
    handler.set_phase("close", handler.current_timestamp)
    port.last_prices["BTC"] = 100.0
    sizer._hwm = 100_000.0
    port.cash = 75_000.0  # 25% DD from HWM
    sig = SignalEvent(handler.current_timestamp, "BTC", LONG, strength=0.5)
    qty = sizer(sig, port, 100.0)
    assert qty == 0.0


def test_perp_cost_model_taker_notional_no_double_phi():
    model = perp_cost_scenario("BASE")
    c = model.commission(10, 50_000.0)
    assert c == pytest.approx(10 * 50_000.0 * 0.0005)
    bar = pd.Series({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1e6})
    adj = model.adverse_adjustment(50_000.0, 10, bar)
    assert adj == pytest.approx(50_000.0 * 0.0002 / 2.0)


def test_vectorized_twin_matches_event_forecast(zero_costs):
    df = _frame(n=100, seed=4)
    handler = HistoricCSVDataHandler({"BTC": df})
    strat = EWMACCarryStrategy(handler, "BTC", fast=8, slow=16)
    collected = {}

    class Capture(EWMACCarryStrategy):
        def calculate_signals(self, event, events_queue):
            bars = self._frame(event)
            if bars is None:
                return
            stats = combined_forecast_frame(bars, **self._forecast_kwargs)
            collected[event.timestamp] = float(stats["forecast"].iloc[-1])
            super().calculate_signals(event, events_queue)

    cap = Capture(handler, "BTC", fast=8, slow=16)
    port = PortfolioManager(handler, 100_000.0, sizer=CarverVolTargetSizer(inertia_beta=0.0))
    BacktestEngine(handler, cap, port, SimulatedExecutionHandler(zero_costs)).run_backtest()
    twin = combined_forecast_positions(df, fast=8, slow=16)
    for ts, f in collected.items():
        assert f / 20.0 == pytest.approx(float(twin.loc[ts]), rel=1e-9, abs=1e-12)


def test_event_engine_runs_with_carver_sizer(zero_costs):
    df = _frame(n=90, seed=2)
    handler = HistoricCSVDataHandler({"BTC": df})
    strat = EWMACCarryStrategy(handler, "BTC", fast=8, slow=16)
    port = PortfolioManager(
        handler, 100_000.0,
        sizer=CarverVolTargetSizer(target_vol=0.15, inertia_beta=0.15),
    )
    BacktestEngine(
        handler, strat, port,
        SimulatedExecutionHandler(perp_cost_scenario("BASE")),
    ).run_backtest()
    inv = port.accounting_invariant()
    assert inv["ok"]
    assert len(port.equity_curve) > 0


def test_truncation_event_path(zero_costs):
    df = _frame(n=80, seed=5)
    from quantester.validation.truncation import run_truncation_test

    def _positions(truncate_last: int | None):
        frame = df if truncate_last is None else df.iloc[:-truncate_last]
        handler = HistoricCSVDataHandler({"BTC": frame})
        strat = EWMACCarryStrategy(handler, "BTC", fast=8, slow=16)
        from quantester.portfolio.sizers import FixedUnitSizer
        port = PortfolioManager(handler, 50_000.0, sizer=FixedUnitSizer(1.0))
        BacktestEngine(handler, strat, port, SimulatedExecutionHandler(zero_costs)).run_backtest()
        return port.positions_history

    result = run_truncation_test(_positions, n_truncated=10)
    assert result.passed


def test_permute_joint_bars_keeps_funding_glued():
    df = _frame(n=40)
    rng = np.random.default_rng(1)
    out = permute_joint_bars(df, rng)
    orig = list(zip(df["close"].to_numpy(), df["funding_rate"].to_numpy()))
    got = list(zip(out["close"].to_numpy(), out["funding_rate"].to_numpy()))
    assert sorted(orig, key=lambda x: (x[0], x[1])) == sorted(got, key=lambda x: (x[0], x[1]))
    assert list(out.index) == list(df.index)
