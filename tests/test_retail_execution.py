"""Retail cost model, liquidity constraints, and execution diagnostics."""

from __future__ import annotations

from queue import Queue

import numpy as np
import pandas as pd
import pytest

from quantester.events import BUY, MARKET_ORDER, MarketEvent, OrderEvent
from quantester.execution.costs import RetailCostModel, retail_cost_scenario
from quantester.execution.simulator import SimulatedExecutionHandler


def _bar(price=100.0, volume=10_000.0, range_pct=0.02):
    return pd.Series(
        {
            "open": price,
            "high": price * (1 + range_pct / 2),
            "low": price * (1 - range_pct / 2),
            "close": price,
            "volume": volume,
        }
    )


def test_retail_cost_components_scale():
    model = RetailCostModel(
        spread_bps=10.0,
        volatility_slippage_factor=0.5,
        impact_factor=1.0,
        impact_exponent=0.5,
        max_participation_rate=0.05,
    )
    bar = _bar(price=100.0, volume=10_000.0, range_pct=0.02)
    tiny = model.cost_components(100.0, quantity=1.0, bar=bar)
    large = model.cost_components(100.0, quantity=500.0, bar=bar)
    assert tiny["half_spread"] == pytest.approx(100.0 * 0.001 / 2)
    assert large["participation"] > tiny["participation"]
    assert large["participation_impact"] > tiny["participation_impact"]
    # Tiny order / huge volume → negligible impact vs spread.
    assert tiny["participation_impact"] < tiny["half_spread"]


def test_retail_scenarios_and_validation():
    base = retail_cost_scenario("BASE")
    stress = retail_cost_scenario("STRESS")
    assert stress.spread_bps > base.spread_bps
    assert stress.max_participation_rate < base.max_participation_rate
    with pytest.raises(ValueError):
        retail_cost_scenario("LUNATIC")
    with pytest.raises(ValueError):
        RetailCostModel(max_participation_rate=0.0)


def test_partial_fill_and_diagnostics():
    model = RetailCostModel(
        spread_bps=5.0,
        volatility_slippage_factor=0.0,
        impact_factor=0.0,
        max_participation_rate=0.05,
    )
    handler = SimulatedExecutionHandler(model, liquidity_policy="partial")
    ts = pd.Timestamp("2024-01-02", tz="UTC")
    bar = _bar(volume=1_000.0)  # max fill = 50
    q = Queue()
    handler.on_market(MarketEvent(ts, {"AAA": bar}, phase="open"), q)
    order = OrderEvent(
        timestamp=ts,
        symbol="AAA",
        order_type=MARKET_ORDER,
        quantity=200.0,
        direction=BUY,
        earliest_fill_time=ts,
    )
    handler.execute_order(order, q)
    assert not q.empty()
    fill = q.get()
    assert fill.quantity == pytest.approx(50.0)
    assert len(handler._pending) == 1
    assert handler._pending[0].quantity == pytest.approx(150.0)
    summary = handler.diagnostics.summary()
    assert summary["n_partial_fills"] == 1
    assert summary["median_participation"] == pytest.approx(0.05)


def test_reject_liquidity_policy():
    model = RetailCostModel(max_participation_rate=0.01)
    handler = SimulatedExecutionHandler(model, liquidity_policy="reject")
    ts = pd.Timestamp("2024-01-02", tz="UTC")
    bar = _bar(volume=1_000.0)
    q = Queue()
    handler.on_market(MarketEvent(ts, {"AAA": bar}, phase="open"), q)
    order = OrderEvent(
        timestamp=ts,
        symbol="AAA",
        order_type=MARKET_ORDER,
        quantity=100.0,
        direction=BUY,
        earliest_fill_time=ts,
    )
    handler.execute_order(order, q)
    assert q.empty()
    assert handler.diagnostics.n_rejected_liquidity == 1


def test_legacy_cost_model_full_fill_by_default():
    from quantester.execution.costs import CostModel

    handler = SimulatedExecutionHandler(CostModel(
        fixed_commission=0, per_share_commission=0,
        spread_pct=0, slippage_vol_coef=0, impact_coef=0,
    ))
    ts = pd.Timestamp("2024-01-02", tz="UTC")
    bar = _bar(volume=100.0)
    q = Queue()
    handler.on_market(MarketEvent(ts, {"AAA": bar}, phase="open"), q)
    order = OrderEvent(
        timestamp=ts, symbol="AAA", order_type=MARKET_ORDER,
        quantity=500.0, direction=BUY, earliest_fill_time=ts,
    )
    handler.execute_order(order, q)
    assert q.get().quantity == pytest.approx(500.0)
