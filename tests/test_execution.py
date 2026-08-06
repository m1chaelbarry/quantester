"""Cost models, stop gap-through, earliest_fill_time enforcement."""

import pandas as pd
import pytest

from quantester.events import BUY, SELL, STOP_ORDER, MARKET_ORDER, MarketEvent, OrderEvent
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler

T0 = pd.Timestamp("2024-01-02")
T1 = pd.Timestamp("2024-01-03")
T2 = pd.Timestamp("2024-01-04")

BAR = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                 "volume": 1e6})


def _market(ts, bar=BAR):
    return MarketEvent(ts, bars={"AAA": bar}, phase="open")


class _Queue(list):
    def put(self, item):
        self.append(item)


def test_commission_and_determinism():
    model = CostModel(fixed_commission=1.0, per_share_commission=0.005)
    assert model.commission(100) == pytest.approx(1.5)
    assert model.commission(0) == 0.0
    a1 = model.adverse_adjustment(100.0, 500, BAR)
    a2 = model.adverse_adjustment(100.0, 500, BAR)
    assert a1 == a2  # deterministic: no simulated noise flow dy (Cross-Ref 1.C)


def test_kyle_lambda_direction():
    model = CostModel()
    small = model.kyle_lambda(100.0, 100, 1e6, 101.0, 99.0)
    large = model.kyle_lambda(100.0, 10_000, 1e6, 101.0, 99.0)
    deep = model.kyle_lambda(100.0, 100, 1e9, 101.0, 99.0)
    assert large > small          # impact grows with own order size dx
    assert deep < small           # impact shrinks with volume/depth
    assert model.kyle_lambda(100.0, 100, 0.0, 101.0, 99.0) == 0.0


def test_market_fill_adverse_side():
    execution = SimulatedExecutionHandler(CostModel())
    queue = _Queue()
    execution.on_market(_market(T0), queue)
    buy = OrderEvent(T0, "AAA", MARKET_ORDER, 100, BUY, earliest_fill_time=T0)
    sell = OrderEvent(T0, "AAA", MARKET_ORDER, 100, SELL, earliest_fill_time=T0)
    execution.execute_order(buy, queue)
    execution.execute_order(sell, queue)
    assert len(queue) == 2
    buy_fill, sell_fill = queue
    assert buy_fill.fill_price > BAR["open"]   # pays adverse adjustment
    assert sell_fill.fill_price < BAR["open"]
    assert buy_fill.commission > 0
    assert buy_fill.slippage_cost > 0


def test_earliest_fill_time_enforcement():
    """Orders park in the ledger until their earliest_fill_time arrives."""
    execution = SimulatedExecutionHandler(CostModel())
    queue = _Queue()
    execution.on_market(_market(T0), queue)
    order = OrderEvent(T0, "AAA", MARKET_ORDER, 100, BUY, earliest_fill_time=T2)
    execution.execute_order(order, queue)
    assert len(queue) == 0                     # not eligible yet

    execution.on_market(_market(T1), queue)
    assert len(queue) == 0                     # still parked at T1
    execution.on_market(_market(T2), queue)
    assert len(queue) == 1                     # filled at T2's open
    assert queue[0].timestamp == T2


def test_untradeable_bar_keeps_order_pending():
    execution = SimulatedExecutionHandler(CostModel())
    queue = _Queue()
    missing = MarketEvent(T0, bars={"AAA": None}, phase="open")
    execution.on_market(missing, queue)
    order = OrderEvent(T0, "AAA", MARKET_ORDER, 100, BUY, earliest_fill_time=T0)
    execution.execute_order(order, queue)
    assert len(queue) == 0                     # no fill without a bar
    execution.on_market(_market(T1), queue)
    assert len(queue) == 1                     # fills at next available bar


def test_stop_gap_through_fills_at_next_available_price():
    """A gapped-through stop fills at the open, never the guaranteed stop."""
    execution = SimulatedExecutionHandler(CostModel(
        fixed_commission=0, per_share_commission=0, spread_pct=0,
        slippage_vol_coef=0, impact_coef=0,
    ))
    queue = _Queue()
    gap_bar = pd.Series({"open": 110.0, "high": 115.0, "low": 108.0,
                         "close": 112.0, "volume": 1e6})
    execution.on_market(_market(T0, gap_bar), queue)
    buy_stop = OrderEvent(T0, "AAA", STOP_ORDER, 100, BUY,
                          earliest_fill_time=T0, stop_price=105.0)
    execution.execute_order(buy_stop, queue)
    assert len(queue) == 1
    assert queue[0].fill_price == 110.0        # open, not the 105 stop

    # Untriggered stop stays pending.
    execution2 = SimulatedExecutionHandler(CostModel())
    queue2 = _Queue()
    execution2.on_market(_market(T0), queue2)
    far_stop = OrderEvent(T0, "AAA", STOP_ORDER, 100, BUY,
                          earliest_fill_time=T0, stop_price=200.0)
    execution2.execute_order(far_stop, queue2)
    assert len(queue2) == 0
