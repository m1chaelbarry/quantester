"""Cost models, stop gap-through, limit resting fills, cancels, timing."""

import pandas as pd
import pytest

from quantester.events import (
    BUY,
    CANCEL_ORDER,
    LIMIT_ORDER,
    SELL,
    STOP_ORDER,
    MARKET_ORDER,
    MarketEvent,
    OrderEvent,
)
from quantester.execution.costs import ConservativeFrictionCostModel, CostModel
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


_ZERO = dict(fixed_commission=0.0, per_share_commission=0.0, spread_pct=0.0,
             slippage_vol_coef=0.0, impact_coef=0.0)


def test_limit_buy_fills_at_limit_price():
    """Touched buy limit fills at the limit when the open sits above it."""
    execution = SimulatedExecutionHandler(CostModel(**_ZERO))
    queue = _Queue()
    bar = pd.Series({"open": 100.0, "high": 100.5, "low": 99.4,
                     "close": 100.2, "volume": 1e6})
    execution.on_market(_market(T0, bar), queue)
    order = OrderEvent(T0, "AAA", LIMIT_ORDER, 10, BUY,
                       earliest_fill_time=T0, limit_price=99.5)
    execution.execute_order(order, queue)
    assert len(queue) == 1
    assert queue[0].fill_price == pytest.approx(99.5)
    assert queue[0].reference_price == pytest.approx(99.5)


def test_limit_buy_gap_down_fills_at_open_with_improvement():
    """Gapped through the limit: fills at the better open, never worse."""
    execution = SimulatedExecutionHandler(CostModel(**_ZERO))
    queue = _Queue()
    bar = pd.Series({"open": 98.0, "high": 98.5, "low": 97.5,
                     "close": 98.2, "volume": 1e6})
    execution.on_market(_market(T0, bar), queue)
    order = OrderEvent(T0, "AAA", LIMIT_ORDER, 10, BUY,
                       earliest_fill_time=T0, limit_price=99.5)
    execution.execute_order(order, queue)
    assert len(queue) == 1
    assert queue[0].fill_price == pytest.approx(98.0)  # open, better than limit


def test_limit_untouched_rests_then_fills_later():
    execution = SimulatedExecutionHandler(CostModel(**_ZERO))
    queue = _Queue()
    high_bar = pd.Series({"open": 100.0, "high": 100.5, "low": 99.6,
                          "close": 100.2, "volume": 1e6})
    execution.on_market(_market(T0, high_bar), queue)
    order = OrderEvent(T0, "AAA", LIMIT_ORDER, 10, BUY,
                       earliest_fill_time=T0, limit_price=99.5)
    execution.execute_order(order, queue)
    assert len(queue) == 0                     # 99.6 low never touched 99.5
    execution.on_market(_market(T1), queue)    # default BAR low = 99.0
    assert len(queue) == 1
    assert queue[0].fill_price == pytest.approx(99.5)


def test_sell_limit_fills_at_max_open_limit():
    execution = SimulatedExecutionHandler(CostModel(**_ZERO))
    queue = _Queue()
    gap_up = pd.Series({"open": 102.0, "high": 102.5, "low": 101.5,
                        "close": 102.2, "volume": 1e6})
    execution.on_market(_market(T0, gap_up), queue)
    order = OrderEvent(T0, "AAA", LIMIT_ORDER, 10, SELL,
                       earliest_fill_time=T0, limit_price=101.0)
    execution.execute_order(order, queue)
    assert len(queue) == 1
    assert queue[0].fill_price == pytest.approx(102.0)  # gap-up open improves

    execution2 = SimulatedExecutionHandler(CostModel(**_ZERO))
    queue2 = _Queue()
    execution2.on_market(_market(T0), queue2)  # BAR: open 100, high 101
    order2 = OrderEvent(T0, "AAA", LIMIT_ORDER, 10, SELL,
                        earliest_fill_time=T0, limit_price=100.5)
    execution2.execute_order(order2, queue2)
    assert len(queue2) == 1
    assert queue2[0].fill_price == pytest.approx(100.5)  # touched at the limit


def test_cancel_order_purges_resting_book():
    """CANCEL pulls every resting order for the symbol; fills already done
    are untouched, and later bars cannot resurrect the purged orders."""
    execution = SimulatedExecutionHandler(CostModel(**_ZERO))
    queue = _Queue()
    execution.on_market(_market(T0), queue)  # BAR low 99.0, high 101.0
    limit = OrderEvent(T0, "AAA", LIMIT_ORDER, 10, BUY,
                       earliest_fill_time=T2, limit_price=99.5)   # parked (timing)
    stop = OrderEvent(T0, "AAA", STOP_ORDER, 10, BUY,
                      earliest_fill_time=T0, stop_price=200.0)    # parked (untouched)
    market = OrderEvent(T0, "AAA", MARKET_ORDER, 7, SELL,
                        earliest_fill_time=T2)                    # committed exit in flight
    execution.execute_order(limit, queue)
    execution.execute_order(stop, queue)
    execution.execute_order(market, queue)
    assert len(queue) == 0
    execution.execute_order(
        OrderEvent(T0, "AAA", CANCEL_ORDER, 0, BUY, earliest_fill_time=T0), queue
    )
    execution.on_market(_market(T1), queue)  # low 99.0 would touch the limit
    execution.on_market(_market(T2), queue)
    # Only the parked MARKET order survives the purge and fills at T2's open;
    # the resting limit and stop are gone.
    assert len(queue) == 1
    assert queue[0].quantity == 7 and queue[0].direction == SELL
    assert queue[0].timestamp == T2


def test_conservative_friction_model_math():
    """C_trade = 2 * (S/2 + mu_fee): full-spread adjustment + doubled fee."""
    model = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
    # Adverse adjustment = 2 * (price * spread_pct / 2) = full spread.
    assert model.adverse_adjustment(50_000.0, 0.1, BAR) == pytest.approx(10.0)
    # Commission = 2 * fee_rate * qty * price (notional-doubled taker fee).
    assert model.commission(0.1, price=50_000.0) == pytest.approx(4.0)
    assert model.commission(0.0, price=50_000.0) == 0.0
    with pytest.raises(ValueError):
        model.commission(0.1)  # notional fees require the reference price
    a1 = model.adverse_adjustment(50_000.0, 0.1, BAR)
    a2 = model.adverse_adjustment(50_000.0, 0.1, BAR)
    assert a1 == a2  # deterministic: parity with the MC fast-track


def test_friction_model_charged_on_limit_fills():
    """Resting maker fills still pay the conservative taker-grade friction."""
    model = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
    execution = SimulatedExecutionHandler(model)
    queue = _Queue()
    bar = pd.Series({"open": 100.0, "high": 100.5, "low": 99.4,
                     "close": 100.2, "volume": 1e6})
    execution.on_market(_market(T0, bar), queue)
    order = OrderEvent(T0, "AAA", LIMIT_ORDER, 10, BUY,
                       earliest_fill_time=T0, limit_price=99.5)
    execution.execute_order(order, queue)
    fill = queue[0]
    assert fill.reference_price == pytest.approx(99.5)
    # fill_price = limit + 2 * half_spread(limit); commission = 2*fee*notional.
    assert fill.fill_price == pytest.approx(99.5 + 99.5 * 0.0002)
    assert fill.commission == pytest.approx(2 * 0.0004 * 10 * 99.5)
