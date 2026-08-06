import pandas as pd

from quantester.events import (
    BUY,
    FILL,
    LONG,
    MARKET,
    ORDER,
    SIGNAL,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)

TS = pd.Timestamp("2024-01-02")


def test_market_event_type_and_phase():
    event = MarketEvent(TS, bars={"AAA": None}, phase="close")
    assert event.type == MARKET
    assert event.phase == "close"
    assert event.bars["AAA"] is None


def test_signal_event_defaults():
    signal = SignalEvent(TS, "AAA", LONG)
    assert signal.type == SIGNAL
    assert signal.delay == 1
    assert signal.strength == 1.0


def test_order_event_carries_earliest_fill_time():
    order = OrderEvent(TS, "AAA", "MARKET", 100, BUY, earliest_fill_time=TS)
    assert order.type == ORDER
    assert order.earliest_fill_time == TS


def test_fill_event_cost_split():
    fill = FillEvent(TS, "AAA", 100, BUY, 10.5, commission=1.5, slippage_cost=0.5)
    assert fill.type == FILL
    assert fill.total_cost == 2.0
