"""Core event containers for the event-driven engine.

Implements the strict four-event lifecycle (Report 1 section 3, Report 2 section 4):
MarketEvent -> SignalEvent -> OrderEvent -> FillEvent.

FillEvent carries execution costs split per Report 2 section 2: c_t (proportional
costs: commissions/fees) and phi_t (implementation shortfall: spread/slippage/impact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

MARKET = "MARKET"
SIGNAL = "SIGNAL"
ORDER = "ORDER"
FILL = "FILL"

LONG = "LONG"
SHORT = "SHORT"
EXIT = "EXIT"

BUY = "BUY"
SELL = "SELL"

MARKET_ORDER = "MARKET"
STOP_ORDER = "STOP"

OPEN = "open"
CLOSE = "close"


@dataclass
class Event:
    type: str
    timestamp: pd.Timestamp


@dataclass
class MarketEvent(Event):
    """New bar(s) available at `timestamp`.

    `bars` maps symbol -> OHLCV row (pd.Series) or None when the symbol has no bar
    at this timestamp (untradeable rather than erased; Cross-Ref-2 section 4.3).
    `phase` is 'open' or 'close' and drives the state-based temporal firewall:
    delay=0 strategies act on open-phase events, delay>=1 on close-phase events.
    """

    bars: dict
    phase: str = CLOSE

    def __init__(self, timestamp: pd.Timestamp, bars: dict, phase: str = CLOSE):
        super().__init__(MARKET, timestamp)
        self.bars = bars
        self.phase = phase


@dataclass
class SignalEvent(Event):
    """Strategy output: direction + strength for one symbol.

    `delay` (bars until execution) and `fill_at` (reference price) implement the
    temporal firewall contract: delay=1 fills at the next bar's open; delay=0 with
    fill_at='open' fills at the current bar's open under the intra-bar guard.
    """

    symbol: str
    signal_type: str  # LONG / SHORT / EXIT
    strength: float = 1.0
    delay: int = 1
    fill_at: str = OPEN

    def __init__(self, timestamp, symbol, signal_type, strength=1.0, delay=1, fill_at=OPEN):
        super().__init__(SIGNAL, timestamp)
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.delay = delay
        self.fill_at = fill_at


@dataclass
class OrderEvent(Event):
    """Sized order produced by the PortfolioManager.

    `earliest_fill_time` is stamped by the engine/portfolio from the signal's delay
    and is enforced by the event-loop ledger in the execution handler (temporal
    firewall; Cross-Ref section 3.1).
    """

    symbol: str
    order_type: str  # MARKET / STOP
    quantity: float  # always positive
    direction: str  # BUY / SELL
    earliest_fill_time: pd.Timestamp
    stop_price: Optional[float] = None

    def __init__(self, timestamp, symbol, order_type, quantity, direction,
                 earliest_fill_time, stop_price=None):
        super().__init__(ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = float(quantity)
        self.direction = direction
        self.earliest_fill_time = earliest_fill_time
        self.stop_price = stop_price


@dataclass
class FillEvent(Event):
    """Execution confirmation.

    commission       -> c_t component (proportional costs, Report 2 section 2)
    slippage_cost    -> phi_t component (spread + slippage + market impact, in currency)
    reference_price  -> pre-cost bar reference price used for the fill
    """

    symbol: str
    quantity: float
    direction: str
    fill_price: float
    commission: float
    slippage_cost: float
    reference_price: float = 0.0

    def __init__(self, timestamp, symbol, quantity, direction, fill_price,
                 commission, slippage_cost, reference_price=0.0):
        super().__init__(FILL, timestamp)
        self.symbol = symbol
        self.quantity = float(quantity)
        self.direction = direction
        self.fill_price = float(fill_price)
        self.commission = float(commission)
        self.slippage_cost = float(slippage_cost)
        self.reference_price = float(reference_price)

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost
