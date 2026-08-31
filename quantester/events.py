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

CORPORATE_ACTION = "CORPORATE_ACTION"  # ex-date dividend cash / split quantity
FUNDING_SETTLEMENT = "FUNDING_SETTLEMENT"  # perpetual funding as ETF-trick cash

LONG = "LONG"
SHORT = "SHORT"
EXIT = "EXIT"

BUY = "BUY"
SELL = "SELL"

MARKET_ORDER = "MARKET"
STOP_ORDER = "STOP"
LIMIT_ORDER = "LIMIT"
MOC_ORDER = "MOC"  # market-on-close: fill at the bar's close print, same bar only
CANCEL_ORDER = "CANCEL"  # purge resting limit/stop orders for the symbol

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
    fill_at='close' requests a market-on-close fill at the CURRENT bar's close
    (close-phase delay>=1 strategies only) — the live-legitimate MOC pattern for
    decisions made on intra-bar information, e.g. Kaufman's close-execution
    stop rule.
    """

    symbol: str
    signal_type: str  # LONG / SHORT / EXIT
    strength: float = 1.0
    delay: int = 1
    fill_at: str = OPEN
    limit_price: Optional[float] = None
    cancel_orders: bool = False
    stop_distance: Optional[float] = None
    hedge_ratio: Optional[float] = None
    hedge_ref_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_only: bool = False
    cap_long_increase: bool = False

    def __init__(self, timestamp, symbol, signal_type, strength=1.0, delay=1,
                 fill_at=OPEN, limit_price=None, cancel_orders=False,
                 stop_distance=None, hedge_ratio=None, hedge_ref_price=None,
                 stop_price=None, stop_only=False, cap_long_increase=False):
        super().__init__(SIGNAL, timestamp)
        if signal_type not in (LONG, SHORT, EXIT):
            raise ValueError(
                f"signal_type must be LONG, SHORT, or EXIT — got {signal_type!r}. "
                "Direction comes from signal_type; do not encode it in strength."
            )
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError(
                f"delay must be an integer >= 0 (0 = this bar's open, "
                f"1 = next bar's open). Got {delay!r}."
            )
        strength = float(strength)
        if signal_type != EXIT and strength <= 0:
            raise ValueError(
                f"strength must be > 0 for {signal_type} signals "
                f"(it scales position size, not direction). Got {strength!r}."
            )
        if fill_at not in (OPEN, CLOSE, "open", "close"):
            raise ValueError(
                f"fill_at must be 'open' or 'close'; got {fill_at!r}."
            )
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.delay = delay
        self.fill_at = fill_at
        # When set, the portfolio sizes the target AT this price and rests a
        # LIMIT order (e.g. tranche ladders priced off latched levels).
        self.limit_price = limit_price
        # Strategies that rest orders (limit ladders) set this on EXIT to purge
        # the book: unfilled resting levels must not survive the exit.
        self.cancel_orders = cancel_orders
        # Price-unit distance to the protective stop; consumed by
        # FractionalRiskSizer as q = equity * risk_fraction / stop_distance.
        self.stop_distance = stop_distance
        # Spread hedge: HedgeRatioSizer uses ratio=1 on Y and β on X, with
        # hedge_ref_price = P_Y so q_X = -β q_Y.
        self.hedge_ratio = hedge_ratio
        self.hedge_ref_price = hedge_ref_price
        # Optional resting STOP_ORDER price; PortfolioManager emits a flatten
        # stop alongside the entry when this is set (opt-in intra-bar stops).
        self.stop_price = stop_price
        # When True, skip the size delta and only rest/replace the protective
        # stop on the current (or sized) quantity — used after a tranche freeze.
        self.stop_only = bool(stop_only)
        # Crowded-long gate: do not increase a long (new longs from flat blocked).
        self.cap_long_increase = bool(cap_long_increase)


@dataclass
class OrderEvent(Event):
    """Sized order produced by the PortfolioManager.

    `earliest_fill_time` is stamped by the engine/portfolio from the signal's delay
    and is enforced by the event-loop ledger in the execution handler (temporal
    firewall; Cross-Ref section 3.1).
    """

    symbol: str
    order_type: str  # MARKET / STOP / LIMIT / CANCEL
    quantity: float  # always positive
    direction: str  # BUY / SELL
    earliest_fill_time: pd.Timestamp
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None

    def __init__(self, timestamp, symbol, order_type, quantity, direction,
                 earliest_fill_time, stop_price=None, limit_price=None):
        super().__init__(ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = float(quantity)
        self.direction = direction
        self.earliest_fill_time = earliest_fill_time
        self.stop_price = stop_price
        self.limit_price = limit_price


@dataclass
class CorporateActionEvent(Event):
    """Ex-date corporate action routed through the queue to the portfolio
    (ruling D9, ticket 25): dividend cash booking (Peterson ch. 11 — longs
    receive, shorts pay) or split quantity adjustment on the RAW price
    ledger. Never a total-return price rewrite: the unadjusted OHLC stream
    is untouched, so there is no double-booking against adjusted closes.

    Routed via the event queue like every component message — not a direct
    data-handler → portfolio bypass.
    """

    symbol: str
    kind: str  # "dividend" | "split"
    dividend_per_share: Optional[float] = None
    split_ratio: Optional[float] = None

    def __init__(self, timestamp, symbol, kind, dividend_per_share=None,
                 split_ratio=None):
        super().__init__(CORPORATE_ACTION, timestamp)
        if kind not in ("dividend", "split"):
            raise ValueError(f"kind must be 'dividend' or 'split'; got {kind!r}.")
        if kind == "dividend":
            if dividend_per_share is None or float(dividend_per_share) < 0:
                raise ValueError("dividend actions need dividend_per_share >= 0")
            dividend_per_share = float(dividend_per_share)
        else:
            if split_ratio is None or float(split_ratio) <= 0:
                raise ValueError("split actions need split_ratio > 0")
            split_ratio = float(split_ratio)
        self.symbol = symbol
        self.kind = kind
        self.dividend_per_share = dividend_per_share
        self.split_ratio = split_ratio


@dataclass
class FundingSettlementEvent(Event):
    """Perpetual Funding Settlement booked as ETF-trick cash, not a price rewrite.

    Longs pay when ``funding_rate`` is positive: ``cash += -qty * rate * price``.
    Routed at close, before valuation. Distinct from D9 dividends (unsigned).
    """

    symbol: str
    funding_rate: float
    price: float

    def __init__(self, timestamp, symbol, funding_rate: float, price: float):
        super().__init__(FUNDING_SETTLEMENT, timestamp)
        self.symbol = symbol
        self.funding_rate = float(funding_rate)
        self.price = float(price)


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
