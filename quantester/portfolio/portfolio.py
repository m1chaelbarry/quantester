"""PortfolioManager: cash/holdings ledger, sizing, risk overlays, order generation.

Signal -> sized OrderEvent conversion stamps `earliest_fill_time` from the
signal's delay via the DataHandler calendar (temporal firewall enforcement).
Fills update the ledger; close-phase valuation marks equity to market and runs
the margin monitor, emitting liquidation orders on leverage breaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import BUY, EXIT, LONG, MARKET_ORDER, SELL, SHORT, OrderEvent
from .base import Portfolio
from .risk import MarginMonitor


class FixedUnitSizer:
    """Target = +/- units * strength per signal (used for fast-track parity)."""

    def __init__(self, units: float = 100.0):
        self.units = float(units)

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        return sign * self.units * signal.strength


class PercentEquitySizer:
    """Target quantity worth pct * equity * strength at the reference price."""

    def __init__(self, pct: float = 0.5):
        self.pct = float(pct)

    def __call__(self, signal, portfolio, ref_price: float) -> float:
        if signal.signal_type == EXIT or ref_price <= 0:
            return 0.0
        sign = 1.0 if signal.signal_type == LONG else -1.0
        dollar_target = portfolio.equity * self.pct * signal.strength
        return sign * dollar_target / ref_price


class PortfolioManager(Portfolio):
    def __init__(self, data_handler, initial_capital: float = 100_000.0,
                 sizer=None, margin_monitor: MarginMonitor | None = None):
        self.data_handler = data_handler
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.sizer = sizer or PercentEquitySizer(0.5)
        self.margin_monitor = margin_monitor

        self.positions: dict = {}
        self.last_prices: dict = {}
        self.fills: list = []
        self.trades: list = []          # completed round-trips
        self._open_lots: dict = {}      # symbol -> {qty, avg_price}
        self._equity_history: list = []
        self._positions_history: list = []

    @property
    def equity(self) -> float:
        value = self.cash
        for symbol, qty in self.positions.items():
            price = self.last_prices.get(symbol)
            if price is not None:
                value += qty * price
        return value

    @property
    def gross_exposure(self) -> float:
        return sum(
            abs(qty) * self.last_prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )

    @property
    def equity_curve(self) -> pd.Series:
        if not self._equity_history:
            return pd.Series(dtype=float)
        idx, vals = zip(*[(t, e) for t, e, _, _ in self._equity_history])
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")

    @property
    def positions_history(self) -> pd.DataFrame:
        if not self._positions_history:
            return pd.DataFrame()
        return pd.DataFrame(self._positions_history).set_index("timestamp").fillna(0.0)

    # ------------------------------------------------------------------ signals

    def _reference_price(self, signal) -> float | None:
        if signal.delay == 0:
            price = self.data_handler.get_current_open(signal.symbol)
            return None if price is None else float(price)
        bars = self.data_handler.get_latest_bars(signal.symbol, 1)
        if bars.empty:
            return None
        return float(bars["close"].iloc[-1])

    def update_from_signal(self, signal, events_queue):
        ref_price = self._reference_price(signal)
        if ref_price is None:
            return  # untradeable at this timestamp (availability mask)
        target = float(self.sizer(signal, self, ref_price))
        current = self.positions.get(signal.symbol, 0.0)
        delta = target - current
        if abs(delta) < 1e-12:
            return
        fill_time = self.data_handler.timestamp_at_offset(signal.timestamp, signal.delay)
        if fill_time is None:
            return  # no future bar exists to fill on (end of data)
        events_queue.put(
            OrderEvent(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                order_type=MARKET_ORDER,
                quantity=abs(delta),
                direction=BUY if delta > 0 else SELL,
                earliest_fill_time=fill_time,
            )
        )

    # -------------------------------------------------------------------- fills

    def update_from_fill(self, fill):
        signed_qty = fill.quantity if fill.direction == BUY else -fill.quantity
        self.cash -= signed_qty * fill.fill_price
        # fill_price already embeds the adverse adjustment: slippage_cost (phi_t)
        # is recorded for cost analytics, never double-charged against cash.
        self.cash -= fill.commission
        self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) + signed_qty
        if abs(self.positions[fill.symbol]) < 1e-12:
            del self.positions[fill.symbol]
        self.fills.append(fill)
        self._book_round_trip(fill, signed_qty)

    def _book_round_trip(self, fill, signed_qty: float):
        lot = self._open_lots.get(fill.symbol)
        if lot is None:
            if abs(signed_qty) > 0:
                self._open_lots[fill.symbol] = {
                    "qty": signed_qty, "avg_price": fill.fill_price, "t0": fill.timestamp,
                }
            return
        if np.sign(signed_qty) == np.sign(lot["qty"]) or signed_qty == 0:
            new_qty = lot["qty"] + signed_qty
            lot["avg_price"] = (
                (abs(lot["qty"]) * lot["avg_price"] + abs(signed_qty) * fill.fill_price)
                / abs(new_qty)
            )
            lot["qty"] = new_qty
            return
        # Closing (fully or partially): realize pnl on the closed portion.
        # Slippage is embedded in entry/exit fill prices; only commissions are
        # charged separately to avoid double-counting.
        closed = min(abs(signed_qty), abs(lot["qty"]))
        pnl = closed * (fill.fill_price - lot["avg_price"]) * np.sign(lot["qty"])
        pnl -= fill.commission
        self.trades.append(
            {
                "symbol": fill.symbol,
                "t0": lot["t0"],
                "t1": fill.timestamp,
                "qty": closed,
                "entry_price": lot["avg_price"],
                "exit_price": fill.fill_price,
                "pnl": pnl,
            }
        )
        remaining = lot["qty"] + signed_qty
        if abs(remaining) < 1e-12:
            del self._open_lots[fill.symbol]
        elif np.sign(remaining) != np.sign(lot["qty"]):
            self._open_lots[fill.symbol] = {
                "qty": remaining, "avg_price": fill.fill_price, "t0": fill.timestamp,
            }
        else:
            lot["qty"] = remaining

    # --------------------------------------------------------------- valuation

    def update_portfolio_valuation(self, market_event, events_queue=None):
        for symbol, bar in market_event.bars.items():
            if bar is not None:
                self.last_prices[symbol] = float(bar["close"])
        equity = self.equity
        self._equity_history.append(
            (market_event.timestamp, equity, self.cash, self.gross_exposure)
        )
        snapshot = {"timestamp": market_event.timestamp}
        snapshot.update(self.positions)
        self._positions_history.append(snapshot)

        if (
            events_queue is not None
            and self.margin_monitor is not None
            and self.margin_monitor.is_breach(equity, self.gross_exposure)
        ):
            self._liquidate(market_event, events_queue)

    def _liquidate(self, market_event, events_queue):
        targets = self.margin_monitor.liquidation_targets(self.positions)
        fill_time = self.data_handler.timestamp_at_offset(market_event.timestamp, 1)
        if fill_time is None:
            return
        for symbol, target in targets.items():
            delta = target - self.positions.get(symbol, 0.0)
            if abs(delta) < 1e-12:
                continue
            events_queue.put(
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=symbol,
                    order_type=MARKET_ORDER,
                    quantity=abs(delta),
                    direction=BUY if delta > 0 else SELL,
                    earliest_fill_time=fill_time,
                )
            )
