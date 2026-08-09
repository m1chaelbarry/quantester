"""PortfolioManager: cash/holdings ledger, sizing, risk overlays, order generation.

Signal -> sized OrderEvent conversion stamps `earliest_fill_time` from the
signal's delay via the DataHandler calendar (temporal firewall enforcement).
Fills update the ledger; close-phase valuation marks equity to market and runs
the margin monitor, emitting liquidation orders on leverage breaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import (
    BUY,
    CANCEL_ORDER,
    LIMIT_ORDER,
    MARKET_ORDER,
    MOC_ORDER,
    OPEN,
    SELL,
    OrderEvent,
)
from .base import Portfolio
from .risk import DailyDrawdownBreaker, MarginMonitor
from .sizers import FixedUnitSizer, FractionalRiskSizer, PercentEquitySizer

__all__ = [
    "FixedUnitSizer",
    "PercentEquitySizer",
    "FractionalRiskSizer",
]


class PortfolioManager(Portfolio):
    def __init__(self, data_handler, initial_capital: float = 100_000.0,
                 sizer=None, margin_monitor: MarginMonitor | None = None,
                 drawdown_breaker: DailyDrawdownBreaker | None = None,
                 cash_yield_rate: float = 0.0, idle_cash_fraction: float = 0.5):
        if cash_yield_rate < 0:
            raise ValueError("cash_yield_rate must be non-negative")
        if not 0.0 <= idle_cash_fraction <= 1.0:
            raise ValueError("idle_cash_fraction must lie in [0, 1]")
        self.data_handler = data_handler
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.sizer = sizer or PercentEquitySizer(0.5)
        self.margin_monitor = margin_monitor
        self.drawdown_breaker = drawdown_breaker
        # Idle-cash yield (notebook-verified): Kaufman accrues HALF the 3-month
        # T-bill rate on unallocated cash while flat (TSM ch. 8); Carver
        # requires including the risk-free rate on undeployed cash in
        # non-derivative backtests (Systematic Trading ch. 12). Effective
        # annualized rate on positive cash = cash_yield_rate * idle_cash_fraction.
        self.cash_yield_rate = float(cash_yield_rate)
        self.idle_cash_fraction = float(idle_cash_fraction)
        self._last_valuation_ts = None

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
        # Limit-priced signals size the target AT the limit price: the strategy
        # declares q = equity * strength / limit_price against latched levels.
        if getattr(signal, "limit_price", None) is not None:
            return float(signal.limit_price)
        if signal.delay == 0:
            price = self.data_handler.get_current_open(signal.symbol)
            return None if price is None else float(price)
        bars = self.data_handler.get_latest_bars(signal.symbol, 1)
        if bars.empty:
            return None
        return float(bars["close"].iloc[-1])

    def update_from_signal(self, signal, events_queue):
        if self.drawdown_breaker is not None and self.drawdown_breaker.halted:
            # Circuit breaker: ALL signal flow is suspended until the daily
            # rollover. Positions are already being flattened by the breaker's
            # own liquidation orders (parked in the ledger and retried at every
            # open until filled), so strategy exits are redundant; dropping
            # them also prevents a duplicate flatten from overselling short.
            return
        if getattr(signal, "fill_at", OPEN) == "close" and signal.delay < 1:
            raise ValueError(
                "fill_at='close' (market-on-close) is only valid for "
                "close-phase strategies (delay >= 1); a delay=0 MOC fill "
                "would trade a close print before it exists."
            )
        if getattr(signal, "cancel_orders", False):
            # Book purge requested (e.g. tranche ladders on EXIT): resting
            # orders must not survive the exit and re-enter on their own. The
            # purge is synchronous on the execution side.
            events_queue.put(
                OrderEvent(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    order_type=CANCEL_ORDER,
                    quantity=0.0,
                    direction=BUY,  # placeholder; unused by the ledger purge
                    earliest_fill_time=signal.timestamp,
                )
            )
        ref_price = self._reference_price(signal)
        if ref_price is None:
            return  # untradeable at this timestamp (availability mask)
        target = float(self.sizer(signal, self, ref_price))
        current = self.positions.get(signal.symbol, 0.0)
        # Margin restriction: block any order that would increase |position|
        # (new entry risk). Risk-reducing shrinks / flips toward flat remain
        # allowed so recovery and intentional exits can proceed.
        if (
            self.margin_monitor is not None
            and self.margin_monitor.restricted
            and abs(target) > abs(current) + 1e-12
        ):
            return
        delta = target - current
        if abs(delta) < 1e-12:
            return
        fill_at_close = getattr(signal, "fill_at", OPEN) == "close"
        if fill_at_close:
            fill_time = signal.timestamp  # this bar's close auction
        else:
            fill_time = self.data_handler.timestamp_at_offset(
                signal.timestamp, signal.delay
            )
            if fill_time is None:
                return  # no future bar exists to fill on (end of data)
        if fill_at_close:
            order_type = MOC_ORDER
        elif signal.limit_price is not None:
            order_type = LIMIT_ORDER
        else:
            order_type = MARKET_ORDER
        events_queue.put(
            OrderEvent(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                order_type=order_type,
                quantity=abs(delta),
                direction=BUY if delta > 0 else SELL,
                earliest_fill_time=fill_time,
                limit_price=signal.limit_price,
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
                    "qty": signed_qty,
                    "avg_price": fill.fill_price,
                    "t0": fill.timestamp,
                    "commission": float(fill.commission),
                }
            return
        if np.sign(signed_qty) == np.sign(lot["qty"]) or signed_qty == 0:
            new_qty = lot["qty"] + signed_qty
            lot["avg_price"] = (
                (abs(lot["qty"]) * lot["avg_price"] + abs(signed_qty) * fill.fill_price)
                / abs(new_qty)
            )
            lot["qty"] = new_qty
            lot["commission"] = float(lot.get("commission", 0.0)) + float(fill.commission)
            return
        # Closing (fully or partially): realize pnl on the closed portion.
        # Slippage is embedded in entry/exit fill prices; commissions on both
        # entry (pro-rated) and exit are deducted so round-trip costs are not
        # understated.
        closed = min(abs(signed_qty), abs(lot["qty"]))
        entry_commission = float(lot.get("commission", 0.0))
        if abs(lot["qty"]) > 1e-12 and closed + 1e-12 < abs(lot["qty"]):
            remain_frac = (abs(lot["qty"]) - closed) / abs(lot["qty"])
            allocated_entry = entry_commission * (1.0 - remain_frac)
            lot["commission"] = entry_commission * remain_frac
            entry_commission = allocated_entry
        else:
            lot["commission"] = 0.0
        pnl = closed * (fill.fill_price - lot["avg_price"]) * np.sign(lot["qty"])
        pnl -= fill.commission + entry_commission
        self.trades.append(
            {
                "symbol": fill.symbol,
                "t0": lot["t0"],
                "t1": fill.timestamp,
                "qty": closed,
                "direction": int(np.sign(lot["qty"])),  # +1 long, -1 short
                "entry_price": lot["avg_price"],
                "exit_price": fill.fill_price,
                "pnl": pnl,
                "commission": fill.commission + entry_commission,
                "entry_commission": entry_commission,
                "exit_commission": fill.commission,
            }
        )
        remaining = lot["qty"] + signed_qty
        if abs(remaining) < 1e-12:
            del self._open_lots[fill.symbol]
        elif np.sign(remaining) != np.sign(lot["qty"]):
            # Flip: this fill's commission was fully attributed to the closed
            # leg above; the new lot starts with zero carried entry commission.
            self._open_lots[fill.symbol] = {
                "qty": remaining,
                "avg_price": fill.fill_price,
                "t0": fill.timestamp,
                "commission": 0.0,
            }
        else:
            lot["qty"] = remaining

    # --------------------------------------------------------------- valuation

    def _accrue_cash_yield(self, timestamp):
        """Accrue the idle-cash yield on positive cash between valuations
        (Kaufman half-T-bill / Carver risk-free inclusion; notebook-verified).
        No borrow charge on negative cash — documented simplification."""
        if (
            self._last_valuation_ts is not None
            and self.cash > 0
            and self.cash_yield_rate > 0
        ):
            days = (timestamp - self._last_valuation_ts).total_seconds() / 86400.0
            if days > 0:
                effective = self.cash_yield_rate * self.idle_cash_fraction
                self.cash += self.cash * effective * days / 365.0
        self._last_valuation_ts = timestamp

    def update_portfolio_valuation(self, market_event, events_queue=None):
        self._accrue_cash_yield(market_event.timestamp)
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
            and self.margin_monitor.update(equity, self.gross_exposure)
        ):
            # Still breached (including re-trips): cancel resting risk, shrink.
            self._margin_liquidate(market_event, events_queue)

        if (
            events_queue is not None
            and self.drawdown_breaker is not None
            and self.drawdown_breaker.update(market_event.timestamp, equity)
        ):
            self._breaker_liquidate(market_event, events_queue)

    def _breaker_liquidate(self, market_event, events_queue):
        """Circuit breaker: cancel every resting order across the book, then
        market-liquidate all open positions at the next bar's open."""
        for symbol in self.data_handler.symbols:
            events_queue.put(
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=symbol,
                    order_type=CANCEL_ORDER,
                    quantity=0.0,
                    direction=BUY,  # placeholder; unused by the ledger purge
                    earliest_fill_time=market_event.timestamp,
                )
            )
        fill_time = self.data_handler.timestamp_at_offset(market_event.timestamp, 1)
        if fill_time is None:
            return  # no future bar exists to liquidate on (end of data)
        for symbol, qty in list(self.positions.items()):
            if abs(qty) < 1e-12:
                continue
            events_queue.put(
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=symbol,
                    order_type=MARKET_ORDER,
                    quantity=abs(qty),
                    direction=SELL if qty > 0 else BUY,
                    earliest_fill_time=fill_time,
                )
            )

    def _margin_liquidate(self, market_event, events_queue):
        """Margin breach: cancel resting entry risk, then shrink positions.

        Restriction remains active (``margin_monitor.restricted``) until
        leverage recovers — strategies cannot increase exposure merely because
        liquidation orders have been queued.
        """
        for symbol in self.data_handler.symbols:
            events_queue.put(
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=symbol,
                    order_type=CANCEL_ORDER,
                    quantity=0.0,
                    direction=BUY,
                    earliest_fill_time=market_event.timestamp,
                )
            )
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

    def accounting_invariant(self, atol: float = 1e-6) -> dict:
        """Core ledger check: equity == cash + marked-to-market positions.

        Subject to documented financing / dividend / borrow mechanics (idle-cash
        yield is already in cash; borrow is a documented simplification).
        """
        mtm = sum(
            qty * self.last_prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )
        expected = self.cash + mtm
        actual = self.equity
        return {
            "cash": self.cash,
            "mtm": mtm,
            "equity": actual,
            "expected_equity": expected,
            "ok": abs(actual - expected) <= atol,
            "abs_diff": abs(actual - expected),
        }
