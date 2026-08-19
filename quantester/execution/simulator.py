"""SimulatedExecutionHandler: fills at earliest_fill_time per the temporal firewall.

- Market orders fill at the current bar's open plus adverse cost adjustments.
  At open phase, cost models must NOT use the same bar's high/low (those prints
  are not yet known); Kaufman/retail range components are evaluated on an
  open-only proxy (zero intrabar range) unless a prior bar is supplied via
  ``data_handler``.
- Stop/limit touch tests run at **close** phase only (full OHLC is then known).
  At open phase only MARKET residuals are eligible.
- Stop orders fill at the next AVAILABLE price after gap-through, never
  guaranteed at the stop price (Cross-Ref-2 section 4.2).
- Limit orders rest until touched: buy limit fills when the bar's low reaches
  the limit, at min(open, limit); sell limits are symmetric.
- Market-on-close orders fill at the close print of their earliest_fill_time
  bar ONLY. Oversized MOCs are rejected entirely under participation caps
  (close auction is one-shot — silent partial stubs are forbidden).
- Cancel orders purge **every** pending order for the symbol (including
  residual MARKET slices). Fresh liquidation MARKET orders are enqueued
  *after* the cancel in the same drain, so they are not self-cancelled.
- Liquidity constraint (RetailCostModel.max_participation_rate): fill quantity
  is clipped to remaining bar capacity ``volume × max_participation − already
  filled this bar`` (aggregate across orders). Reject mode refuses fills that
  would exceed the cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..events import (
    BUY,
    CANCEL_ORDER,
    LIMIT_ORDER,
    MARKET_ORDER,
    MOC_ORDER,
    OPEN,
    STOP_ORDER,
    FillEvent,
)
from .base import ExecutionHandler
from .costs import CostModel


@dataclass
class ExecutionDiagnostics:
    """Retail execution diagnostics accumulated over a backtest."""

    participations: list = field(default_factory=list)
    impacts_bps: list = field(default_factory=list)
    n_fills: int = 0
    n_partial_fills: int = 0
    n_rejected_liquidity: int = 0
    n_full_fills: int = 0

    def record_fill(
        self,
        *,
        participation: float,
        impact_bps: float,
        requested_qty: float,
        filled_qty: float,
    ) -> None:
        self.n_fills += 1
        self.participations.append(float(participation))
        self.impacts_bps.append(float(impact_bps))
        if filled_qty + 1e-12 < requested_qty:
            self.n_partial_fills += 1
        else:
            self.n_full_fills += 1

    def record_rejection(self) -> None:
        self.n_rejected_liquidity += 1

    def summary(self) -> dict:
        parts = np.asarray(self.participations, dtype=float)
        impacts = np.asarray(self.impacts_bps, dtype=float)
        n = len(parts)

        def _pct(arr, q):
            return float(np.quantile(arr, q)) if len(arr) else 0.0

        return {
            "n_fills": self.n_fills,
            "n_partial_fills": self.n_partial_fills,
            "n_full_fills": self.n_full_fills,
            "n_rejected_liquidity": self.n_rejected_liquidity,
            "pct_partially_filled": (
                self.n_partial_fills / self.n_fills if self.n_fills else 0.0
            ),
            "pct_rejected_liquidity": (
                self.n_rejected_liquidity
                / max(self.n_fills + self.n_rejected_liquidity, 1)
            ),
            "median_participation": _pct(parts, 0.5),
            "p95_participation": _pct(parts, 0.95),
            "max_participation": float(parts.max()) if n else 0.0,
            "pct_trades_gt_1pct_bar_volume": (
                float((parts > 0.01).mean()) if n else 0.0
            ),
            "pct_trades_gt_5pct_bar_volume": (
                float((parts > 0.05).mean()) if n else 0.0
            ),
            "median_impact_bps": _pct(impacts, 0.5),
            "p95_impact_bps": _pct(impacts, 0.95),
            "max_impact_bps": float(impacts.max()) if n else 0.0,
        }


def _bar_has(bar, key: str) -> bool:
    if bar is None:
        return False
    if isinstance(bar, dict):
        return key in bar
    index = getattr(bar, "index", None)
    if index is not None:
        return key in index
    return hasattr(bar, "keys") and key in bar.keys()


class SimulatedExecutionHandler(ExecutionHandler):
    """Event-driven fill simulator with optional retail liquidity constraints.

    Parameters
    ----------
    cost_model :
        Deterministic cost model shared with the MC fast-track.
    liquidity_policy :
        ``"partial"`` clips oversized MARKET fills and leaves residuals pending;
        ``"reject"`` refuses the entire fill; ``"none"`` disables clipping.
        MOC is always all-or-nothing under a participation cap.
    data_handler :
        Optional DataHandler for prior-bar cost proxies at open phase.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        liquidity_policy: str = "partial",
        data_handler=None,
    ):
        if liquidity_policy not in {"partial", "reject", "none"}:
            raise ValueError(
                "liquidity_policy must be 'partial', 'reject', or 'none'"
            )
        self.cost_model = cost_model or CostModel()
        self.liquidity_policy = liquidity_policy
        self.data_handler = data_handler
        self._pending: list = []
        self._bars: dict = {}
        self._timestamp = None
        self._phase = OPEN
        self._filled_this_bar: dict = {}
        self.fills: list = []
        self.rejections: list = []
        self.diagnostics = ExecutionDiagnostics()

    def on_market(self, market_event, events_queue):
        self._bars = market_event.bars
        self._timestamp = market_event.timestamp
        self._phase = market_event.phase
        if market_event.phase == OPEN:
            self._filled_this_bar = {}
        still_pending = []
        for order in self._pending:
            if order.earliest_fill_time > self._timestamp:
                still_pending.append(order)
                continue
            if not self._order_eligible_this_phase(order):
                still_pending.append(order)
                continue
            if not self._try_fill(order, events_queue):
                still_pending.append(order)
        self._pending = still_pending

    def execute_order(self, order, events_queue):
        if order.order_type == CANCEL_ORDER:
            # Purge every pending order for the symbol, including residual
            # MARKET slices from partial fills. Liquidation MARKETs are
            # enqueued after CANCEL in the same drain and are unaffected.
            self._pending = [
                o for o in self._pending if o.symbol != order.symbol
            ]
            return
        if order.order_type == MOC_ORDER:
            if (
                self._timestamp is not None
                and order.earliest_fill_time == self._timestamp
                and self._phase != OPEN
            ):
                self._try_fill_moc(order, events_queue)
            elif (
                self._timestamp is not None
                and order.earliest_fill_time == self._timestamp
                and self._phase == OPEN
            ):
                # MOC submitted at open of its bar: park until close phase.
                self._pending.append(order)
            # else: stale MOC — expire
            return
        if (
            self._timestamp is not None
            and order.earliest_fill_time <= self._timestamp
            and self._order_eligible_this_phase(order)
            and self._try_fill(order, events_queue)
        ):
            return
        self._pending.append(order)

    def _order_eligible_this_phase(self, order) -> bool:
        if order.order_type == MARKET_ORDER:
            return self._phase == OPEN
        if order.order_type in (STOP_ORDER, LIMIT_ORDER, MOC_ORDER):
            return self._phase != OPEN
        return False

    def _resolve_bar(self, symbol):
        bar = self._bars.get(symbol)
        if bar is not None and _bar_has(bar, "high"):
            return bar
        # Open-phase MarketEvent may carry open-only stubs; fall back to full bar.
        if self.data_handler is not None and self._timestamp is not None:
            full = self.data_handler.bar_at(symbol, self._timestamp)
            if full is not None:
                return full
        return bar

    def _max_fill_qty(self, symbol, bar) -> float | None:
        if self.liquidity_policy == "none":
            return None
        model = self.cost_model
        if not hasattr(model, "max_fill_quantity"):
            return None
        cap = float(model.max_fill_quantity(float(bar["volume"])))
        already = float(self._filled_this_bar.get(symbol, 0.0))
        return max(cap - already, 0.0)

    def _cost_proxy_bar(self, symbol, bar):
        """Bar fields visible for cost models at the current phase.

        Open-phase fills must not use the same bar's high/low (look-ahead).
        Prefer the prior bar's range via the data handler; otherwise collapse
        to a zero-range proxy at the open.
        """
        if self._phase != OPEN:
            return bar
        if self.data_handler is not None:
            prev = self.data_handler.get_latest_bars(symbol, 1)
            if not prev.empty:
                return {
                    "open": float(bar["open"]),
                    "high": float(prev["high"].iloc[-1]),
                    "low": float(prev["low"].iloc[-1]),
                    "close": float(prev["close"].iloc[-1]),
                    "volume": float(prev["volume"].iloc[-1]),
                }
        open_ = float(bar["open"])
        volume = float(bar["volume"]) if _bar_has(bar, "volume") else 0.0
        return {
            "open": open_,
            "high": open_,
            "low": open_,
            "close": open_,
            "volume": volume,
        }

    def _try_fill_moc(self, order, events_queue) -> bool:
        bar = self._resolve_bar(order.symbol)
        if bar is None or not _bar_has(bar, "close"):
            return False
        requested = float(order.quantity)
        max_qty = self._max_fill_qty(order.symbol, bar)
        if max_qty is not None and requested > max_qty + 1e-12:
            # All-or-nothing: never leave a silent residual stub after MOC.
            self.diagnostics.record_rejection()
            self.rejections.append(order)
            return True
        return self._try_fill(order, events_queue, allow_residual=False)

    def _try_fill(self, order, events_queue, allow_residual: bool = True) -> bool:
        bar = self._resolve_bar(order.symbol)
        if bar is None:
            return False
        if order.order_type == MARKET_ORDER:
            if not _bar_has(bar, "open"):
                return False
            reference = float(bar["open"])
        elif order.order_type == STOP_ORDER:
            reference = self._stop_reference(order, bar)
            if reference is None:
                return False
        elif order.order_type == LIMIT_ORDER:
            reference = self._limit_reference(order, bar)
            if reference is None:
                return False
        elif order.order_type == MOC_ORDER:
            if order.earliest_fill_time != self._timestamp:
                return False
            if not _bar_has(bar, "close"):
                return False
            reference = float(bar["close"])
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        requested_qty = float(order.quantity)
        fill_qty = requested_qty
        max_qty = self._max_fill_qty(order.symbol, bar)
        if max_qty is not None and fill_qty > max_qty + 1e-12:
            if self.liquidity_policy == "reject" or max_qty <= 0:
                self.diagnostics.record_rejection()
                self.rejections.append(order)
                return True
            fill_qty = max_qty

        if fill_qty <= 0:
            self.diagnostics.record_rejection()
            self.rejections.append(order)
            return True

        cost_bar = self._cost_proxy_bar(order.symbol, bar)
        adjustment = self.cost_model.adverse_adjustment(reference, fill_qty, cost_bar)
        if order.direction == BUY:
            fill_price = reference + adjustment
        else:
            fill_price = max(reference - adjustment, 1e-12)

        fill = FillEvent(
            timestamp=self._timestamp,
            symbol=order.symbol,
            quantity=fill_qty,
            direction=order.direction,
            fill_price=fill_price,
            commission=self.cost_model.commission(fill_qty, price=reference),
            slippage_cost=adjustment * fill_qty,
            reference_price=reference,
        )
        self.fills.append(fill)
        events_queue.put(fill)
        self._filled_this_bar[order.symbol] = (
            self._filled_this_bar.get(order.symbol, 0.0) + fill_qty
        )

        volume = float(bar["volume"]) if _bar_has(bar, "volume") else 0.0
        participation = fill_qty / volume if volume > 0 else 0.0
        impact_bps = (adjustment / reference * 10_000.0) if reference > 0 else 0.0
        if hasattr(self.cost_model, "cost_components"):
            comps = self.cost_model.cost_components(reference, fill_qty, cost_bar)
            if reference > 0:
                impact_bps = comps["participation_impact"] / reference * 10_000.0
            participation = comps["participation"]
        self.diagnostics.record_fill(
            participation=participation,
            impact_bps=impact_bps,
            requested_qty=requested_qty,
            filled_qty=fill_qty,
        )

        residual = requested_qty - fill_qty
        if residual > 1e-12 and allow_residual and self.liquidity_policy == "partial":
            order.quantity = residual
            return False
        return True

    @staticmethod
    def _limit_reference(order, bar) -> float | None:
        limit = order.limit_price
        if limit is None:
            raise ValueError("LIMIT order requires limit_price")
        open_, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if order.direction == BUY:
            if low > limit:
                return None
            return min(open_, limit)
        else:
            if high < limit:
                return None
            return max(open_, limit)

    @staticmethod
    def _stop_reference(order, bar) -> float | None:
        stop = order.stop_price
        open_, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if order.direction == BUY:
            if high < stop:
                return None
            return max(stop, open_)
        else:
            if low > stop:
                return None
            return min(stop, open_)
