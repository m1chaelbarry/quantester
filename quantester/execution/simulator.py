"""SimulatedExecutionHandler: fills at earliest_fill_time per the temporal firewall.

- Market orders fill at the current bar's open plus adverse cost adjustments.
- Stop orders fill at the next AVAILABLE price after gap-through, never
  guaranteed at the stop price (Cross-Ref-2 section 4.2: perfect stop execution
  understates tail risk and would silently unbound optimal-f): a buy stop
  gapped through at the open fills at the open, not at the stop.
- Orders whose earliest_fill_time has not yet arrived are parked in the
  pending-order ledger and re-evaluated on each open-phase MarketEvent
  (state-tracking ledger; Cross-Ref section 3.1).
"""

from __future__ import annotations

from ..events import BUY, MARKET_ORDER, OPEN, STOP_ORDER, FillEvent
from .base import ExecutionHandler
from .costs import CostModel


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, cost_model: CostModel | None = None):
        self.cost_model = cost_model or CostModel()
        self._pending: list = []
        self._bars: dict = {}
        self._timestamp = None
        self.fills: list = []

    def on_market(self, market_event, events_queue):
        if market_event.phase != OPEN:
            return
        self._bars = market_event.bars
        self._timestamp = market_event.timestamp
        still_pending = []
        for order in self._pending:
            if order.earliest_fill_time <= self._timestamp:
                if not self._try_fill(order, events_queue):
                    still_pending.append(order)  # untradeable bar: keep parked
            else:
                still_pending.append(order)
        self._pending = still_pending

    def execute_order(self, order, events_queue):
        if (
            self._timestamp is not None
            and order.earliest_fill_time <= self._timestamp
            and self._try_fill(order, events_queue)
        ):
            return
        self._pending.append(order)

    # ------------------------------------------------------------------ fills

    def _try_fill(self, order, events_queue) -> bool:
        bar = self._bars.get(order.symbol)
        if bar is None:
            return False  # availability mask: no fill without a bar
        if order.order_type == MARKET_ORDER:
            reference = float(bar["open"])
        elif order.order_type == STOP_ORDER:
            reference = self._stop_reference(order, bar)
            if reference is None:
                return False  # stop not triggered this bar; stays pending
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        adjustment = self.cost_model.adverse_adjustment(
            reference, order.quantity, bar
        )
        if order.direction == BUY:
            fill_price = reference + adjustment
        else:
            fill_price = max(reference - adjustment, 1e-12)

        fill = FillEvent(
            timestamp=self._timestamp,
            symbol=order.symbol,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=self.cost_model.commission(order.quantity),
            slippage_cost=adjustment * order.quantity,
            reference_price=reference,
        )
        self.fills.append(fill)
        events_queue.put(fill)
        return True

    @staticmethod
    def _stop_reference(order, bar) -> float | None:
        """Next available price after the stop is touched, including gap-through."""
        stop = order.stop_price
        open_, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if order.direction == BUY:
            if high < stop:
                return None
            return max(stop, open_)  # gapped open fills at the open, not the stop
        else:
            if low > stop:
                return None
            return min(stop, open_)
