"""SimulatedExecutionHandler: fills at earliest_fill_time per the temporal firewall.

- Market orders fill at the current bar's open plus adverse cost adjustments.
- Stop orders fill at the next AVAILABLE price after gap-through, never
  guaranteed at the stop price (Cross-Ref-2 section 4.2: perfect stop execution
  understates tail risk and would silently unbound optimal-f): a buy stop
  gapped through at the open fills at the open, not at the stop.
- Limit orders rest in the ledger until touched: a buy limit fills when the
  bar's low reaches the limit, at min(open, limit) — a gap down through the
  limit earns price improvement at the open, never a worse price. Sell limits
  are symmetric (high reached, fills at max(open, limit)).
- Market-on-close orders fill at the close print of their earliest_fill_time
  bar ONLY, and never park in the ledger: missing the close auction expires
  the order (live MOC semantics). The exact-timestamp gate confines the fill
  to the close-phase drain of its own bar, so a MOC fill can never use a
  close print that was not yet known when the decision was made.
- Cancel orders purge every RESTING order (limit/stop) for the symbol from
  the ledger; parked market orders are committed executions in flight and are
  never purged (a cancel must not annul a liquidation already on its way).
- Orders whose earliest_fill_time has not yet arrived are parked in the
  pending-order ledger and re-evaluated on each open-phase MarketEvent
  (state-tracking ledger; Cross-Ref section 3.1).
- Liquidity constraint (RetailCostModel.max_participation_rate): oversized
  market/MOC orders are partially filled up to bar_volume × max_participation
  by default; remaining quantity stays pending for subsequent bars. Limits and
  stops use the same clip when the cost model exposes the knobs. Reject mode
  refuses any fill that would exceed the cap.
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
    """Retail execution diagnostics accumulated over a backtest.

    Participation and impact statistics answer whether orders were genuinely
    retail-sized relative to bar liquidity — "retail-sized" alone is not a
    sufficient assumption (a retail order can still dominate a thin bar).
    """

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


class SimulatedExecutionHandler(ExecutionHandler):
    """Event-driven fill simulator with optional retail liquidity constraints.

    Parameters
    ----------
    cost_model :
        Deterministic cost model shared with the MC fast-track.
    liquidity_policy :
        ``"partial"`` (default research behaviour) clips oversized fills and
        leaves the residual pending across subsequent bars; ``"reject"``
        refuses the entire fill and records a liquidity rejection;
        ``"none"`` disables participation clipping (legacy full-fill behaviour).
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        liquidity_policy: str = "partial",
    ):
        if liquidity_policy not in {"partial", "reject", "none"}:
            raise ValueError(
                "liquidity_policy must be 'partial', 'reject', or 'none'"
            )
        self.cost_model = cost_model or CostModel()
        self.liquidity_policy = liquidity_policy
        self._pending: list = []
        self._bars: dict = {}
        self._timestamp = None
        self.fills: list = []
        self.diagnostics = ExecutionDiagnostics()

    def on_market(self, market_event, events_queue):
        if market_event.phase != OPEN:
            return
        self._bars = market_event.bars
        self._timestamp = market_event.timestamp
        still_pending = []
        for order in self._pending:
            if order.earliest_fill_time <= self._timestamp:
                if not self._try_fill(order, events_queue):
                    still_pending.append(order)  # untradeable / residual
            else:
                still_pending.append(order)
        self._pending = still_pending

    def execute_order(self, order, events_queue):
        if order.order_type == MOC_ORDER:
            # MOC never parks: fill on its own bar's close or expire.
            # A partial MOC residual also expires (close auction is one-shot).
            self._try_fill(order, events_queue, allow_residual=False)
            return
        if order.order_type == CANCEL_ORDER:
            # Synchronous book purge: resting orders (limit/stop) for the
            # symbol are pulled immediately, regardless of phase or
            # earliest_fill_time. Parked MARKET orders are executions already
            # committed (exits/liquidations) and are never purged.
            self._pending = [
                o for o in self._pending
                if not (o.symbol == order.symbol and o.order_type != MARKET_ORDER)
            ]
            return
        if (
            self._timestamp is not None
            and order.earliest_fill_time <= self._timestamp
            and self._try_fill(order, events_queue)
        ):
            return
        self._pending.append(order)

    # ------------------------------------------------------------------ fills

    def _max_fill_qty(self, bar) -> float | None:
        if self.liquidity_policy == "none":
            return None
        model = self.cost_model
        if hasattr(model, "max_fill_quantity"):
            return float(model.max_fill_quantity(float(bar["volume"])))
        return None

    def _try_fill(self, order, events_queue, allow_residual: bool = True) -> bool:
        bar = self._bars.get(order.symbol)
        if bar is None:
            return False  # availability mask: no fill without a bar
        if order.order_type == MARKET_ORDER:
            reference = float(bar["open"])
        elif order.order_type == STOP_ORDER:
            reference = self._stop_reference(order, bar)
            if reference is None:
                return False  # stop not triggered this bar; stays pending
        elif order.order_type == LIMIT_ORDER:
            reference = self._limit_reference(order, bar)
            if reference is None:
                return False  # limit not touched this bar; keeps resting
        elif order.order_type == MOC_ORDER:
            if order.earliest_fill_time != self._timestamp:
                return False  # stale MOC: its close auction has passed
            reference = float(bar["close"])
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        requested_qty = float(order.quantity)
        fill_qty = requested_qty
        max_qty = self._max_fill_qty(bar)
        if max_qty is not None and fill_qty > max_qty + 1e-12:
            if self.liquidity_policy == "reject" or max_qty <= 0:
                self.diagnostics.record_rejection()
                return True  # consumed / rejected — do not keep pending
            # partial: fill what liquidity allows
            fill_qty = max_qty

        if fill_qty <= 0:
            self.diagnostics.record_rejection()
            return True

        adjustment = self.cost_model.adverse_adjustment(reference, fill_qty, bar)
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

        volume = float(bar["volume"])
        participation = fill_qty / volume if volume > 0 else 0.0
        impact_bps = (adjustment / reference * 10_000.0) if reference > 0 else 0.0
        # Prefer the model's participation-impact component when available.
        if hasattr(self.cost_model, "cost_components"):
            comps = self.cost_model.cost_components(reference, fill_qty, bar)
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
            return False  # keep pending with remaining quantity
        return True

    @staticmethod
    def _limit_reference(order, bar) -> float | None:
        """Resting limit fill price: the limit, or the open when gapped through
        (price improvement — a limit fill is never worse than the limit)."""
        limit = order.limit_price
        if limit is None:
            raise ValueError("LIMIT order requires limit_price")
        open_, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if order.direction == BUY:
            if low > limit:
                return None  # not touched
            return min(open_, limit)
        else:
            if high < limit:
                return None
            return max(open_, limit)

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
