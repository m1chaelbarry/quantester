"""ExecutionHandler abstract base class (Report 1 section 2.4).

Simulates the physical realities of trading during backtests; in production the
same interface routes to live broker APIs (research-to-live parity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ExecutionHandler(ABC):
    @abstractmethod
    def on_market(self, market_event, events_queue):
        """Receive open-phase MarketEvents; drain the pending-order ledger."""
        ...

    @abstractmethod
    def execute_order(self, order, events_queue):
        """Fill an eligible OrderEvent (pushing a FillEvent) or park it."""
        ...
