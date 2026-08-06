"""Portfolio abstract base class: the central gearbox of the engine.

Translates raw strategy signals into precise target orders while enforcing risk
overlays, capital limits, and margin monitoring (Report 1 section 2.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Portfolio(ABC):
    @abstractmethod
    def update_from_signal(self, signal, events_queue):
        """Size a SignalEvent into an OrderEvent (risk checks included)."""
        ...

    @abstractmethod
    def update_from_fill(self, fill):
        """Update the cash/holdings ledger from a FillEvent."""
        ...

    @abstractmethod
    def update_portfolio_valuation(self, market_event, events_queue=None):
        """Mark-to-market on close-phase MarketEvents; may emit liquidation orders."""
        ...

    @property
    @abstractmethod
    def equity_curve(self):
        ...

    @property
    @abstractmethod
    def positions_history(self):
        ...
