"""Strategy abstract base class (interface verbatim from Report 1 section 2.2).

A strategy is a sandboxed, state-free mathematical engine interacting exclusively
with standardized containers (Bar/Event), enabling research-to-live parity.

Temporal-firewall contract:
- `delay` bars until execution (1 = classic T+1, 0 = Delay-0 at the bar's open).
- delay=0 strategies act on open-phase MarketEvents and only see data strictly
  before the fill timestamp (intra-bar guard, enforced by the DataHandler).
- Strategies that support Monte Carlo fast-track validation must also declare a
  vectorized twin (`vectorized_signals`) producing target positions from full
  price history; a parity test proves twin and event-driven forms agree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Strategy(ABC):
    delay: int = 1
    fill_at: str = "open"

    @abstractmethod
    def calculate_signals(self, event, events_queue):
        """Listen to MarketEvents, process data, generate SignalEvents."""
        ...

    def matches_phase(self, phase: str) -> bool:
        """delay=0 strategies consume open-phase events; delay>=1 close-phase."""
        if self.delay == 0:
            return phase == "open"
        return phase == "close"

    def vectorized_signals(self, data: dict):
        """Vectorized twin: full-history target positions (per symbol) for the
        Monte Carlo fast-track. Not every strategy has a closed-form twin."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a vectorized twin; "
            "Monte Carlo fast-track validation is unavailable for it."
        )
