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

from ..events import EXIT, LONG, SHORT, SignalEvent


class Strategy(ABC):
    delay: int = 1
    fill_at: str = "open"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Soft documentation hook — concrete strategies still set delay on the
        # instance in __init__. Validated at emit time / engine start.

    @abstractmethod
    def calculate_signals(self, event, events_queue):
        """Listen to MarketEvents, process data, generate SignalEvents."""
        ...

    def matches_phase(self, phase: str) -> bool:
        """delay=0 strategies consume open-phase events; delay>=1 close-phase."""
        delay = getattr(self, "delay", 1)
        if delay == 0:
            return phase == "open"
        return phase == "close"

    def vectorized_signals(self, data: dict):
        """Vectorized twin: full-history target positions (per symbol) for the
        Monte Carlo fast-track. Not every strategy has a closed-form twin."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a vectorized twin "
            "(vectorized_signals). Without it you cannot run Monte Carlo "
            "fast-track checks (MCPT). Either implement the twin so it matches "
            "calculate_signals exactly, or stick to event-loop validation only. "
            "See docs/tutorials/creating-a-strategy.md step 3c."
        )

    def emit_target(
        self,
        events_queue,
        timestamp,
        symbol: str,
        target: float,
        *,
        strength: float | None = None,
        position_attr: str = "_position",
    ) -> bool:
        """Emit LONG / SHORT / EXIT only when the target changes.

        This is the safe default for custom strategies: re-emitting the same
        target every bar creates redundant orders and burns commissions.

        Parameters
        ----------
        target
            Desired position sign/size scale: ``>0`` long, ``<0`` short, ``0`` flat.
        strength
            Scales the sizer (keep in ``(0, 1]``). Defaults to ``abs(target)``
            when target is non-zero, else ``1.0``.
        position_attr
            Instance attribute that stores the last emitted target
            (default ``_position``).

        Returns
        -------
        bool
            True if a signal was emitted, False if the target was unchanged.
        """
        delay = getattr(self, "delay", 1)
        if not isinstance(delay, int) or delay < 0:
            raise ValueError(
                f"{type(self).__name__}.delay must be an integer >= 0 "
                f"(0 = fill at this bar's open, 1 = fill at next bar's open). "
                f"Got {delay!r}."
            )
        current = float(getattr(self, position_attr, 0.0))
        target = float(target)
        if target == current:
            return False
        if target == 0.0:
            signal_type = EXIT
            sig_strength = 1.0 if strength is None else float(strength)
        elif target > 0:
            signal_type = LONG
            sig_strength = abs(target) if strength is None else float(strength)
        else:
            signal_type = SHORT
            sig_strength = abs(target) if strength is None else float(strength)
        events_queue.put(
            SignalEvent(
                timestamp,
                symbol,
                signal_type,
                strength=sig_strength,
                delay=delay,
                fill_at=getattr(self, "fill_at", "open"),
            )
        )
        setattr(self, position_attr, target)
        return True
