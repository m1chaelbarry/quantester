"""DataHandler abstract base class.

The look-ahead firewall: downstream components may ONLY observe market data
through this interface. Visibility is per-timestamp, not per-bar (Cross-Ref-2
section 3.B): during a bar's open phase, get_latest_bars excludes that bar and
only its open print is separately available; during the close phase the full bar
becomes visible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager

import pandas as pd


class DataHandler(ABC):
    @property
    @abstractmethod
    def symbols(self) -> list:
        ...

    @property
    @abstractmethod
    def continue_backtest(self) -> bool:
        ...

    @abstractmethod
    def prime_data(self) -> None:
        """Reset the stream to just before the first bar."""
        ...

    @abstractmethod
    def advance(self) -> tuple:
        """Move to the next timestamp. Returns (timestamp, bars dict).

        bars maps symbol -> OHLCV row (pd.Series) or None when the symbol is
        untradeable at this timestamp (availability mask; timestamps are never
        erased, Cross-Ref-2 section 4.3).
        """
        ...

    @abstractmethod
    def set_phase(self, phase: str, timestamp: pd.Timestamp) -> None:
        """Set the temporal-firewall context ('open' or 'close') for visibility."""
        ...

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Trailing n bars visible under the current firewall context."""
        ...

    @abstractmethod
    def get_current_open(self, symbol: str):
        """Current bar's open print (open phase only); None if untradeable."""
        ...

    @abstractmethod
    def timestamp_at_offset(self, timestamp: pd.Timestamp, n: int):
        """Timestamp n bars after `timestamp` on the master calendar (None past the end)."""
        ...

    @abstractmethod
    def bar_at(self, symbol: str, timestamp: pd.Timestamp):
        """Execution-side lookup of a full bar at a timestamp (None if unavailable)."""
        ...

    @contextmanager
    def seal_source_ohlcv(self):
        """Block ``source_ohlcv`` while strategies generate signals.

        Research scripts and post-run analysis keep the accessor; the engine
        holds this seal only around ``calculate_signals``. Nested seals stay
        closed until the outermost context exits.
        """
        self._source_ohlcv_seal_depth = getattr(self, "_source_ohlcv_seal_depth", 0) + 1
        try:
            yield
        finally:
            self._source_ohlcv_seal_depth -= 1

    def source_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Full loaded OHLCV frame for research scripts (not for live signals).

        Strategies must use ``get_latest_bars`` under the temporal firewall.
        Example scripts and post-run analysis should call this instead of
        reaching into private ``_data``.

        Raises ``PermissionError`` when called from ``calculate_signals``.
        """
        self._ensure_source_ohlcv_unsealed()
        return self._source_ohlcv(symbol)

    def _ensure_source_ohlcv_unsealed(self) -> None:
        if getattr(self, "_source_ohlcv_seal_depth", 0) > 0:
            raise PermissionError(
                "source_ohlcv() is sealed during calculate_signals; "
                "read market data with get_latest_bars() / get_current_open() "
                "under the temporal firewall. source_ohlcv() is for research "
                "and post-run analysis only."
            )

    def _source_ohlcv(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose source_ohlcv(); "
            "use a StreamingDataHandler-based feed or load frames directly."
        )

    def corporate_actions_at(self, timestamp) -> list:
        """Corporate-action events (dividend cash / split quantity) whose
        ex-date is ``timestamp`` — routed onto the event queue by the engine
        at the bar's open, before any fill or valuation (ruling D9).         Feeds
        without a corporate-action schedule return an empty list."""
        return []

    def funding_settlements_at(self, timestamp) -> list:
        """Funding Settlement events for ``timestamp`` (daily-bar extras).

        Routed by the engine at close, before valuation. Feeds without a
        ``funding_rate`` extra return an empty list. NaN rates are skipped.
        """
        return []

    @property
    @abstractmethod
    def current_timestamp(self):
        ...
