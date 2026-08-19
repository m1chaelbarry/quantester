"""DataHandler abstract base class.

The look-ahead firewall: downstream components may ONLY observe market data
through this interface. Visibility is per-timestamp, not per-bar (Cross-Ref-2
section 3.B): during a bar's open phase, get_latest_bars excludes that bar and
only its open print is separately available; during the close phase the full bar
becomes visible.
"""

from __future__ import annotations

import contextlib
import warnings
from abc import ABC, abstractmethod

import pandas as pd


class DataHandler(ABC):
    # Seal switch (synthesis §5.2): False warns when source_ohlcv() is called
    # from inside Strategy.calculate_signals; True raises PermissionError.
    seal_source_ohlcv: bool = False

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

    def source_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Full loaded OHLCV frame for research scripts (not for live signals).

        Strategies must use ``get_latest_bars`` under the temporal firewall.
        Example scripts and post-run analysis should call this instead of
        reaching into private ``_data``.

        Implementations that expose the raw frame MUST call
        ``_check_source_ohlcv_access()`` first so firewall violations during
        signal dispatch are surfaced (warned, or sealed via
        ``seal_source_ohlcv = True``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose source_ohlcv(); "
            "use a StreamingDataHandler-based feed or load frames directly."
        )

    @contextlib.contextmanager
    def signal_scope(self):
        """Mark Strategy.calculate_signals dispatch on this handler.

        The engine wraps every ``calculate_signals`` call in this scope so
        that firewall-bypassing raw-frame reads (``source_ohlcv``) can be
        detected at the exact moment they leak.
        """
        depth = getattr(self, "_signal_scope_depth", 0)
        self._signal_scope_depth = depth + 1
        try:
            yield
        finally:
            self._signal_scope_depth = depth

    def _check_source_ohlcv_access(self) -> None:
        """Warn (or raise, when sealed) if called inside ``signal_scope``."""
        if getattr(self, "_signal_scope_depth", 0) <= 0:
            return
        msg = (
            f"{type(self).__name__}.source_ohlcv() was called from "
            "Strategy.calculate_signals: the raw frame bypasses the temporal "
            "firewall (look-ahead risk). Strategies must read through "
            "get_latest_bars()/get_current_open(); source_ohlcv() is for "
            "post-run research only."
        )
        if getattr(self, "seal_source_ohlcv", False):
            raise PermissionError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    @property
    @abstractmethod
    def current_timestamp(self):
        ...
