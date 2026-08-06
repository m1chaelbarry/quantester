"""DataHandler abstract base class.

The look-ahead firewall: downstream components may ONLY observe market data
through this interface. Visibility is per-timestamp, not per-bar (Cross-Ref-2
section 3.B): during a bar's open phase, get_latest_bars excludes that bar and
only its open print is separately available; during the close phase the full bar
becomes visible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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

    @property
    @abstractmethod
    def current_timestamp(self):
        ...
