"""StreamingDataHandler: reusable streaming engine over pre-loaded OHLCV frames.

All historic data feeds (CSV, yfinance, ccxt, ...) converge on the same
in-memory representation -- one normalized DataFrame per symbol -- and share
the look-ahead firewall implemented here:

- Outer-join master calendar with per-symbol availability masks: a missing bar
  marks the asset untradeable at that timestamp instead of erasing the
  timestamp (Cross-Ref-2 section 4.3 supersedes Report 1's incomplete-bar
  dropping rule, which deletes high-stress/illiquid periods and induces
  selection bias).
- Phase-aware visibility (Cross-Ref-2 section 3.B): during a bar's open phase,
  get_latest_bars excludes that bar and only its open print is separately
  available via get_current_open; during the close phase the full bar becomes
  visible.

Normalized frame contract (enforced at construction): lowercase
open/high/low/close/volume columns, unique sorted tz-naive datetime index
(naive == UTC for exchange-provided timestamps).
"""

from __future__ import annotations

import pandas as pd

from .base import DataHandler

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_ohlcv_frame(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Coerce a raw frame to the normalized contract; raises ValueError listing
    any missing columns so provider wiring fails fast with a clear message."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Frame for {symbol or '<unknown symbol>'} is missing required "
            f"OHLCV columns {missing}; got {list(df.columns)}."
        )
    out = df[~df.index.duplicated(keep="first")].sort_index()
    # Canonical ns resolution: provider frames arrive as datetime64[s/ms/us]
    # (pandas 3 infers resolution from source), which must not leak into
    # cross-provider index comparisons.
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    return out[list(REQUIRED_COLUMNS)].astype(float)


class StreamingDataHandler(DataHandler):
    """DataHandler over a {symbol: normalized OHLCV DataFrame} map."""

    def __init__(self, frames: dict):
        if not frames:
            raise ValueError("StreamingDataHandler requires at least one symbol.")
        self._symbols = list(frames.keys())
        self._data = {
            symbol: normalize_ohlcv_frame(df, symbol) for symbol, df in frames.items()
        }

        # Outer join: union of every symbol's timestamps; nothing is dropped.
        master = sorted(set().union(*[set(df.index) for df in self._data.values()]))
        self._master_index = pd.DatetimeIndex(master)
        self._position_of = {ts: i for i, ts in enumerate(self._master_index)}

        self._ptr = -1
        self._ts = None
        self._phase = "close"
        self._bars = {}

    @property
    def symbols(self) -> list:
        return self._symbols

    @property
    def current_timestamp(self):
        return self._ts

    @property
    def continue_backtest(self) -> bool:
        return self._ptr < len(self._master_index) - 1

    def prime_data(self) -> None:
        self._ptr = -1
        self._ts = None
        self._bars = {}

    def advance(self) -> tuple:
        if not self.continue_backtest:
            raise IndexError("No further bars to stream.")
        self._ptr += 1
        self._ts = self._master_index[self._ptr]
        self._bars = {}
        for symbol, df in self._data.items():
            if self._ts in df.index:
                self._bars[symbol] = df.loc[self._ts]
            else:
                self._bars[symbol] = None  # availability mask: untradeable, not erased
        return self._ts, self._bars

    def set_phase(self, phase: str, timestamp: pd.Timestamp) -> None:
        self._phase = phase
        self._ts = timestamp

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        df = self._data[symbol]
        if self._ts is None:
            return df.iloc[0:0]
        if self._phase == "open":
            # Intra-bar guard: only bars strictly before the current one are visible.
            visible = df.loc[df.index < self._ts]
        else:
            visible = df.loc[df.index <= self._ts]
        return visible.tail(n)

    def get_current_open(self, symbol: str):
        bar = self._bars.get(symbol)
        return None if bar is None else float(bar["open"])

    def timestamp_at_offset(self, timestamp: pd.Timestamp, n: int):
        idx = self._position_of.get(timestamp)
        if idx is None:
            return None
        target = idx + n
        if target >= len(self._master_index):
            return None
        return self._master_index[target]

    def bar_at(self, symbol: str, timestamp: pd.Timestamp):
        """Execution-side lookup of a full bar at a timestamp (None if unavailable)."""
        df = self._data[symbol]
        if timestamp in df.index:
            return df.loc[timestamp]
        return None
