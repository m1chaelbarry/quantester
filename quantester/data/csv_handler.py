"""HistoricCSVDataHandler: per-symbol OHLCV CSVs over a unified master calendar.

Multi-symbol alignment uses an outer-join timestamp union with per-symbol
availability masks: a missing bar marks the asset untradeable at that timestamp
instead of erasing the timestamp (Cross-Ref-2 section 4.3 supersedes Report 1's
incomplete-bar dropping rule, which deletes high-stress/illiquid periods and
induces selection bias).

CSV schema: datetime,open,high,low,close,volume
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class HistoricCSVDataHandler:
    def __init__(self, csv_map: dict):
        """csv_map: symbol -> path to CSV (or pre-loaded DataFrame indexed by datetime)."""
        self._symbols = list(csv_map.keys())
        self._data = {}
        for symbol, source in csv_map.items():
            if isinstance(source, pd.DataFrame):
                df = source.copy()
            else:
                df = pd.read_csv(Path(source), parse_dates=["datetime"], index_col="datetime")
            df = df[~df.index.duplicated(keep="first")].sort_index()
            self._data[symbol] = df[["open", "high", "low", "close", "volume"]].astype(float)

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
