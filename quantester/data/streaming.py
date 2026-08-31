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
open/high/low/close/volume columns, unique sorted **timezone-aware UTC**
datetime index. Optional extra columns (funding_rate, open_interest, dvol, …)
pass through and share the same temporal-firewall visibility as close.
Provider adapters localize/convert at the ingestion boundary so the core
engine never mixes exchange-local naive, UTC-naive, and aware timestamps.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd

from .base import DataHandler

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_ohlcv_frame(
    df: pd.DataFrame,
    symbol: str = "",
    *,
    on_duplicates: str = "raise",
) -> pd.DataFrame:
    """Coerce a raw frame to the normalized contract; raises ValueError listing
    any missing columns so provider wiring fails fast with a clear message.

    Timestamps are normalized to timezone-aware UTC. Tz-naive inputs are
    localized as UTC (callers that hold exchange-local wall times must convert
    explicitly in the provider adapter before reaching this helper).

    Duplicate timestamps fail by default (``on_duplicates='raise'``). Pass
    ``on_duplicates='keep_first'`` or ``'keep_last'`` only when the caller has
    explicitly chosen a reconciliation policy.
    """
    from .audit import ensure_utc_index

    if on_duplicates not in {"raise", "keep_first", "keep_last"}:
        raise ValueError(
            "on_duplicates must be 'raise', 'keep_first', or 'keep_last'"
        )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Frame for {symbol or '<unknown symbol>'} is missing required "
            f"OHLCV columns {missing}; got {list(df.columns)}."
        )
    if df.index.duplicated().any():
        n_dup = int(df.index.duplicated().sum())
        if on_duplicates == "raise":
            raise ValueError(
                f"Frame for {symbol or '<unknown symbol>'} has {n_dup} "
                "duplicate timestamp(s); pass on_duplicates='keep_first' or "
                "'keep_last' to choose an explicit reconciliation policy."
            )
        keep = "first" if on_duplicates == "keep_first" else "last"
        df = df[~df.index.duplicated(keep=keep)]
    out = df.sort_index()
    out.index = ensure_utc_index(out.index)
    ohlcv = out[list(REQUIRED_COLUMNS)].astype(float)
    extras = [c for c in out.columns if c not in REQUIRED_COLUMNS]
    if not extras:
        return ohlcv
    extra_df = out[extras].copy()
    for col in extras:
        extra_df[col] = pd.to_numeric(extra_df[col], errors="coerce")
    return pd.concat([ohlcv, extra_df], axis=1)


class StreamingDataHandler(DataHandler):
    """DataHandler over a {symbol: normalized OHLCV DataFrame} map.

    ``corporate_actions`` (optional): ``{symbol: DataFrame}`` whose index is
    the ex-date timestamps (normalized to UTC like the bars) with a
    ``dividend`` and/or ``split`` column; nonzero rows become
    ``CorporateActionEvent``s routed onto the queue at that bar's open
    (ruling D9). Bars stay RAW — the cash/quantity effects book on the
    portfolio ledger, never inside the price series.
    """

    def __init__(self, frames: dict, corporate_actions: dict | None = None):
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
        self._ca_by_ts: dict = {}
        if corporate_actions:
            self.set_corporate_actions(corporate_actions)

    def set_corporate_actions(self, corporate_actions: dict) -> None:
        """Register/replace the ex-date corporate-action schedule.

        Ex-dates that match no bar timestamp for the symbol are dropped with
        a warning (an ex-date must land on a bar to book against it).
        """
        from ..events import CorporateActionEvent

        by_ts: dict = {}
        for symbol, frame in corporate_actions.items():
            if symbol not in self._data:
                raise ValueError(
                    f"corporate_actions for unknown symbol {symbol!r}; "
                    f"known={self._symbols}"
                )
            ca = frame.copy()
            idx = pd.DatetimeIndex(ca.index)
            ca.index = idx if idx.tz is not None else idx.tz_localize("UTC")
            bar_index = self._data[symbol].index
            unmatched = ca.index.difference(bar_index)
            if len(unmatched):
                warnings.warn(
                    f"{symbol}: {len(unmatched)} corporate-action ex-date(s) "
                    f"match no bar timestamp and are dropped: "
                    f"{[str(t.date()) for t in unmatched]}",
                    UserWarning,
                    stacklevel=2,
                )
                ca = ca.loc[ca.index.intersection(bar_index)]
            for ts, row in ca.iterrows():
                events = []
                dividend = float(row.get("dividend", 0.0) or 0.0)
                split = float(row.get("split", 0.0) or 0.0)
                if dividend > 0:
                    events.append(
                        CorporateActionEvent(ts, symbol, "dividend",
                                             dividend_per_share=dividend)
                    )
                if split > 0:
                    events.append(
                        CorporateActionEvent(ts, symbol, "split",
                                             split_ratio=split)
                    )
                if events:
                    by_ts.setdefault(ts, []).extend(events)
        self._ca_by_ts = by_ts

    def corporate_actions_at(self, timestamp) -> list:
        return list(self._ca_by_ts.get(timestamp, ()))

    @property
    def symbols(self) -> list:
        return self._symbols

    @property
    def n_bars(self) -> int:
        """Number of timestamps on the master (outer-join) calendar."""
        return len(self._master_index)

    @property
    def first_timestamp(self):
        """First master-calendar timestamp, or None if empty."""
        return None if self._master_index.empty else self._master_index[0]

    @property
    def last_timestamp(self):
        """Last master-calendar timestamp, or None if empty."""
        return None if self._master_index.empty else self._master_index[-1]

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

    def funding_settlements_at(self, timestamp) -> list:
        """Emit Funding Settlements from the current bar's ``funding_rate`` extra."""
        from ..events import FundingSettlementEvent

        events = []
        for symbol, bar in self._bars.items():
            if bar is None:
                continue
            if "funding_rate" not in bar.index:
                continue
            rate = bar["funding_rate"]
            if pd.isna(rate):
                continue
            rate_f = float(rate)
            if not math.isfinite(rate_f):
                continue
            events.append(
                FundingSettlementEvent(
                    timestamp, symbol, rate_f, float(bar["close"])
                )
            )
        return events

    def _source_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Return a copy of the loaded OHLCV frame for ``symbol``."""
        if symbol not in self._data:
            raise KeyError(f"unknown symbol {symbol!r}; known={list(self._data)}")
        return self._data[symbol].copy()
