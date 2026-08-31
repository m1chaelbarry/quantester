"""Daily aggregation of perpetual extras (funding, OI, DVOL).

8-hour funding prints are summed onto UTC days (the three settlements the
exchange actually charges). OI and DVOL last-print onto the same daily index.
"""

from __future__ import annotations

import pandas as pd


def to_utc_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def daily_sum(series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Sum sub-daily prints onto UTC calendar days, aligned to ``daily_index``."""
    if series is None or len(series) == 0:
        return pd.Series(float("nan"), index=daily_index)
    s = series.copy()
    s.index = to_utc_index(s.index)
    grouped = s.groupby(s.index.floor("D")).sum()
    keys = to_utc_index(daily_index).floor("D")
    return grouped.reindex(keys).set_axis(daily_index)


def daily_last(series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(float("nan"), index=daily_index)
    s = series.copy()
    s.index = to_utc_index(s.index)
    grouped = s.groupby(s.index.floor("D")).last()
    keys = to_utc_index(daily_index).floor("D")
    return grouped.reindex(keys).set_axis(daily_index)


def attach_extras(
    ohlcv: pd.DataFrame,
    *,
    funding: pd.Series | None = None,
    open_interest: pd.Series | None = None,
    dvol: pd.Series | None = None,
) -> pd.DataFrame:
    """Join extras onto an OHLCV frame. Missing series stay absent (not zeros)."""
    out = ohlcv.copy()
    idx = out.index
    if funding is not None:
        out["funding_rate"] = daily_sum(funding, idx)
    if open_interest is not None:
        out["open_interest"] = daily_last(open_interest, idx)
    if dvol is not None:
        out["dvol"] = daily_last(dvol, idx)
    return out
