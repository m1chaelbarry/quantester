"""Align a lower-frequency macro/FX series onto a daily bar calendar."""

from __future__ import annotations

import pandas as pd


def as_daily_reindex(
    calendar,
    series: pd.Series | pd.DataFrame,
    *,
    method: str = "ffill",
) -> pd.Series | pd.DataFrame:
    """Reindex ``series`` onto ``calendar`` (DatetimeIndex or OHLCV-like index).

    Default ``method='ffill'`` carries the last known macro observation forward
    onto each trading day (standard exogenous-feature join). Pass
    ``method=None`` for a left join with NaNs on non-observation days.

    ``method='bfill'`` is rejected: pulling a later print onto earlier bars is
    look-ahead if the aligned series is used as a trading feature.

    Both sides are normalized to timezone-aware UTC before aligning. This is a
    calendar join only — no quantitative formula to notebook-verify.
    """
    if method == "bfill":
        raise ValueError(
            "method='bfill' is look-ahead on a trading-feature join: a later "
            "macro print would leak into earlier bars. Use method='ffill' "
            "(causal) or method=None (NaNs on non-observation days)."
        )
    if method not in {"ffill", None}:
        raise ValueError("method must be 'ffill' or None")

    cal = pd.DatetimeIndex(calendar)
    if cal.tz is None:
        cal = cal.tz_localize("UTC")
    else:
        cal = cal.tz_convert("UTC")
    cal = cal.sort_values()

    if isinstance(series, pd.Series):
        data = series.copy()
    elif isinstance(series, pd.DataFrame):
        data = series.copy()
    else:
        raise TypeError("series must be a pandas Series or DataFrame")

    idx = pd.DatetimeIndex(data.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    data.index = idx
    data = data.sort_index()
    # Drop duplicate stamps (keep last publication) before reindex.
    data = data[~data.index.duplicated(keep="last")]

    if method is None:
        return data.reindex(cal)
    return data.reindex(cal, method=method)
