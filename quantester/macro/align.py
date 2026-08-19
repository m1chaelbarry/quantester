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

    ``method='bfill'`` is refused outright: backward-filling pulls future
    macro prints into past bars, a look-ahead leak when the join feeds
    trading features (3rd-cross-reference synthesis §1.10).

    Both sides are normalized to timezone-aware UTC before aligning. This is a
    calendar join only — no quantitative formula to notebook-verify.
    """
    if method == "bfill":
        raise ValueError(
            "method='bfill' is forbidden on trading-feature joins: it leaks "
            "future observations into past bars (look-ahead). Use 'ffill' "
            "(causal) or None (explicit NaNs)."
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
