"""NBP (Narodowy Bank Polski) FX mid rates — free HTTPS API, no key."""

from __future__ import annotations

import pandas as pd

from quantester.data._http import http_get_json, import_requests

_NBP_BASE = "https://api.nbp.pl/api/exchangerates/rates"
_MAX_SPAN_DAYS = 93


def _chunk_ranges(start: pd.Timestamp, end: pd.Timestamp):
    """Yield inclusive [lo, hi] date pairs respecting the 93-day NBP limit."""
    lo = start.normalize()
    end = end.normalize()
    while lo <= end:
        hi = min(lo + pd.Timedelta(days=_MAX_SPAN_DAYS - 1), end)
        yield lo, hi
        lo = hi + pd.Timedelta(days=1)


def load_nbp_fx(
    code: str = "USD",
    start=None,
    end=None,
    *,
    table: str = "A",
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch NBP FX mid rates for ``code`` between ``start`` and ``end``.

    Uses table A (average mid) by default. ``start``/``end`` are required
    calendar dates; the client automatically chunks requests into ≤93-day
    windows. Returns a float Series of ``mid`` rates indexed by UTC dates
    (``effectiveDate``). Holidays with no table publication are simply absent.

    HTTPS-only base URL (required since Aug 2025).
    """
    if start is None or end is None:
        raise ValueError("load_nbp_fx requires both start= and end= dates.")
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")

    table = table.upper().strip()
    code = code.upper().strip()
    mids: list[float] = []
    dates: list[pd.Timestamp] = []

    for lo, hi in _chunk_ranges(start_ts, end_ts):
        url = (
            f"{_NBP_BASE}/{table}/{code}/"
            f"{lo.strftime('%Y-%m-%d')}/{hi.strftime('%Y-%m-%d')}/"
        )
        requests = import_requests()
        try:
            payload = http_get_json(
                url, params={"format": "json"}, timeout=timeout,
            )
        except requests.HTTPError as exc:
            # A window with no published rates returns 404; skip empty chunks.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                continue
            raise
        rates = payload.get("rates") or []
        for row in rates:
            dates.append(pd.Timestamp(row["effectiveDate"], tz="UTC"))
            if "mid" in row:
                mids.append(float(row["mid"]))
            elif "bid" in row and "ask" in row:
                mids.append(0.5 * (float(row["bid"]) + float(row["ask"])))
            else:
                raise ValueError(
                    f"NBP rate row missing mid/bid/ask: {row!r}"
                )

    if not mids:
        raise ValueError(
            f"NBP returned no rates for {code!r} table {table} between "
            f"{start_ts.date()} and {end_ts.date()}."
        )
    series = pd.Series(
        mids,
        index=pd.DatetimeIndex(dates, name="datetime"),
        name=f"NBP:{table}:{code}",
        dtype=float,
    )
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()
