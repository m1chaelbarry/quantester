"""World Bank Open Data indicator series (REST v2, no API key)."""

from __future__ import annotations

import pandas as pd

from quantester.data._http import http_get_json

_WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


def load_world_bank(
    indicator: str,
    country: str = "USA",
    *,
    start: int | None = None,
    end: int | None = None,
    per_page: int = 20_000,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch one World Bank indicator for one country as a UTC-dated Series.

    ``indicator``: e.g. ``\"FP.CPI.TOTL.ZG\"`` (inflation) or ``\"SP.POP.TOTL\"``.
    ``country``: ISO2/ISO3 code accepted by the API (``\"USA\"``, ``\"PL\"``, …).
    ``start`` / ``end``: optional integer years (``date=START:END`` query).

    Returns a float Series named ``{country}:{indicator}`` indexed by
    year-start timestamps (UTC). Null World Bank observations are dropped.
    """
    params: dict = {"format": "json", "per_page": int(per_page)}
    if start is not None or end is not None:
        lo = int(start if start is not None else 1960)
        hi = int(end if end is not None else 2100)
        params["date"] = f"{lo}:{hi}"

    url = _WB_URL.format(country=country, indicator=indicator)
    payload = http_get_json(url, params=params, timeout=timeout)
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(
            f"World Bank returned an unexpected payload for "
            f"{country!r}/{indicator!r}."
        )
    rows = payload[1] or []
    if not rows:
        raise ValueError(
            f"World Bank returned no observations for "
            f"{country!r}/{indicator!r}."
        )

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows:
        if row is None or row.get("value") is None:
            continue
        # Annual (or period) label like "2020"; localize as Jan 1 UTC.
        dates.append(pd.Timestamp(f"{row['date']}-01-01", tz="UTC"))
        values.append(float(row["value"]))
    if not values:
        raise ValueError(
            f"World Bank observations for {country!r}/{indicator!r} were "
            "all null."
        )
    series = pd.Series(values, index=pd.DatetimeIndex(dates, name="datetime"),
                       name=f"{country}:{indicator}", dtype=float)
    return series.sort_index()
