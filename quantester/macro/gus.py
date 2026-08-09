"""GUS BDL (Statistics Poland) variable time series — free REST API."""

from __future__ import annotations

import pandas as pd

from quantester.data._http import import_requests, resolve_api_key

_BDL_URL = "https://bdl.stat.gov.pl/api/v1/data/by-variable/{variable_id}"
_ENV_KEY = "QUANTESTER_GUS_API_KEY"


def load_gus_variable(
    variable_id: int | str,
    *,
    years: list[int] | None = None,
    unit_level: int = 0,
    unit_id: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch one BDL variable as a UTC year-start Series.

    ``variable_id``: BDL variable id (integer or digit string).
    ``years``: optional list of years; when omitted the API returns its default
        recent window.
    ``unit_level``: territorial level (0 = Poland aggregate in many series).
    ``unit_id``: optional specific territorial unit id; when set, that unit's
        values are selected. Otherwise the first result row is used.
    ``api_key``: optional ``X-ClientId`` (or ``QUANTESTER_GUS_API_KEY``) for
        higher quotas; anonymous access is allowed.

    Returns a float Series named ``GUS:{variable_id}``.
    """
    key = resolve_api_key(
        api_key, env_var=_ENV_KEY, required=False, provider="GUS",
    )
    params: list[tuple[str, str]] = [("format", "json"),
                                     ("unit-level", str(int(unit_level)))]
    if years:
        for y in years:
            params.append(("year", str(int(y))))

    headers = {}
    if key:
        headers["X-ClientId"] = key

    url = _BDL_URL.format(variable_id=variable_id)
    # requests accepts a list of pairs for repeated query keys (year=…).
    requests = import_requests()
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    results = payload.get("results") or []
    if not results:
        raise ValueError(
            f"GUS BDL returned no results for variable {variable_id!r}."
        )

    row = None
    if unit_id is not None:
        for candidate in results:
            if str(candidate.get("id")) == str(unit_id):
                row = candidate
                break
        if row is None:
            raise ValueError(
                f"GUS BDL variable {variable_id!r} has no unit_id={unit_id!r}; "
                f"got {[r.get('id') for r in results][:8]}…"
            )
    else:
        row = results[0]

    values = row.get("values") or []
    dates: list[pd.Timestamp] = []
    vals: list[float] = []
    for item in values:
        if item is None or item.get("val") is None:
            continue
        year = item.get("year")
        dates.append(pd.Timestamp(f"{year}-01-01", tz="UTC"))
        vals.append(float(item["val"]))
    if not vals:
        raise ValueError(
            f"GUS BDL variable {variable_id!r} had no numeric values."
        )
    series = pd.Series(
        vals,
        index=pd.DatetimeIndex(dates, name="datetime"),
        name=f"GUS:{variable_id}",
        dtype=float,
    )
    return series.sort_index()
