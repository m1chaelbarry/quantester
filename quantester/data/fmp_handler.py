"""FMPDataHandler: free EOD OHLCV from Financial Modeling Prep (stable API).

Data is downloaded once at construction (historic batch replay; not a live
feed), normalized to the StreamingDataHandler contract, and streamed under the
same temporal firewall and availability-mask semantics as HistoricCSVDataHandler.

Requires ``pip install "quantester[data]"`` and a free FMP API key. Pass
``api_key=`` or set ``QUANTESTER_FMP_API_KEY``.

FMP stable contract (official docs, Aug 2026):
- ``GET https://financialmodelingprep.com/stable/historical-price-eod/full``
- Query: ``symbol``, optional ``from`` / ``to`` (YYYY-MM-DD), ``apikey``
- JSON array of ``{date, open, high, low, close, volume, ...}`` (often
  newest-first). Free tier ~250 req/day and roughly ~5 years EOD depth.

Daily session dates are UTC-localized. Retrieval only — no quantitative
formulas.
"""

from __future__ import annotations

import pandas as pd

from ._http import http_get_json, resolve_api_key
from .streaming import StreamingDataHandler

_FMP_EOD_URL = (
    "https://financialmodelingprep.com/stable/historical-price-eod/full"
)
_ENV_KEY = "QUANTESTER_FMP_API_KEY"


def _fmt_iso_date(value) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _normalize_fmp_rows(rows, symbol: str) -> pd.DataFrame:
    if not isinstance(rows, list):
        raise ValueError(
            f"FMP returned unexpected payload for {symbol!r}: "
            f"{type(rows).__name__} (expected a JSON list of bars)."
        )
    if not rows:
        raise ValueError(
            f"FMP returned no EOD rows for {symbol!r}; check the ticker, "
            "date range, and free-tier history depth."
        )
    if isinstance(rows[0], dict) and "Error Message" in rows[0]:
        raise ValueError(f"FMP error for {symbol!r}: {rows[0]['Error Message']}")
    df = pd.DataFrame(rows)
    # Error payloads sometimes arrive as a single dict wrapped by callers.
    needed = {"date", "open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"FMP EOD for {symbol!r} missing columns {sorted(missing)}; "
            f"got {list(df.columns)}."
        )
    idx = pd.to_datetime(df["date"])
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(idx, name="datetime").tz_localize("UTC")
    return out.sort_index()


def _download_ohlcv(
    symbol: str,
    *,
    start=None,
    end=None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch one FMP symbol; separated for testability (monkeypatch here)."""
    key = resolve_api_key(
        api_key, env_var=_ENV_KEY, required=True, provider="FMP",
    )
    params: dict = {"symbol": symbol, "apikey": key}
    d_from, d_to = _fmt_iso_date(start), _fmt_iso_date(end)
    if d_from is not None:
        params["from"] = d_from
    if d_to is not None:
        params["to"] = d_to
    payload = http_get_json(_FMP_EOD_URL, params=params, timeout=timeout)
    if isinstance(payload, dict):
        # Some error responses are a single JSON object.
        if "Error Message" in payload:
            raise ValueError(
                f"FMP error for {symbol!r}: {payload['Error Message']}"
            )
        # Occasionally wrapped as {"historical": [...]} on legacy paths.
        if "historical" in payload:
            payload = payload["historical"]
    return _normalize_fmp_rows(payload, symbol)


class FMPDataHandler(StreamingDataHandler):
    """Stream Financial Modeling Prep EOD OHLCV through the event engine.

    symbols: ticker or list (e.g. ``\"AAPL\"`` or ``[\"AAPL\", \"MSFT\"]``).
    start/end: optional ISO date bounds (``from`` / ``to``).
    api_key: free FMP key; defaults to ``QUANTESTER_FMP_API_KEY``.
    """

    def __init__(
        self,
        symbols,
        start=None,
        end=None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        if isinstance(symbols, str):
            symbols = [symbols]
        frames = {
            symbol: _download_ohlcv(
                symbol,
                start=start,
                end=end,
                api_key=api_key,
                timeout=timeout,
            )
            for symbol in symbols
        }
        super().__init__(frames)
