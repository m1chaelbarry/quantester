"""StooqDataHandler: free daily/intraday OHLCV CSV from Stooq.

Data is downloaded once at construction (historic batch replay; not a live
feed), normalized to the StreamingDataHandler contract, and streamed under the
same temporal firewall and availability-mask semantics as HistoricCSVDataHandler.

Requires ``pip install "quantester[data]"`` (for ``requests``) and a free Stooq
API key (CAPTCHA form on https://stooq.com/q/d/?s=spy.us&get_apikey). Pass
``api_key=`` or set ``QUANTESTER_STOOQ_API_KEY``.

Stooq CSV contract (portal docs / download URL):
- URL: ``https://stooq.com/q/d/l/?s={symbol}&d1=YYYYMMDD&d2=YYYYMMDD&i=d&apikey=``
- Columns: Date,Open,High,Low,Close,Volume (often newest-first)
- Symbols use exchange suffixes (``.us``, ``.uk``, ``.de``, …); FX/crypto often
  ``.v`` (e.g. ``eurusd.v``)
- Quota exhaustion may return HTTP 200 with body text
  ``Exceeded the daily hits limit`` instead of CSV — we detect that explicitly.

Daily session dates are UTC-localized (same convention as YFinanceDataHandler
daily bars). This module implements retrieval only — no quantitative formulas.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd

from ._http import http_get_text, resolve_api_key
from .streaming import StreamingDataHandler

_STOOQ_CSV_URL = "https://stooq.com/q/d/l/"
_ENV_KEY = "QUANTESTER_STOOQ_API_KEY"
_OHLCV_MAP = {
    "Date": "datetime",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def _fmt_stooq_date(value) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.strftime("%Y%m%d")


def _normalize_stooq_csv(text: str, symbol: str) -> pd.DataFrame:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"Stooq returned an empty body for {symbol!r}.")
    if "Exceeded the daily hits limit" in stripped:
        raise ValueError(
            f"Stooq daily hit limit exceeded while fetching {symbol!r}; "
            "retry tomorrow or reduce request volume."
        )
    # Non-CSV error pages (HTML / plain text) lack the Date header.
    head = stripped.splitlines()[0].lower()
    if "date" not in head:
        raise ValueError(
            f"Stooq returned a non-CSV response for {symbol!r}: "
            f"{stripped[:160]!r}"
        )
    raw = pd.read_csv(StringIO(stripped))
    rename = {c: _OHLCV_MAP[c] for c in raw.columns if c in _OHLCV_MAP}
    df = raw.rename(columns=rename)
    missing = [c for c in ("datetime", "open", "high", "low", "close", "volume")
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stooq CSV for {symbol!r} missing columns {missing}; "
            f"got {list(raw.columns)}."
        )
    if df.empty:
        raise ValueError(
            f"Stooq returned no rows for {symbol!r}; check the ticker "
            "(include suffix, e.g. 'aapl.us') and date range."
        )
    idx = pd.to_datetime(df["datetime"])
    # Daily/calendar stamps: localize as UTC labels (exchange session date).
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = df.drop(columns=["datetime"])
    df.index = pd.DatetimeIndex(idx, name="datetime").tz_localize("UTC")
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _download_ohlcv(
    symbol: str,
    *,
    start=None,
    end=None,
    interval: str = "d",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch one Stooq symbol; separated for testability (monkeypatch here)."""
    key = resolve_api_key(
        api_key, env_var=_ENV_KEY, required=True, provider="Stooq",
    )
    params: dict = {"s": symbol.lower(), "i": interval, "apikey": key}
    d1, d2 = _fmt_stooq_date(start), _fmt_stooq_date(end)
    if d1 is not None:
        params["d1"] = d1
    if d2 is not None:
        params["d2"] = d2
    text = http_get_text(_STOOQ_CSV_URL, params=params, timeout=timeout)
    return _normalize_stooq_csv(text, symbol)


class StooqDataHandler(StreamingDataHandler):
    """Stream Stooq OHLCV CSV through the event engine.

    symbols: Stooq ticker(s) with suffix (e.g. ``\"aapl.us\"`` or
        ``[\"aapl.us\", \"msft.us\"]``).
    start/end: optional calendar bounds (``d1``/``d2``); naive = calendar date.
    interval: Stooq interval code (``d``, ``w``, ``m``, ``60``, ``5``, …).
    api_key: free Stooq key; defaults to ``QUANTESTER_STOOQ_API_KEY``.
    """

    def __init__(
        self,
        symbols,
        start=None,
        end=None,
        interval: str = "d",
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
                interval=interval,
                api_key=api_key,
                timeout=timeout,
            )
            for symbol in symbols
        }
        super().__init__(frames)
        self._interval = interval
