"""AKShareDataHandler: free OHLCV via AKShare (China A-shares + US daily).

Data is downloaded once at construction (historic batch replay; not a live
feed), normalized to the StreamingDataHandler contract, and streamed under the
same temporal firewall and availability-mask semantics as HistoricCSVDataHandler.

Requires the optional dependency:  pip install "quantester[akshare]"
(also included in ``quantester[data]``). No API key.

Opinionated v1 surface (not a full AKShare mirror):
- ``market=\"cn\"`` (default): ``ak.stock_zh_a_hist`` — 6-digit A-share codes,
  Chinese column names mapped to OHLCV; ``adjust`` in {\"\", \"qfq\", \"hfq\"}.
- ``market=\"us\"``: ``ak.stock_us_daily`` — Yahoo/Sina US symbols; English
  columns; ``adjust`` in {\"\", \"qfq\"}.

Daily session dates are UTC-localized. Volume for A-shares is typically in
lots (手) as returned by the portal — we pass it through unchanged. Retrieval
only — no quantitative formulas.
"""

from __future__ import annotations

import pandas as pd

from .streaming import StreamingDataHandler

_CN_MAP = {
    "日期": "datetime",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}
_US_MAP = {
    "date": "datetime",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def _import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError(
            "AKShareDataHandler requires akshare: "
            "pip install 'quantester[akshare]'"
        ) from exc
    return ak


def _fmt_yyyymmdd(value) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y%m%d")


def _localize_daily_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx, name="datetime").tz_localize("UTC")


def _normalize_cn(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(
            f"AKShare stock_zh_a_hist returned no data for {symbol!r}; "
            "check the 6-digit code and date range."
        )
    df = raw.rename(columns=_CN_MAP)
    missing = [c for c in ("datetime", "open", "high", "low", "close", "volume")
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"AKShare CN frame for {symbol!r} missing {missing}; "
            f"got {list(raw.columns)}."
        )
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = _localize_daily_index(pd.to_datetime(df["datetime"]))
    return out.sort_index()


def _normalize_us(raw: pd.DataFrame, symbol: str, start=None, end=None) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(
            f"AKShare stock_us_daily returned no data for {symbol!r}."
        )
    df = raw.rename(columns={c: _US_MAP.get(str(c).lower(), c) for c in raw.columns})
    # stock_us_daily may already use lowercase English names.
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for src, dst in _US_MAP.items():
        if src in df.columns:
            rename[src] = dst
        elif src in lower:
            rename[lower[src]] = dst
    df = df.rename(columns=rename)
    missing = [c for c in ("datetime", "open", "high", "low", "close", "volume")
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"AKShare US frame for {symbol!r} missing {missing}; "
            f"got {list(raw.columns)}."
        )
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = _localize_daily_index(pd.to_datetime(df["datetime"]))
    out = out.sort_index()
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        # Inclusive calendar end like most daily portals; cut at end-of-day.
        end_ts = pd.Timestamp(end, tz="UTC")
        out = out.loc[out.index <= end_ts]
    if out.empty:
        raise ValueError(
            f"AKShare US daily for {symbol!r} has no rows in the requested "
            "start/end window."
        )
    return out


def _download_ohlcv(
    symbol: str,
    *,
    market: str = "cn",
    start=None,
    end=None,
    adjust: str = "qfq",
    period: str = "daily",
) -> pd.DataFrame:
    """One symbol via AKShare; separated for testability (monkeypatch here)."""
    ak = _import_akshare()
    market = market.lower().strip()
    if market in {"cn", "a", "zh", "china"}:
        kwargs = {
            "symbol": symbol,
            "period": period,
            "adjust": adjust,
        }
        d0, d1 = _fmt_yyyymmdd(start), _fmt_yyyymmdd(end)
        if d0 is not None:
            kwargs["start_date"] = d0
        if d1 is not None:
            kwargs["end_date"] = d1
        raw = ak.stock_zh_a_hist(**kwargs)
        return _normalize_cn(raw, symbol)
    if market in {"us", "usa"}:
        raw = ak.stock_us_daily(symbol=symbol, adjust=adjust)
        return _normalize_us(raw, symbol, start=start, end=end)
    raise ValueError(
        f"AKShareDataHandler market must be 'cn' or 'us'; got {market!r}."
    )


class AKShareDataHandler(StreamingDataHandler):
    """Stream AKShare OHLCV through the event engine.

    symbols: ticker or list. For ``market='cn'`` use 6-digit codes
        (``\"000001\"``); for ``market='us'`` use Yahoo-style tickers
        (``\"AAPL\"``).
    market: ``\"cn\"`` (default) or ``\"us\"``.
    start/end: date bounds (YYYYMMDD for CN API; UTC filter for US).
    adjust: ``\"qfq\"`` forward-adjusted (default), ``\"hfq\"`` (CN only),
        or ``\"\"`` unadjusted.
    period: CN only — ``\"daily\"`` / ``\"weekly\"`` / ``\"monthly\"``.
    """

    def __init__(
        self,
        symbols,
        start=None,
        end=None,
        market: str = "cn",
        adjust: str = "qfq",
        period: str = "daily",
    ):
        if isinstance(symbols, str):
            symbols = [symbols]
        frames = {
            symbol: _download_ohlcv(
                symbol,
                market=market,
                start=start,
                end=end,
                adjust=adjust,
                period=period,
            )
            for symbol in symbols
        }
        super().__init__(frames)
        self._market = market
        self._period = period
