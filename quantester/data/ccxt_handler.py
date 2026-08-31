"""CCXTDataHandler: free crypto OHLCV from 100+ exchanges via ccxt.

Public fetch_ohlcv endpoints need no API key. Data is downloaded once at
construction (the engine replays a historic batch; this is not a live
websocket feed), normalized to the StreamingDataHandler contract, and streamed
under the same temporal firewall and availability-mask semantics as
HistoricCSVDataHandler.

Requires the optional dependency:  pip install "quantester[ccxt]"

Implementation notes (ccxt 4.x API, per official docs):
- fetch_ohlcv(symbol, timeframe, since, limit) returns a list of
  [timestamp_ms, open, high, low, close, volume] rows, ascending; exchanges
  cap page size (sometimes below the requested limit), so history is walked
  forward by advancing since = last_timestamp + timeframe_ms until a page
  returns no new rows or the walk reaches the requested end / the present.
- Timestamps are milliseconds since epoch (UTC); we store timezone-aware UTC
  so the master calendar stays comparable across providers.
- The still-forming candle for the current period is dropped by default
  (drop_incomplete=True): its high/low/close contain prints that would not
  have existed at the bar's open, i.e. implicit look-ahead inside the last
  bar. Pass drop_incomplete=False to keep it.
- Exchange instances are created with enableRateLimit=True so pagination
  self-throttles instead of tripping exchange bans.

This module implements data retrieval only -- no quantitative formulas, so
there is nothing to verify against the quant-literature notebook; the
firewall/availability-mask semantics it inherits are notebook-verified.
"""

from __future__ import annotations

import pandas as pd

from .streaming import StreamingDataHandler

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _import_ccxt():
    try:
        import ccxt
    except ImportError as exc:
        raise ImportError(
            "CCXTDataHandler requires ccxt: pip install 'quantester[ccxt]'"
        ) from exc
    return ccxt


def _to_ms(dt) -> int | None:
    """Parse a datetime-like to epoch milliseconds (UTC), or pass through None."""
    if dt is None:
        return None
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def _apply_binance_www_host(exchange) -> None:
    """Reroute fapi.* to www.binance.com when the regional API returns 451."""
    api = exchange.urls.get("api") or {}
    for key, url in list(api.items()):
        if isinstance(url, str) and "fapi.binance.com" in url:
            api[key] = url.replace("https://fapi.binance.com", "https://www.binance.com")


def _make_exchange(exchange_id: str, config: dict | None = None,
                   geo_safe: bool = False):
    """Construct a rate-limited public ccxt exchange (offline-safe; no
    load_markets call here)."""
    ccxt = _import_ccxt()
    if not hasattr(ccxt, exchange_id):
        raise ValueError(
            f"Unknown ccxt exchange id {exchange_id!r}; "
            "see ccxt.exchanges for the supported list."
        )
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, **(config or {})})
    if not exchange.has.get("fetchOHLCV"):
        raise ValueError(f"Exchange {exchange_id!r} does not support fetch_ohlcv.")
    if geo_safe and exchange_id in {"binance", "binanceusdm"}:
        _apply_binance_www_host(exchange)
    return exchange


def _fetch_symbol_ohlcv(exchange, symbol: str, timeframe: str = "1d",
                        since_ms: int | None = None, until_ms: int | None = None,
                        limit: int = 1000, drop_incomplete: bool = True,
                        ) -> pd.DataFrame:
    """Walk fetch_ohlcv pages forward and return a normalized OHLCV frame.

    Takes an already-constructed exchange so tests can drive it with a fake;
    all pagination/normalization logic lives here.
    """
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()
    rows: list[list] = []
    seen: set[int] = set()
    since = since_ms
    last_ts = -1

    while True:
        page = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since,
                                    limit=limit)
        if not page:
            break
        new = [r for r in page if r[0] > last_ts and r[0] not in seen]
        if not new:
            break  # exchange ignored `since` and returned stale rows
        for r in new:
            seen.add(r[0])
        rows.extend(new)
        last_ts = new[-1][0]
        if until_ms is not None and last_ts >= until_ms:
            break
        if last_ts + tf_ms >= now_ms:
            break  # caught up: the next bar would be the still-forming one
        # NB: a short page does NOT mean history is exhausted -- exchanges cap
        # page size below the requested limit (e.g. Coinbase returns 300).
        since = last_ts + tf_ms

    if not rows:
        raise ValueError(
            f"ccxt exchange {getattr(exchange, 'id', exchange)!r} returned no "
            f"OHLCV data for {symbol!r} ({timeframe}); check the symbol, "
            "date range, and timeframe."
        )

    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["timestamp_ms", *_COLUMNS])
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.pop("timestamp_ms"), unit="ms", utc=True),
        name="datetime",
    )

    if until_ms is not None:
        cutoff = pd.to_datetime(until_ms, unit="ms", utc=True)
        df = df.loc[df.index <= cutoff]

    if drop_incomplete and len(df):
        # .value is tz-agnostic (epoch ns), unlike Timestamp.timestamp() which
        # interprets naive stamps in the system local timezone.
        bar_open_ms = df.index[-1].value // 1_000_000
        if bar_open_ms + tf_ms > now_ms:
            df = df.iloc[:-1]

    if df.empty:
        raise ValueError(
            f"No completed {timeframe} bars for {symbol!r} in the requested "
            "range (the only bar may still be forming; pass "
            "drop_incomplete=False to keep it)."
        )
    return df


def _history_to_series(rows, value_key: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True)
    vals = [r.get(value_key) for r in rows]
    return pd.Series(vals, index=idx, dtype=float)


_MAX_HISTORY_PAGES = 80


def _page_funding_history(exchange, symbol: str, since_ms, until_ms,
                          limit: int = 1000) -> list:
    """Walk ``fetch_funding_rate_history``; a single page is typically 1000 prints."""
    rows: list = []
    since = since_ms
    last_ts = -1
    for _ in range(_MAX_HISTORY_PAGES):
        kwargs = {"since": since, "limit": limit}
        page = exchange.fetch_funding_rate_history(symbol, **kwargs)
        if not page:
            break
        new = [r for r in page if r.get("timestamp") is not None and r["timestamp"] > last_ts]
        if not new:
            break
        rows.extend(new)
        last_ts = int(new[-1]["timestamp"])
        if until_ms is not None and last_ts >= until_ms:
            break
        since = last_ts + 1
    return rows


def _safe_fetch_funding(exchange, symbol: str, since_ms, until_ms) -> pd.Series | None:
    if not getattr(exchange, "has", {}).get("fetchFundingRateHistory"):
        return None
    try:
        rows = _page_funding_history(exchange, symbol, since_ms, until_ms)
    except Exception:
        return None
    if until_ms is not None:
        rows = [r for r in rows if r.get("timestamp") is not None and r["timestamp"] <= until_ms]
    series = _history_to_series(rows, "fundingRate")
    return series if len(series) else None


def _safe_fetch_oi(exchange, symbol: str, since_ms, until_ms) -> pd.Series | None:
    has = getattr(exchange, "has", {})
    if not has.get("fetchOpenInterestHistory"):
        return None
    # Binance USDM rejects historical startTime on this endpoint (~30d window).
    try:
        rows = exchange.fetch_open_interest_history(symbol, timeframe="1d", limit=500)
    except Exception:
        try:
            rows = exchange.fetch_open_interest_history(symbol, since=since_ms)
        except Exception:
            return None
    series = _history_to_series(rows or [], "openInterestAmount")
    if series.isna().all():
        series = _history_to_series(rows or [], "openInterestValue")
    if until_ms is not None and len(series):
        cutoff = pd.to_datetime(until_ms, unit="ms", utc=True)
        series = series.loc[series.index <= cutoff]
    return series if len(series) else None


def _dvol_from_index_rows(rows) -> pd.Series:
    """Deribit volatility index rows: ``[ts, open, high, low, close]``."""
    idx, vals = [], []
    for r in rows:
        if isinstance(r, dict):
            ts = r.get("timestamp")
            vol = r.get("volatility", r.get("value"))
        elif isinstance(r, (list, tuple)) and len(r) >= 5:
            ts, vol = r[0], r[4]
        else:
            continue
        if ts is None or vol is None:
            continue
        idx.append(ts)
        vals.append(float(vol))
    if not idx:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=pd.to_datetime(idx, unit="ms", utc=True), dtype=float)


def _safe_fetch_dvol(exchange_id: str, since_ms, until_ms) -> pd.Series | None:
    try:
        ex = _make_exchange(exchange_id)
    except Exception:
        return None
    start = int(since_ms or 0)
    end = int(until_ms or getattr(ex, "milliseconds", lambda: 0)() or 0)
    chunks: list = []
    if hasattr(ex, "public_get_get_volatility_index_data") and end:
        guard_end = end
        for _ in range(_MAX_HISTORY_PAGES):
            try:
                raw = ex.public_get_get_volatility_index_data({
                    "currency": "BTC",
                    "resolution": "1D",
                    "start_timestamp": start,
                    "end_timestamp": guard_end,
                })
            except Exception:
                chunks = []
                break
            result = raw.get("result") if isinstance(raw, dict) else None
            data = (result or {}).get("data") if isinstance(result, dict) else None
            if not data:
                break
            chunks.extend(data)
            first_ts = int(data[0][0])
            if first_ts <= start + 86_400_000:
                break
            cont = result.get("continuation")
            nxt = int(cont) if cont is not None else first_ts - 1
            if nxt >= guard_end:
                break
            guard_end = nxt
    if not chunks and getattr(ex, "has", {}).get("fetchVolatilityHistory"):
        try:
            rows = ex.fetch_volatility_history("BTC")
        except Exception:
            rows = None
        if rows:
            series = _dvol_from_index_rows(rows)
            if until_ms is not None and len(series):
                cutoff = pd.to_datetime(until_ms, unit="ms", utc=True)
                series = series.loc[series.index <= cutoff]
            return series if len(series) else None
    if not chunks:
        return None
    series = _dvol_from_index_rows(chunks)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if until_ms is not None and len(series):
        cutoff = pd.to_datetime(until_ms, unit="ms", utc=True)
        series = series.loc[series.index <= cutoff]
    if since_ms is not None and len(series):
        floor = pd.to_datetime(since_ms, unit="ms", utc=True)
        series = series.loc[series.index >= floor]
    return series if len(series) else None


class CCXTDataHandler(StreamingDataHandler):
    """Stream crypto-exchange OHLCV through the event engine.

    symbols: market symbol or list (e.g. "BTC/USDT" or ["BTC/USDT", "ETH/USDT"]).
    exchange: ccxt exchange id ("binance", "kraken", "coinbase", ...).
    timeframe: ccxt timeframe string ("1m", "1h", "1d", ...).
    start/end: datetime-like bounds; naive values are interpreted as UTC.
    limit: page size per fetch_ohlcv call (exchange-capped, typically <= 1000).
    drop_incomplete: drop the still-forming final candle (default True).
    exchange_config: extra keys merged into the ccxt exchange constructor
        config (e.g. {"apiKey": ...} -- not needed for public OHLCV).
    include_extras: when True, also fetch funding-rate history and open
        interest (same exchange) plus Deribit DVOL when ``dvol_exchange``
        is set (default ``deribit``). Fail-open: missing capabilities leave
        the extra column absent. Funding history is paginated (a single
        Binance page is ~1000 8h prints).
    geo_safe: when True, Binance USDM fapi hosts are rewritten to
        ``www.binance.com`` (regional 451 on ``fapi.binance.com``).
    """

    def __init__(self, symbols, exchange: str = "binance", timeframe: str = "1d",
                 start=None, end=None, limit: int = 1000,
                 drop_incomplete: bool = True, exchange_config: dict | None = None,
                 include_extras: bool = False, dvol_exchange: str = "deribit",
                 geo_safe: bool = False):
        if isinstance(symbols, str):
            symbols = [symbols]
        ex = _make_exchange(exchange, exchange_config, geo_safe=geo_safe)
        since_ms, until_ms = _to_ms(start), _to_ms(end)
        frames = {
            symbol: _fetch_symbol_ohlcv(
                ex, symbol, timeframe=timeframe, since_ms=since_ms,
                until_ms=until_ms, limit=limit, drop_incomplete=drop_incomplete,
            )
            for symbol in symbols
        }
        if include_extras:
            from .crypto_extras import attach_extras

            dvol = _safe_fetch_dvol(dvol_exchange, since_ms, until_ms)
            for symbol, frame in list(frames.items()):
                funding = _safe_fetch_funding(ex, symbol, since_ms, until_ms)
                oi = _safe_fetch_oi(ex, symbol, since_ms, until_ms)
                frames[symbol] = attach_extras(
                    frame, funding=funding, open_interest=oi, dvol=dvol,
                )
        super().__init__(frames)
        self._exchange_id = exchange
        self._timeframe = timeframe
