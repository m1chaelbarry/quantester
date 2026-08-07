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
- Timestamps are milliseconds since epoch (UTC); we store tz-naive UTC so the
  master calendar stays comparable across providers.
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


def _make_exchange(exchange_id: str, config: dict | None = None):
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
        pd.to_datetime(df.pop("timestamp_ms"), unit="ms"), name="datetime"
    )

    if until_ms is not None:
        cutoff = pd.to_datetime(until_ms, unit="ms")
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
    """

    def __init__(self, symbols, exchange: str = "binance", timeframe: str = "1d",
                 start=None, end=None, limit: int = 1000,
                 drop_incomplete: bool = True, exchange_config: dict | None = None):
        if isinstance(symbols, str):
            symbols = [symbols]
        ex = _make_exchange(exchange, exchange_config)
        since_ms, until_ms = _to_ms(start), _to_ms(end)
        frames = {
            symbol: _fetch_symbol_ohlcv(
                ex, symbol, timeframe=timeframe, since_ms=since_ms,
                until_ms=until_ms, limit=limit, drop_incomplete=drop_incomplete,
            )
            for symbol in symbols
        }
        super().__init__(frames)
        self._exchange_id = exchange
        self._timeframe = timeframe
