"""YFinanceDataHandler: free daily/intraday OHLCV from Yahoo Finance via yfinance.

Data is downloaded once at construction (the engine replays a historic batch;
this is not a live websocket feed), normalized to the StreamingDataHandler
contract, and streamed under the same temporal firewall and availability-mask
semantics as HistoricCSVDataHandler.

Requires the optional dependency:  pip install "quantester[yfinance]"

Normalization notes (yfinance >= 0.2 / 1.x API, per official docs):
- Ticker.history() returns capitalized columns (Open/High/Low/Close/Volume,
  plus Dividends/Stock Splits) indexed by a tz-aware 'Date'/'Datetime' index
  in the exchange's local timezone (e.g. America/New_York for US listings);
  we convert to timezone-aware UTC so the core engine has a single timestamp
  standard (DST transitions become explicit UTC instants).
- auto_adjust is pinned explicitly (ruling D9, ticket 25): the DEFAULT is
  ``auto_adjust=False`` — a raw (unadjusted) price ledger whose nonzero
  Dividends / Stock Splits rows become CorporateActionEvents booked as cash
  / quantity on the portfolio (Peterson ch. 11 cash booking). Pass
  ``auto_adjust=True`` for a total-return RANKING mode: adjusted OHLC with
  no CA events (dividends live inside the adjusted prices — never
  double-booked).

Corporate-action / adjustment semantics: with ``auto_adjust=False``
(default), OHLC is raw and corporate actions ride the event queue.
Survivorship and historical universe membership are NOT handled by this
provider — Yahoo's current listing set is subject to survivorship bias and
must be documented by the researcher.

This module implements data retrieval only -- no quantitative formulas, so
there is nothing to verify against the quant-literature notebook; the
firewall/availability-mask semantics it inherits are notebook-verified.
"""

from __future__ import annotations

import pandas as pd

from .streaming import StreamingDataHandler

_OHLCV_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
              "Volume": "volume"}


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "YFinanceDataHandler requires yfinance: "
            "pip install 'quantester[yfinance]'"
        ) from exc
    return yf


def _normalize_history(raw: pd.DataFrame, symbol: str, interval: str = "1d") -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned no data for {symbol!r}; check the ticker, "
            "date range, and interval."
        )
    df = raw.rename(columns=_OHLCV_MAP)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        # Daily/weekly/monthly: keep exchange-local calendar dates as UTC labels
        # so cross-provider daily joins stay on the same session date.
        # Intraday: convert to absolute UTC (wall-clock relabel would mis-time fills).
        if interval in {"1d", "1wk", "1mo", "5d", "3mo"}:
            idx = idx.tz_localize(None).tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")
    df.index = pd.DatetimeIndex(idx, name="datetime")
    return df


def _download_ohlcv(symbol: str, start=None, end=None, interval: str = "1d",
                    auto_adjust: bool = False, **history_kwargs) -> pd.DataFrame:
    """One symbol via yfinance; separated for testability (monkeypatch here)."""
    yf = _import_yfinance()
    raw = yf.Ticker(symbol).history(
        start=start, end=end, interval=interval,
        auto_adjust=auto_adjust, **history_kwargs,
    )
    return _normalize_history(raw, symbol, interval=interval)


def _extract_corporate_actions(df: pd.DataFrame) -> pd.DataFrame | None:
    """Nonzero ``Dividends`` / ``Stock Splits`` rows of a normalized Yahoo
    history frame as an ex-date CA schedule (columns dividend / split)."""
    parts = {}
    if "Dividends" in df.columns:
        div = df["Dividends"].astype(float)
        parts["dividend"] = div[div > 0]
    if "Stock Splits" in df.columns:
        split = df["Stock Splits"].astype(float)
        parts["split"] = split[split > 0]
    if not parts:
        return None
    return pd.concat(parts, axis=1).fillna(0.0)


class YFinanceDataHandler(StreamingDataHandler):
    """Stream Yahoo Finance OHLCV through the event engine.

    symbols: ticker or list of tickers (e.g. "AAPL" or ["AAPL", "MSFT"]).
    start/end: anything pandas.Timestamp parses; end is exclusive (yfinance).
    interval: yfinance interval string ("1d", "1h", "1wk", ...).
    auto_adjust: default False (D9) — raw prices + corporate-action events
        (dividend cash / split quantity) routed through the queue. True is a
        documented total-return ranking mode and suppresses CA events so
        dividends are never double-booked.
    """

    def __init__(self, symbols, start=None, end=None, interval: str = "1d",
                 auto_adjust: bool = False, **history_kwargs):
        if isinstance(symbols, str):
            symbols = [symbols]
        frames = {}
        ca_frames = {}
        for symbol in symbols:
            raw = _download_ohlcv(symbol, start=start, end=end,
                                  interval=interval, auto_adjust=auto_adjust,
                                  **history_kwargs)
            if not auto_adjust:
                ca = _extract_corporate_actions(raw)
                if ca is not None:
                    ca_frames[symbol] = ca
            frames[symbol] = raw
        super().__init__(frames, corporate_actions=ca_frames or None)
        self._interval = interval
