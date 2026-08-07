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
- auto_adjust is passed through explicitly (default True) because yfinance
  changed the library default across releases; adjusted OHLC removes
  artificial price gaps from splits/dividends, which is what a total-return
  style backtest should consume.

Corporate-action / adjustment semantics: with ``auto_adjust=True`` (default),
OHLC is split/dividend adjusted; raw (unadjusted) prices require
``auto_adjust=False``. Survivorship and historical universe membership are
NOT handled by this provider — Yahoo's current listing set is subject to
survivorship bias and must be documented by the researcher.

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


def _normalize_history(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned no data for {symbol!r}; check the ticker, "
            "date range, and interval."
        )
    df = raw.rename(columns=_OHLCV_MAP)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        # Keep exchange-local wall times (calendar dates for daily bars) and
        # stamp them as timezone-aware UTC labels. Converting NY midnights to
        # absolute UTC would shift daily bars to 05:00 and break cross-provider
        # calendar alignment; intraday users who need absolute UTC should
        # prefer ccxt or pre-convert before constructing the handler.
        idx = idx.tz_localize(None)
    idx = idx.tz_localize("UTC")
    df.index = pd.DatetimeIndex(idx, name="datetime")
    return df


def _download_ohlcv(symbol: str, start=None, end=None, interval: str = "1d",
                    auto_adjust: bool = True, **history_kwargs) -> pd.DataFrame:
    """One symbol via yfinance; separated for testability (monkeypatch here)."""
    yf = _import_yfinance()
    raw = yf.Ticker(symbol).history(
        start=start, end=end, interval=interval,
        auto_adjust=auto_adjust, **history_kwargs,
    )
    return _normalize_history(raw, symbol)


class YFinanceDataHandler(StreamingDataHandler):
    """Stream Yahoo Finance OHLCV through the event engine.

    symbols: ticker or list of tickers (e.g. "AAPL" or ["AAPL", "MSFT"]).
    start/end: anything pandas.Timestamp parses; end is exclusive (yfinance).
    interval: yfinance interval string ("1d", "1h", "1wk", ...).
    auto_adjust: adjust OHLC for splits/dividends (default True, pinned
        explicitly so behavior does not depend on the yfinance version).
    """

    def __init__(self, symbols, start=None, end=None, interval: str = "1d",
                 auto_adjust: bool = True, **history_kwargs):
        if isinstance(symbols, str):
            symbols = [symbols]
        frames = {
            symbol: _download_ohlcv(symbol, start=start, end=end,
                                    interval=interval, auto_adjust=auto_adjust,
                                    **history_kwargs)
            for symbol in symbols
        }
        super().__init__(frames)
        self._interval = interval
