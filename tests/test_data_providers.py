"""Provider handlers (yfinance/ccxt): normalization, pagination, firewall and
engine parity — all with stubbed fetchers, no network access required."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import quantester.data.ccxt_handler as ch
import quantester.data.yfinance_handler as yh
from quantester.data import (
    CCXTDataHandler,
    HistoricCSVDataHandler,
    YFinanceDataHandler,
)
from quantester.engine import BacktestEngine
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import FixedUnitSizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv

DAY_MS = 86_400_000
T0 = pd.Timestamp("2024-01-01", tz="UTC")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

def _yf_raw(df: pd.DataFrame) -> pd.DataFrame:
    """What yfinance returns: capitalized cols, exchange-local tz-aware index
    (midnight America/New_York for US daily bars), extra Dividends columns."""
    raw = df.rename(columns={c: c.capitalize() for c in df.columns})
    idx = pd.DatetimeIndex(raw.index)
    # Synthetic fixtures are UTC; map calendar dates to exchange-local midnights.
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    raw.index = idx.tz_localize("America/New_York")
    raw = raw.rename_axis("Date")
    raw["Dividends"] = 0.0
    raw["Stock Splits"] = 0.0
    return raw


def _install_fake_yf(monkeypatch, frames: dict):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start=None, end=None, interval="1d", auto_adjust=True,
                    **kwargs):
            raw = _yf_raw(frames[self.symbol])

            def _as_ny(ts):
                ts = pd.Timestamp(ts)
                if ts.tzinfo is None:
                    return ts.tz_localize("America/New_York")
                return ts.tz_convert("America/New_York")

            if start is not None:
                start = _as_ny(start)
                raw = raw.loc[raw.index >= start]
            if end is not None:
                end = _as_ny(end)
                raw = raw.loc[raw.index < end]
            return raw

    monkeypatch.setattr(yh, "_import_yfinance",
                        lambda: SimpleNamespace(Ticker=FakeTicker))


def _ccxt_rows(df: pd.DataFrame) -> list:
    # Explicit datetime64[ms] cast: asi8/view follow the index's native unit.
    epoch_ms = df.index.to_numpy(dtype="datetime64[ms]").astype("int64")
    cols = df[["open", "high", "low", "close", "volume"]].to_numpy()
    return [[int(ms), *map(float, row)] for ms, row in zip(epoch_ms, cols)]


class FakeExchange:
    """Faithful stand-in for a public ccxt exchange: since/limit paging."""

    id = "fakeex"
    has = {"fetchOHLCV": True}

    def __init__(self, rows_by_symbol: dict, now_ms: int):
        self._rows = rows_by_symbol
        self._now_ms = now_ms
        self.calls = []

    def parse_timeframe(self, timeframe):
        return 86400

    def milliseconds(self):
        return self._now_ms

    def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=None):
        self.calls.append((symbol, since, limit))
        rows = [r for r in self._rows[symbol] if since is None or r[0] >= since]
        return rows[: limit or len(rows)]


def _make_frames():
    aaa = make_synthetic_ohlcv("AAA", n_bars=120, seed=11)
    bbb = make_synthetic_ohlcv("BBB", n_bars=120, seed=22, missing_every=7)
    return {"AAA": aaa, "BBB": bbb}


def _yf_handler(monkeypatch, frames, **kwargs):
    _install_fake_yf(monkeypatch, frames)
    return YFinanceDataHandler(list(frames), **kwargs)


def _ccxt_handler(monkeypatch, frames, now_ms=None, **kwargs):
    now_ms = now_ms if now_ms is not None else int(
        max(df.index[-1] for df in frames.values()).value // 1_000_000
    ) + 10 * DAY_MS
    exchange = FakeExchange({s: _ccxt_rows(df) for s, df in frames.items()}, now_ms)
    monkeypatch.setattr(ch, "_make_exchange", lambda *a, **k: exchange)
    return CCXTDataHandler(list(frames), timeframe="1d", limit=1000, **kwargs)


# --------------------------------------------------------------------------
# yfinance normalization
# --------------------------------------------------------------------------

def test_yfinance_normalizes_columns_tz_and_dtypes(monkeypatch):
    frames = _make_frames()
    handler = _yf_handler(monkeypatch, frames)
    df = handler._data["AAA"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert df.index.name == "datetime"
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    # Wall-time calendar dates are preserved as UTC-labeled midnights.
    from quantester.data.streaming import normalize_ohlcv_frame

    pd.testing.assert_frame_equal(df, normalize_ohlcv_frame(frames["AAA"]),
                                  check_freq=False)


def test_yfinance_start_end_window_passed_through(monkeypatch):
    frames = _make_frames()
    start, end = frames["AAA"].index[5], frames["AAA"].index[25]
    handler = _yf_handler(monkeypatch, frames, start=start, end=end)
    assert handler._data["AAA"].index[0] == start
    assert handler._data["AAA"].index[-1] < end


def test_yfinance_empty_response_raises(monkeypatch):
    _install_fake_yf(monkeypatch, {"AAA": make_synthetic_ohlcv("AAA").iloc[0:0]})
    with pytest.raises(ValueError, match="no data"):
        YFinanceDataHandler("AAA")


def test_yfinance_import_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"quantester\[yfinance\]"):
        yh._import_yfinance()


# --------------------------------------------------------------------------
# D9 (ticket 25): unadjusted default + corporate-action extraction
# --------------------------------------------------------------------------


def _install_recording_fake_yf(monkeypatch, frames, seen):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start=None, end=None, interval="1d",
                    auto_adjust=True, **kwargs):
            seen[self.symbol] = auto_adjust
            return _yf_raw(frames[self.symbol])

    monkeypatch.setattr(yh, "_import_yfinance",
                        lambda: SimpleNamespace(Ticker=FakeTicker))


def test_yfinance_default_is_unadjusted_with_ca_schedule(monkeypatch):
    """D9: the default research path loads RAW prices (auto_adjust=False)
    and carries the dividend/split schedule as corporate-action events."""
    frames = _make_frames()
    seen = {}
    _install_recording_fake_yf(monkeypatch, frames, seen)
    YFinanceDataHandler("AAA")
    assert seen["AAA"] is False


def _frames_with_actions():
    frames = _make_frames()
    return frames


def test_yfinance_dividends_become_corporate_action_events(monkeypatch):
    frames = _make_frames()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start=None, end=None, interval="1d",
                    auto_adjust=True, **kwargs):
            raw = _yf_raw(frames[self.symbol])
            # Two dividends and one split in the window.
            raw.iloc[10, raw.columns.get_loc("Dividends")] = 0.25
            raw.iloc[40, raw.columns.get_loc("Dividends")] = 0.30
            raw.iloc[60, raw.columns.get_loc("Stock Splits")] = 2.0
            return raw

    monkeypatch.setattr(yh, "_import_yfinance",
                        lambda: SimpleNamespace(Ticker=FakeTicker))
    handler = YFinanceDataHandler("AAA")
    bars = handler._data["AAA"]
    div_10 = handler.corporate_actions_at(bars.index[10])
    assert len(div_10) == 1 and div_10[0].kind == "dividend"
    assert div_10[0].dividend_per_share == pytest.approx(0.25)
    split_60 = handler.corporate_actions_at(bars.index[60])
    assert len(split_60) == 1 and split_60[0].kind == "split"
    assert split_60[0].split_ratio == pytest.approx(2.0)
    # Bars carry no CA columns and no price rewrite.
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


def test_yfinance_auto_adjust_true_suppresses_ca_events(monkeypatch):
    """auto_adjust=True (total-return ranking mode) must not double-book:
    dividends are already inside the adjusted prices, so no CA events."""
    frames = _make_frames()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start=None, end=None, interval="1d",
                    auto_adjust=True, **kwargs):
            raw = _yf_raw(frames[self.symbol])
            raw.iloc[10, raw.columns.get_loc("Dividends")] = 0.25
            return raw

    monkeypatch.setattr(yh, "_import_yfinance",
                        lambda: SimpleNamespace(Ticker=FakeTicker))
    handler = YFinanceDataHandler("AAA", auto_adjust=True)
    bars = handler._data["AAA"]
    assert handler.corporate_actions_at(bars.index[10]) == []


# --------------------------------------------------------------------------
# ccxt pagination + normalization
# --------------------------------------------------------------------------

def test_ccxt_pagination_walks_forward():
    df = make_synthetic_ohlcv("AAA", n_bars=7, seed=3)
    now_ms = int(df.index[-1].value // 1_000_000) + 10 * DAY_MS
    exchange = FakeExchange({"AAA": _ccxt_rows(df)}, now_ms)

    out = ch._fetch_symbol_ohlcv(exchange, "AAA", limit=3)

    # 7 rows at limit=3 -> pages of 3, 3, 1; `since` advances by last+tf.
    # Page size alone never terminates the walk (exchanges cap below limit),
    # so a final probe past the last bar returns empty and stops the loop.
    assert [c[1] for c in exchange.calls] == [
        None,
        int(df.index[2].value // 1_000_000) + DAY_MS,
        int(df.index[5].value // 1_000_000) + DAY_MS,
        int(df.index[6].value // 1_000_000) + DAY_MS,
    ]
    from quantester.data.streaming import normalize_ohlcv_frame

    pd.testing.assert_frame_equal(normalize_ohlcv_frame(out),
                                  normalize_ohlcv_frame(df), check_freq=False)


def test_ccxt_stale_page_breaks_without_looping():
    df = make_synthetic_ohlcv("AAA", n_bars=5, seed=4)
    now_ms = int(df.index[-1].value // 1_000_000) + 10 * DAY_MS

    class StickyExchange(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=None):
            self.calls.append((symbol, since, limit))
            return self._rows[symbol][:2]  # ignores since, repeats page

    exchange = StickyExchange({"AAA": _ccxt_rows(df)}, now_ms)
    out = ch._fetch_symbol_ohlcv(exchange, "AAA", limit=2)  # full page -> next call
    assert len(exchange.calls) == 2  # second page yields nothing new -> stop
    assert len(out) == 2


def test_ccxt_incomplete_candle_dropped_by_default():
    df = make_synthetic_ohlcv("AAA", n_bars=6, seed=5)
    last_open_ms = int(df.index[-1].value // 1_000_000)
    exchange = FakeExchange({"AAA": _ccxt_rows(df)}, now_ms=last_open_ms + 1000)

    dropped = ch._fetch_symbol_ohlcv(exchange, "AAA", drop_incomplete=True)
    kept = ch._fetch_symbol_ohlcv(exchange, "AAA", drop_incomplete=False)
    assert len(dropped) == len(df) - 1
    assert len(kept) == len(df)


def test_ccxt_until_ms_filters_tail():
    df = make_synthetic_ohlcv("AAA", n_bars=10, seed=6)
    now_ms = int(df.index[-1].value // 1_000_000) + 10 * DAY_MS
    exchange = FakeExchange({"AAA": _ccxt_rows(df)}, now_ms)
    until_ms = int(df.index[4].value // 1_000_000)
    out = ch._fetch_symbol_ohlcv(exchange, "AAA", until_ms=until_ms)
    assert out.index[-1] == df.index[4]
    assert out.index.tz is not None


def test_ccxt_empty_and_only_incomplete_raise():
    now_ms = int(T0.value // 1_000_000)
    with pytest.raises(ValueError, match="no OHLCV data"):
        ch._fetch_symbol_ohlcv(FakeExchange({"AAA": []}, now_ms), "AAA")

    df = make_synthetic_ohlcv("AAA", n_bars=1, seed=7)
    forming = FakeExchange({"AAA": _ccxt_rows(df)},
                           now_ms=int(df.index[0].value // 1_000_000) + 1)
    with pytest.raises(ValueError, match="drop_incomplete"):
        ch._fetch_symbol_ohlcv(forming, "AAA", drop_incomplete=True)


def test_ccxt_unknown_exchange_and_missing_capability(monkeypatch):
    monkeypatch.setattr(ch, "_import_ccxt", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="Unknown ccxt exchange"):
        ch._make_exchange("not_an_exchange")

    class NoOhlcv:
        has = {"fetchOHLCV": False}

        def __init__(self, config):
            pass

    monkeypatch.setattr(ch, "_import_ccxt",
                        lambda: SimpleNamespace(broken=NoOhlcv))
    with pytest.raises(ValueError, match="fetch_ohlcv"):
        ch._make_exchange("broken")


def test_ccxt_import_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "ccxt", None)
    with pytest.raises(ImportError, match=r"quantester\[ccxt\]"):
        ch._import_ccxt()


def test_to_ms_interprets_naive_as_utc():
    naive = pd.Timestamp("2024-01-01")
    assert ch._to_ms(naive) == int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    aware = pd.Timestamp("2024-01-01", tz="America/New_York")  # 05:00 UTC
    assert ch._to_ms(aware) == int(T0.value // 1_000_000) + 5 * 3_600_000
    assert ch._to_ms(None) is None


# --------------------------------------------------------------------------
# Firewall / mask / engine parity across providers
# --------------------------------------------------------------------------

@pytest.fixture(params=["csv", "yfinance", "ccxt"])
def any_handler(request, monkeypatch):
    frames = _make_frames()
    if request.param == "csv":
        return HistoricCSVDataHandler(frames)
    if request.param == "yfinance":
        return _yf_handler(monkeypatch, frames)
    return _ccxt_handler(monkeypatch, frames)


def test_providers_share_firewall_and_masks(any_handler):
    handler = any_handler
    assert handler.symbols == ["AAA", "BBB"]

    handler.prime_data()
    ts, bars = handler.advance()
    assert ts == handler.current_timestamp

    # Availability mask: BBB's 8th bar (index 50::7 within 40 -> none here,
    # so find a real gap) is untradeable, never erased.
    gap_ts = None
    while handler.continue_backtest:
        ts, bars = handler.advance()
        if bars["BBB"] is None:
            gap_ts = ts
            break
    assert gap_ts is not None and bars["AAA"] is not None
    assert handler.get_current_open("BBB") is None

    # Open phase: current bar excluded; close phase: included.
    handler.set_phase("open", ts)
    assert handler.get_latest_bars("AAA", 10**6).index.max() < ts
    handler.set_phase("close", ts)
    assert handler.get_latest_bars("AAA", 10**6).index.max() == ts

    # bar_at + timestamp_at_offset work identically on every provider.
    assert handler.bar_at("AAA", ts) is not None
    first = handler._master_index[0]
    assert handler.timestamp_at_offset(first, 1) == handler._master_index[1]
    assert handler.timestamp_at_offset(handler._master_index[-1], 1) is None


def test_engine_positions_identical_across_providers(monkeypatch, zero_costs):
    frames = _make_frames()

    def run(handler):
        strategy = MovingAverageCrossStrategy(handler, "AAA", fast=3, slow=8)
        portfolio = PortfolioManager(handler, 100_000.0,
                                     sizer=FixedUnitSizer(100))
        engine = BacktestEngine(handler, strategy, portfolio,
                                SimulatedExecutionHandler(zero_costs))
        engine.run_backtest()
        return portfolio

    reference = run(HistoricCSVDataHandler(frames))
    for make in (_yf_handler, _ccxt_handler):
        portfolio = run(make(monkeypatch, frames))
        pd.testing.assert_series_equal(portfolio.positions_history["AAA"],
                                       reference.positions_history["AAA"],
                                       check_freq=False)
        assert len(portfolio.fills) == len(reference.fills) > 0
        for got, want in zip(portfolio.fills, reference.fills):
            assert got.timestamp == want.timestamp
            assert got.fill_price == pytest.approx(want.fill_price, rel=1e-12)


def test_real_exchange_constructs_offline():
    ccxt = pytest.importorskip("ccxt")
    exchange = ch._make_exchange("kraken")
    assert exchange.enableRateLimit is True
    assert exchange.has["fetchOHLCV"] is True


def test_funding_history_paginates_by_since():
    class Pager:
        has = {"fetchFundingRateHistory": True}

        def fetch_funding_rate_history(self, symbol, since=None, limit=None):
            if since is None or since <= 1_000:
                return [
                    {"timestamp": 1_000 + 8_000 * i, "fundingRate": 0.0001 * (i + 1)}
                    for i in range(3)
                ]
            if since == 1_000 + 16_000 + 1:
                return [
                    {"timestamp": 1_000 + 24_000, "fundingRate": 0.0004},
                    {"timestamp": 1_000 + 32_000, "fundingRate": 0.0005},
                ]
            return []

    rows = ch._page_funding_history(Pager(), "BTC/USDT:USDT", since_ms=0, until_ms=None)
    assert len(rows) == 5
    assert rows[-1]["fundingRate"] == pytest.approx(0.0005)


def test_dvol_index_rows_use_close():
    rows = [[1_700_000_000_000, 50.0, 60.0, 40.0, 55.5]]
    s = ch._dvol_from_index_rows(rows)
    assert float(s.iloc[0]) == pytest.approx(55.5)


def test_apply_binance_www_host_rewrites_fapi():
    class Dummy:
        urls = {"api": {
            "fapiPublic": "https://fapi.binance.com/fapi/v1",
            "fapiData": "https://fapi.binance.com/futures/data",
            "public": "https://api.binance.com/api/v3",
        }}

    ch._apply_binance_www_host(Dummy)
    assert Dummy.urls["api"]["fapiPublic"].startswith("https://www.binance.com")
    assert Dummy.urls["api"]["fapiData"] == "https://www.binance.com/futures/data"
    assert Dummy.urls["api"]["public"].startswith("https://api.binance.com")
