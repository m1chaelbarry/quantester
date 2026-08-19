"""Stooq / FMP / AKShare handlers + macro overlays — offline stubbed tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import quantester.data.akshare_handler as akh
import quantester.data.fmp_handler as fmph
import quantester.data.stooq_handler as stooqh
import quantester.macro.gus as gus_mod
import quantester.macro.nbp as nbp_mod
import quantester.macro.worldbank as wb_mod
from quantester.data import (
    AKShareDataHandler,
    FMPDataHandler,
    StooqDataHandler,
)
from quantester.data._http import resolve_api_key
from quantester.data.streaming import normalize_ohlcv_frame
from quantester.macro import as_daily_reindex, load_gus_variable, load_nbp_fx, load_world_bank
from quantester.utils.synthetic import make_synthetic_ohlcv

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def _ohlcv_csv(df: pd.DataFrame, *, newest_first: bool = True) -> str:
    out = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }).copy()
    out.insert(0, "Date", out.index.tz_convert("UTC").strftime("%Y-%m-%d"))
    if newest_first:
        out = out.iloc[::-1]
    return out.to_csv(index=False)


def _fmp_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for ts, row in df.iloc[::-1].iterrows():
        rows.append({
            "date": ts.tz_convert("UTC").strftime("%Y-%m-%d"),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    return rows


# --------------------------------------------------------------------------
# API key helper
# --------------------------------------------------------------------------

def test_resolve_api_key_prefers_arg_then_env(monkeypatch):
    monkeypatch.delenv("QUANTESTER_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="QUANTESTER_TEST_KEY"):
        resolve_api_key(None, env_var="QUANTESTER_TEST_KEY", provider="Test")
    monkeypatch.setenv("QUANTESTER_TEST_KEY", " from-env ")
    assert resolve_api_key(None, env_var="QUANTESTER_TEST_KEY",
                           provider="Test") == "from-env"
    assert resolve_api_key(" explicit ", env_var="QUANTESTER_TEST_KEY",
                           provider="Test") == "explicit"
    assert resolve_api_key(None, env_var="QUANTESTER_TEST_KEY",
                           required=False, provider="Test") == "from-env"
    monkeypatch.delenv("QUANTESTER_TEST_KEY", raising=False)
    assert resolve_api_key(None, env_var="QUANTESTER_TEST_KEY",
                           required=False, provider="Test") is None


# --------------------------------------------------------------------------
# Stooq
# --------------------------------------------------------------------------

def test_stooq_normalizes_and_sorts(monkeypatch):
    frame = make_synthetic_ohlcv("AAA", n_bars=8, seed=1)
    monkeypatch.setattr(
        stooqh, "http_get_text",
        lambda *a, **k: _ohlcv_csv(frame),
    )
    handler = StooqDataHandler("aapl.us", api_key="k")
    got = handler.source_ohlcv("aapl.us")
    pd.testing.assert_frame_equal(
        got, normalize_ohlcv_frame(frame), check_freq=False,
    )


def test_stooq_quota_and_empty_errors(monkeypatch):
    monkeypatch.setattr(
        stooqh, "http_get_text",
        lambda *a, **k: "Exceeded the daily hits limit",
    )
    with pytest.raises(ValueError, match="hit limit"):
        StooqDataHandler("aapl.us", api_key="k")

    monkeypatch.setattr(stooqh, "http_get_text", lambda *a, **k: "")
    with pytest.raises(ValueError, match="empty"):
        StooqDataHandler("aapl.us", api_key="k")


def test_stooq_requires_api_key(monkeypatch):
    monkeypatch.delenv("QUANTESTER_STOOQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="QUANTESTER_STOOQ_API_KEY"):
        StooqDataHandler("aapl.us")


# --------------------------------------------------------------------------
# FMP
# --------------------------------------------------------------------------

def test_fmp_normalizes_newest_first(monkeypatch):
    frame = make_synthetic_ohlcv("AAA", n_bars=6, seed=2)
    monkeypatch.setattr(
        fmph, "http_get_json",
        lambda *a, **k: _fmp_rows(frame),
    )
    handler = FMPDataHandler("AAPL", api_key="k")
    pd.testing.assert_frame_equal(
        handler.source_ohlcv("AAPL"),
        normalize_ohlcv_frame(frame),
        check_freq=False,
    )


def test_fmp_error_payload(monkeypatch):
    monkeypatch.setattr(
        fmph, "http_get_json",
        lambda *a, **k: {"Error Message": "Limit Reach"},
    )
    with pytest.raises(ValueError, match="Limit Reach"):
        FMPDataHandler("AAPL", api_key="k")

    monkeypatch.setattr(fmph, "http_get_json", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no EOD rows"):
        FMPDataHandler("AAPL", api_key="k")


# --------------------------------------------------------------------------
# AKShare
# --------------------------------------------------------------------------

def test_akshare_cn_and_us_normalization(monkeypatch):
    frame = make_synthetic_ohlcv("AAA", n_bars=5, seed=3)

    class FakeAK:
        def stock_zh_a_hist(self, **kwargs):
            raw = pd.DataFrame({
                "日期": frame.index.tz_convert(None).strftime("%Y-%m-%d"),
                "开盘": frame["open"].to_numpy(),
                "最高": frame["high"].to_numpy(),
                "最低": frame["low"].to_numpy(),
                "收盘": frame["close"].to_numpy(),
                "成交量": frame["volume"].to_numpy(),
            })
            return raw

        def stock_us_daily(self, symbol, adjust=""):
            raw = frame.rename_axis("date").reset_index()
            raw["date"] = raw["date"].dt.tz_convert(None)
            return raw[["date", "open", "high", "low", "close", "volume"]]

    monkeypatch.setattr(akh, "_import_akshare", lambda: FakeAK())
    cn = AKShareDataHandler("000001", market="cn",
                            start="2024-01-01", end="2024-12-31")
    pd.testing.assert_frame_equal(
        cn.source_ohlcv("000001"),
        normalize_ohlcv_frame(frame),
        check_freq=False,
    )
    us = AKShareDataHandler("AAPL", market="us")
    pd.testing.assert_frame_equal(
        us.source_ohlcv("AAPL"),
        normalize_ohlcv_frame(frame),
        check_freq=False,
    )


def test_akshare_import_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", None)
    with pytest.raises(ImportError, match=r"quantester\[akshare\]"):
        akh._import_akshare()


def test_akshare_bad_market(monkeypatch):
    monkeypatch.setattr(akh, "_import_akshare", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="market must be"):
        AKShareDataHandler("AAPL", market="eu")


# --------------------------------------------------------------------------
# Macro
# --------------------------------------------------------------------------

def test_as_daily_reindex_ffill():
    cal = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    series = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-03"]).tz_localize("UTC"),
    )
    aligned = as_daily_reindex(cal, series)
    assert list(aligned.values) == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_as_daily_reindex_bfill_hard_fails():
    """bfill leaks future macro prints into past bars (synthesis §1.10):
    the trading-feature join must hard-error, not silently allow it."""
    cal = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    series = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-03"]).tz_localize("UTC"),
    )
    with pytest.raises(ValueError, match="bfill"):
        as_daily_reindex(cal, series, method="bfill")


def test_load_world_bank_parses_v2(monkeypatch):
    payload = [
        {"page": 1},
        [
            {"date": "2020", "value": 1.5},
            {"date": "2021", "value": None},
            {"date": "2022", "value": 2.5},
        ],
    ]
    monkeypatch.setattr(wb_mod, "http_get_json", lambda *a, **k: payload)
    series = load_world_bank("FP.CPI.TOTL.ZG", "USA")
    assert list(series.values) == [1.5, 2.5]
    assert series.index[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert series.name == "USA:FP.CPI.TOTL.ZG"


def test_load_nbp_fx_chunks_and_mids(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=30.0):
        calls.append(url)
        # Two overlapping windows; return one rate each.
        if "2024-01-01" in url:
            return {
                "rates": [
                    {"effectiveDate": "2024-01-02", "mid": 4.0},
                ]
            }
        return {
            "rates": [
                {"effectiveDate": "2024-04-01", "mid": 4.1},
            ]
        }

    monkeypatch.setattr(nbp_mod, "http_get_json", fake_get)
    monkeypatch.setattr(nbp_mod, "import_requests", lambda: SimpleNamespace(
        HTTPError=type("HTTPError", (Exception,), {}),
    ))
    series = load_nbp_fx("USD", start="2024-01-01", end="2024-04-10")
    assert len(calls) >= 2  # chunked past 93 days? 100 days -> 2 chunks
    assert list(series.values) == [4.0, 4.1]
    assert series.name == "NBP:A:USD"


def test_load_gus_variable_selects_unit(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "000000000000",
                        "values": [
                            {"year": "2020", "val": 10},
                            {"year": "2021", "val": None},
                            {"year": "2022", "val": 12},
                        ],
                    }
                ]
            }

    class FakeRequests:
        def get(self, *a, **k):
            return FakeResponse()

    monkeypatch.setattr(gus_mod, "import_requests", lambda: FakeRequests())
    series = load_gus_variable(3643, years=[2020, 2021, 2022])
    assert list(series.values) == [10.0, 12.0]
    assert series.name == "GUS:3643"


def test_firewall_parity_new_providers(monkeypatch):
    """New bar providers share StreamingDataHandler firewall semantics."""
    frame = make_synthetic_ohlcv("AAA", n_bars=40, seed=9)
    monkeypatch.setattr(
        stooqh, "http_get_text", lambda *a, **k: _ohlcv_csv(frame),
    )
    monkeypatch.setattr(
        fmph, "http_get_json", lambda *a, **k: _fmp_rows(frame),
    )

    class FakeAK:
        def stock_zh_a_hist(self, **kwargs):
            return pd.DataFrame({
                "日期": frame.index.tz_convert(None).strftime("%Y-%m-%d"),
                "开盘": frame["open"].to_numpy(),
                "最高": frame["high"].to_numpy(),
                "最低": frame["low"].to_numpy(),
                "收盘": frame["close"].to_numpy(),
                "成交量": frame["volume"].to_numpy(),
            })

    monkeypatch.setattr(akh, "_import_akshare", lambda: FakeAK())

    handlers = [
        StooqDataHandler("aapl.us", api_key="k"),
        FMPDataHandler("AAPL", api_key="k"),
        AKShareDataHandler("000001", market="cn"),
    ]
    for handler in handlers:
        handler.prime_data()
        ts, bars = handler.advance()
        handler.set_phase("open", ts)
        assert handler.get_latest_bars(handler.symbols[0], 10**6).empty or (
            handler.get_latest_bars(handler.symbols[0], 10**6).index.max() < ts
            if len(handler.get_latest_bars(handler.symbols[0], 10**6)) else True
        )
        # After first advance at open, no prior bars → empty is OK.
        handler.set_phase("close", ts)
        assert handler.get_latest_bars(handler.symbols[0], 1).index.max() == ts
