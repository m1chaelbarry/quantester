"""Binance Vision public dumps (USDT-M) — used when fapi REST is geo-capped.

Daily ``metrics`` zips hold 5-minute open interest. REST
``openInterestHist`` only covers ~30 days, which is not enough for a
crowded-long filter on a multi-year Combined Forecast study.

Official layout: https://github.com/binance/binance-public-data
"""

from __future__ import annotations

import csv
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

VISION = "https://data.binance.vision/data/futures/um"
_UA = {"User-Agent": "quantester/0.1 (research; Binance Vision dumps)"}


def _get(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers=_UA)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def last_open_interest_from_metrics_zip(blob: bytes) -> tuple[pd.Timestamp, float] | None:
    """Return (UTC timestamp, sum_open_interest) from the last 5-minute row."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            rows = list(csv.reader(io.TextIOWrapper(raw, encoding="utf-8")))
    if not rows:
        return None
    header = rows[0]
    data = rows[1:] if header and "open_interest" in ",".join(header).lower() else rows
    if not data:
        return None
    last = data[-1]
    if header and "open_interest" in ",".join(header).lower():
        cols = {name: i for i, name in enumerate(header)}
        ts_raw = last[cols.get("create_time", 0)]
        oi_raw = last[cols.get("sum_open_interest", 2)]
    else:
        ts_raw, oi_raw = last[0], last[2]
    ts = pd.Timestamp(ts_raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts, float(oi_raw)


def fetch_um_daily_open_interest(
    symbol: str = "BTCUSDT",
    start="2021-01-01",
    end=None,
    max_workers: int = 24,
) -> pd.Series:
    """Last-print open interest per UTC day from Vision daily metrics zips."""
    start_ts = pd.Timestamp(start)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    if end is None:
        end_ts = pd.Timestamp.now(tz="UTC").normalize()
    else:
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    days = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D", tz="UTC")

    def _one(day: pd.Timestamp):
        ymd = day.strftime("%Y-%m-%d")
        url = f"{VISION}/daily/metrics/{symbol}/{symbol}-metrics-{ymd}.zip"
        try:
            blob = _get(url)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        except URLError:
            return None
        parsed = last_open_interest_from_metrics_zip(blob)
        return parsed

    points: list[tuple[pd.Timestamp, float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, day): day for day in days}
        for fut in as_completed(futs):
            got = fut.result()
            if got is not None:
                points.append(got)
    if not points:
        return pd.Series(dtype=float)
    points.sort(key=lambda p: p[0])
    idx = pd.DatetimeIndex([p[0] for p in points], name="datetime")
    return pd.Series([p[1] for p in points], index=idx, dtype=float)
