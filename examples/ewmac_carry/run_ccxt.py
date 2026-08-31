#!/usr/bin/env python3
"""Fetch Binance USDT-M BTC + Deribit DVOL extras, then run the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from quantester.data.ccxt_handler import CCXTDataHandler

from run import main as run_pipeline
from market import SYMBOL


def load_ccxt():
    handler = CCXTDataHandler(
        SYMBOL, exchange="binance", timeframe="1d",
        start="2021-01-01", include_extras=True, dvol_exchange="deribit",
    )
    return handler.source_ohlcv(SYMBOL)


if __name__ == "__main__":
    df = None
    try:
        df = load_ccxt()
        out = HERE / "output"
        out.mkdir(exist_ok=True)
        df.to_csv(out / "BTC_PERP_CCXT.csv", index_label="datetime")
        extras = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
        print(f"CCXT bars={len(df)} extras={extras}")
    except Exception as exc:
        print(f"CCXT fetch failed ({exc}); falling back to synthetic demo.")
        df = None
    run_pipeline(ohlcv=df)
