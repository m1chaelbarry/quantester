#!/usr/bin/env python3
"""Fetch Binance USDT-M BTC + Deribit DVOL, then run the research pipeline.

Uses ``binanceusdm`` with ``geo_safe=True`` (www.binance.com) because regional
``fapi.binance.com`` returns HTTP 451. Open interest comes from Binance Vision
daily metrics (REST OI history is ~30 days). Does **not** fall back to GBM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from quantester.data.binance_vision import fetch_um_daily_open_interest
from quantester.data.ccxt_handler import CCXTDataHandler
from quantester.data.crypto_extras import attach_extras

from run import main as run_pipeline
from market import SYMBOL

NATIVE = "BTCUSDT"


def _coverage(df, col: str) -> str:
    if col not in df.columns:
        return "absent"
    s = df[col]
    return f"{int(s.notna().sum())}/{len(s)} ({s.notna().mean():.0%})"


def load_binance_um(start: str = "2021-01-01"):
    handler = CCXTDataHandler(
        SYMBOL,
        exchange="binanceusdm",
        timeframe="1d",
        start=start,
        include_extras=True,
        dvol_exchange="deribit",
        geo_safe=True,
    )
    df = handler.source_ohlcv(SYMBOL)
    print(
        f"CCXT {SYMBOL} bars={len(df)} "
        f"{df.index[0].date()} → {df.index[-1].date()} "
        f"funding={_coverage(df, 'funding_rate')} "
        f"rest_oi={_coverage(df, 'open_interest')} "
        f"dvol={_coverage(df, 'dvol')}"
    )
    print("  fetching Binance Vision daily open-interest metrics …")
    oi = fetch_um_daily_open_interest(
        NATIVE, start=df.index[0], end=df.index[-1],
    )
    if len(oi):
        df = attach_extras(df, open_interest=oi)
        print(f"  vision OI prints={len(oi)}  aligned={_coverage(df, 'open_interest')}")
    else:
        print("  vision OI empty; leaving REST OI (fail-open)")
    return df


def main():
    p = argparse.ArgumentParser(description="EWMAC+carry on live Binance USDT-M BTC")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--full", action="store_true", default=True)
    p.add_argument("--quick", action="store_true", help="lighter MCPT (demo scale)")
    args, extra = p.parse_known_args()
    df = load_binance_um(start=args.start)
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "BTC_PERP_CCXT.csv", index_label="datetime")
    extras = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    print(f"saved {out / 'BTC_PERP_CCXT.csv'} extras={extras}")
    if args.quick:
        sys.argv = [sys.argv[0], *extra]
    else:
        sys.argv = [sys.argv[0], "--full", *extra]
    return run_pipeline(ohlcv=df)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Live fetch/pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
