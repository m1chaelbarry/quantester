"""End-to-end example: free real market data -> one-call backtest.

Run from the repo root:  python examples/market_data/run.py

Requires the optional data extras and network access:
    pip install "quantester[data]"

Stooq / FMP also need API keys:
    export QUANTESTER_STOOQ_API_KEY=...
    export QUANTESTER_FMP_API_KEY=...

Without network, extras, or keys the script prints guidance and continues.
"""

from __future__ import annotations

from quantester import (
    MovingAverageCrossStrategy,
    load_akshare,
    load_crypto,
    load_fmp,
    load_stooq,
    load_yahoo,
    run_backtest,
)


def describe(frames: dict) -> None:
    for symbol, df in frames.items():
        print(
            f"  {symbol}: bars={len(df)}  "
            f"range={df.index[0].date()} -> {df.index[-1].date()}"
        )


def _run_ma(data: dict, symbol: str, label: str) -> None:
    describe(data)
    result = run_backtest(
        data, MovingAverageCrossStrategy, symbol=symbol, fast=10, slow=40,
    )
    print(
        f"  MA(10/40) on {symbol}: sharpe={result.sharpe:+.3f}  "
        f"mdd={result.max_drawdown:+.2%}  fills={len(result.fills)}  [{label}]"
    )


def yfinance_demo() -> None:
    print("\n[yfinance] AAPL/MSFT daily, 2022-2024 (adjusted OHLC)")
    data = load_yahoo(
        ["AAPL", "MSFT"], start="2022-01-01", end="2025-01-01", interval="1d",
    )
    _run_ma(data, "AAPL", "yfinance")


def ccxt_demo() -> None:
    print("\n[ccxt] BTC/USD + ETH/USD daily from Coinbase, 2023-2024")
    data = load_crypto(
        ["BTC/USD", "ETH/USD"], exchange="coinbase",
        timeframe="1d", start="2023-01-01", end="2025-01-01",
    )
    _run_ma(data, "BTC/USD", "ccxt")


def stooq_demo() -> None:
    print("\n[stooq] aapl.us daily (needs QUANTESTER_STOOQ_API_KEY)")
    data = load_stooq(
        "aapl.us", start="2022-01-01", end="2025-01-01", interval="d",
    )
    _run_ma(data, "aapl.us", "stooq")


def fmp_demo() -> None:
    print("\n[fmp] AAPL EOD (needs QUANTESTER_FMP_API_KEY)")
    data = load_fmp("AAPL", start="2022-01-01", end="2025-01-01")
    _run_ma(data, "AAPL", "fmp")


def akshare_demo() -> None:
    print("\n[akshare] A-share 000001 daily (forward-adjusted)")
    data = load_akshare(
        "000001", market="cn", start="2022-01-01", end="2025-01-01",
    )
    _run_ma(data, "000001", "akshare")


def main() -> None:
    print("=" * 72)
    print("Quantester example: free market-data providers")
    print("=" * 72)
    demos = [
        ("yfinance", yfinance_demo),
        ("ccxt", ccxt_demo),
        ("stooq", stooq_demo),
        ("fmp", fmp_demo),
        ("akshare", akshare_demo),
    ]
    for name, demo in demos:
        try:
            demo()
        except ImportError as exc:
            print(f"\n[{name}] skipped: {exc}")
        except Exception as exc:  # network/HTTP/rate-limit/missing-key failures
            print(
                f"\n[{name}] unavailable ({type(exc).__name__}: {exc}); "
                "needs network and, for Stooq/FMP, a free API key."
            )


if __name__ == "__main__":
    main()
