"""End-to-end example: free real market data -> one-call backtest.

Run from the repo root:  python examples/market_data/run.py

Requires the optional data extras and network access:
    pip install "quantester[data]"

Without network or extras the script prints guidance and exits cleanly.
"""

from __future__ import annotations

from quantester import MovingAverageCrossStrategy, load_crypto, load_yahoo, run_backtest


def describe(frames: dict) -> None:
    for symbol, df in frames.items():
        print(
            f"  {symbol}: bars={len(df)}  "
            f"range={df.index[0].date()} -> {df.index[-1].date()}"
        )


def yfinance_demo() -> None:
    print("\n[yfinance] AAPL/MSFT daily, 2022-2024 (adjusted OHLC)")
    data = load_yahoo(
        ["AAPL", "MSFT"], start="2022-01-01", end="2025-01-01", interval="1d",
    )
    describe(data)
    result = run_backtest(
        data, MovingAverageCrossStrategy, symbol="AAPL", fast=10, slow=40,
    )
    print(
        f"  MA(10/40) on AAPL: sharpe={result.sharpe:+.3f}  "
        f"mdd={result.max_drawdown:+.2%}  fills={len(result.fills)}"
    )


def ccxt_demo() -> None:
    print("\n[ccxt] BTC/USD + ETH/USD daily from Coinbase, 2023-2024")
    data = load_crypto(
        ["BTC/USD", "ETH/USD"], exchange="coinbase",
        timeframe="1d", start="2023-01-01", end="2025-01-01",
    )
    describe(data)
    result = run_backtest(
        data, MovingAverageCrossStrategy, symbol="BTC/USD", fast=10, slow=40,
    )
    print(
        f"  MA(10/40) on BTC/USD: sharpe={result.sharpe:+.3f}  "
        f"mdd={result.max_drawdown:+.2%}  fills={len(result.fills)}"
    )


def main() -> None:
    print("=" * 72)
    print("Quantester example: free market-data providers (yfinance, ccxt)")
    print("=" * 72)
    for name, demo in [("yfinance", yfinance_demo), ("ccxt", ccxt_demo)]:
        try:
            demo()
        except ImportError as exc:
            print(f"\n[{name}] skipped: {exc}")
        except Exception as exc:  # network/HTTP/rate-limit failures
            print(
                f"\n[{name}] unavailable ({type(exc).__name__}: {exc}); "
                "the provider needs network access to fetch data."
            )


if __name__ == "__main__":
    main()
