"""End-to-end example: free real market data (yfinance + ccxt) -> event-driven
backtest with the same temporal firewall as the CSV feed.

Run from the repo root:  python examples/market_data/run.py

Requires the optional data extras and network access:
    pip install "quantester[data]"

Both providers return historic batches that the engine replays bar-by-bar;
nothing about the strategy/portfolio/execution wiring changes versus the CSV
feed. Without network or extras the script prints guidance and exits cleanly.
"""

from __future__ import annotations

from quantester.analytics.performance import summarize
from quantester.engine import BacktestEngine
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy

INITIAL_CAPITAL = 100_000.0


def run_backtest(handler, symbol: str, fast: int = 10, slow: int = 40):
    strategy = MovingAverageCrossStrategy(handler, symbol, fast=fast, slow=slow,
                                          direction="both")
    portfolio = PortfolioManager(handler, INITIAL_CAPITAL,
                                 sizer=PercentEquitySizer(0.9))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(CostModel()))
    engine.run_backtest()
    return portfolio


def describe(handler) -> None:
    first, last = handler._master_index[0], handler._master_index[-1]
    print(f"  symbols={handler.symbols}  bars={len(handler._master_index)}  "
          f"range={first.date()} -> {last.date()}")


def yfinance_demo() -> None:
    from quantester.data import YFinanceDataHandler

    print("\n[yfinance] AAPL/MSFT daily, 2022-2024 (adjusted OHLC)")
    handler = YFinanceDataHandler(["AAPL", "MSFT"], start="2022-01-01",
                                  end="2025-01-01", interval="1d",
                                  auto_adjust=True)
    describe(handler)
    portfolio = run_backtest(handler, "AAPL")
    stats = summarize(portfolio.equity_curve)
    print(f"  MA(10/40) on AAPL: sharpe={stats['sharpe']:+.3f}  "
          f"mdd={stats['max_drawdown']:+.2%}  fills={len(portfolio.fills)}")


def ccxt_demo() -> None:
    from quantester.data import CCXTDataHandler

    print("\n[ccxt] BTC/USD + ETH/USD daily from Coinbase, 2023-2024")
    handler = CCXTDataHandler(["BTC/USD", "ETH/USD"], exchange="coinbase",
                              timeframe="1d", start="2023-01-01",
                              end="2025-01-01")
    describe(handler)
    portfolio = run_backtest(handler, "BTC/USD")
    stats = summarize(portfolio.equity_curve)
    print(f"  MA(10/40) on BTC/USD: sharpe={stats['sharpe']:+.3f}  "
          f"mdd={stats['max_drawdown']:+.2%}  fills={len(portfolio.fills)}")


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
            print(f"\n[{name}] unavailable ({type(exc).__name__}: {exc}); "
                  "the provider needs network access to fetch data.")


if __name__ == "__main__":
    main()
