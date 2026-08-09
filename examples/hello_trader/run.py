"""The shortest Quantester backtest — readable for traders, not just coders.

What this does:
  1. Builds synthetic daily prices for one symbol (AAA)
  2. Runs a moving-average crossover
  3. Prints a plain-English summary
  4. Checks for look-ahead leaks
  5. Saves a tearsheet chart

Run from the repo root::

    python examples/hello_trader/run.py
"""

from __future__ import annotations

from pathlib import Path

from quantester import (
    MovingAverageCrossStrategy,
    generate_tearsheet,
    make_synthetic_ohlcv,
    run_backtest,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    print("=" * 72)
    print("Quantester hello: MA crossover in a few lines")
    print("=" * 72)

    prices = make_synthetic_ohlcv("AAA", n_bars=504, seed=1)
    result = run_backtest(
        prices,
        MovingAverageCrossStrategy,
        symbol="AAA",
        fast=10,
        slow=40,
        direction="both",
        capital=100_000.0,
        equity_pct=0.9,
    )
    result.print_summary()
    print(result.check_lookahead())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chart = OUTPUT_DIR / "hello_tearsheet.png"
    generate_tearsheet(
        result.equity,
        chart,
        title="Hello trader: MA(10/40) on AAA",
    )
    print(f"\nTearsheet saved to {chart}")


if __name__ == "__main__":
    main()
