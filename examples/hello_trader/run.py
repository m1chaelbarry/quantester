"""The shortest Quantester backtest — readable for traders, not just coders.

What this does:
  1. Builds synthetic daily prices for one symbol (AAA)
  2. Runs a moving-average crossover (buy when the fast average crosses above
     the slow average; sell / short on the opposite cross)
  3. Prints a plain-English summary (return, Sharpe, drawdown, trade count)
  4. Saves a tearsheet chart

Run from the repo root::

    python examples/hello_trader/run.py

No deep imports, no manual wiring of five modules — ``run_backtest`` does that.
When you outgrow this, copy ``examples/ma_cross/`` or
``examples/custom_strategy/`` and see ``docs/for-traders.md``.
"""

from __future__ import annotations

from pathlib import Path

from quantester import MovingAverageCrossStrategy, run_backtest
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.utils.synthetic import make_synthetic_ohlcv

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    print("=" * 72)
    print("Quantester hello: MA crossover in a few lines")
    print("=" * 72)

    # Prices: one symbol, ~2 years of daily bars. Seed keeps the run repeatable.
    prices = make_synthetic_ohlcv("AAA", n_bars=504, seed=1)

    # Strategy idea in plain words:
    #   fast=10, slow=40  →  10-day average vs 40-day average
    #   direction="both"  →  go long on up-cross, short on down-cross
    result = run_backtest(
        prices,
        MovingAverageCrossStrategy,
        symbol="AAA",
        fast=10,
        slow=40,
        direction="both",
        capital=100_000.0,
        equity_pct=0.9,  # use up to 90% of the account per signal
    )
    result.print_summary()

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
