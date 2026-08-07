"""End-to-end example: BTC three-tranche pullback ladder with latching.

Strategy (TranchePullbackStrategy): long only above the 200-day SMA; on the
flat -> active transition the 20-day peak close and ATR(14) are LATCHED and
three resting limit orders go out at peak - k*1.5*ATR (k = 1, 2, 3) sized to
25% / 35% / 40% of latch-time equity. Exit all tranches at the next open when
the close recovers to SMA(5); hard stop at close <= peak - 5.0*ATR.

Safeguards demonstrated:
- ConservativeFrictionCostModel: C_trade = 2 * (S_bid-ask/2 + mu_fee) charged
  on every fill (maker tranches included — deliberately pessimistic).
- DailyDrawdownBreaker: a 4.5% intraday equity loss (0.5% cushion under the
  5% prop-firm limit) liquidates everything, cancels the resting ladder, and
  suspends entries until the daily rollover.

This synthetic GBM path is a PLUMBING demo (fills, latching, breaker,
truncation check) — GBM has no mean-reversion edge, so churn pays only
friction here. The real-data profitability evaluation lives in
examples/tranche_pullback/run_ccxt.py.

Run from the repo root:  python examples/tranche_pullback/run.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

import numpy as np
import pandas as pd

from quantester.analytics.performance import summarize
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.portfolio.risk import DailyDrawdownBreaker
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv
from quantester.validation.truncation import run_truncation_test

INITIAL_CAPITAL = 25_000.0  # small-account flavor
SYMBOL = "BTC"
FRICTION = ConservativeFrictionCostModel(
    spread_pct=0.0002,  # 2 bps full bid-ask -> 2x half-spread charged per fill
    fee_rate=0.0004,    # 4 bps taker fee   -> doubled on notional
)


def run_backtest(df: pd.DataFrame, breaker: bool = True,
                 truncate_last: int | None = None) -> PortfolioManager:
    if truncate_last:
        df = df.iloc[:-truncate_last]
    handler = HistoricCSVDataHandler({SYMBOL: df})
    strategy = TranchePullbackStrategy(handler, SYMBOL)
    portfolio = PortfolioManager(
        handler,
        INITIAL_CAPITAL,
        sizer=PercentEquitySizer(1.0),  # fractions travel on signal strength
        drawdown_breaker=DailyDrawdownBreaker(0.045) if breaker else None,
    )
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(FRICTION))
    engine.run_backtest()
    return portfolio


def report(portfolio: PortfolioManager, label: str) -> None:
    stats = summarize(portfolio.equity_curve)
    print(f"[{label}] total return {stats['total_return']:+.2%}  "
          f"sharpe {stats['sharpe']:+.3f}  max DD {stats['max_drawdown']:+.2%}  "
          f"calmar {stats['calmar']:+.3f}")
    print(f"[{label}] round-trips: {len(portfolio.trades)}  "
          f"fills: {len(portfolio.fills)}  "
          f"friction paid: "
          f"{sum(f.total_cost for f in portfolio.fills):,.2f}")
    breaker = portfolio.drawdown_breaker
    if breaker is not None:
        print(f"[{label}] circuit-breaker trips: {breaker.triggered_count}")


def make_crash_scenario() -> pd.DataFrame:
    """Handcrafted path: gentle bull ramp (latch + ladder), a dip filling the
    first two tranches, then a -25% black-swan day to fire the breaker."""
    n_ramp = 200
    closes = np.linspace(90.0, 100.0, n_ramp)
    bars, prev = [], closes[0]
    for c in closes:
        bars.append((prev, max(prev, c), min(prev, c), c))
        prev = c
    bars += [
        (100.0, 100.0, 99.80, 99.83),  # dip fills T1, T2 (60% deployed)
        (99.83, 99.90, 99.82, 99.88),  # quiet day, still long
        (99.88, 99.90, 74.50, 75.00),  # black swan: -25% while long
        (74.00, 74.50, 73.50, 74.20),  # breaker liquidation fills at this open
        (74.20, 74.30, 74.10, 74.25),  # halted: any entry signal is dropped
        (74.25, 74.40, 74.20, 74.35),  # rollover: halt cleared
    ]
    idx = pd.bdate_range("2024-01-01", periods=len(bars))
    return pd.DataFrame(
        {"open": [b[0] for b in bars], "high": [b[1] for b in bars],
         "low": [b[2] for b in bars], "close": [b[3] for b in bars],
         "volume": [1e6] * len(bars)},
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


def main():
    print("=" * 72)
    print("Quantester example: BTC tranche pullback ladder (25/35/40, latched)")
    print("=" * 72)

    # Scenario 1: long-run synthetic BTC-like bull market with pullbacks.
    df = make_synthetic_ohlcv(SYMBOL, n_bars=1200, s0=20_000.0, mu=0.60,
                              sigma=0.60, seed=11)
    portfolio = run_backtest(df)
    report(portfolio, "bull-market GBM")
    tranche_buys = [f for f in portfolio.fills if f.direction == "BUY"]
    if portfolio.trades:
        print(f"  tranche entries: {len(tranche_buys)} across "
              f"{len(portfolio.trades)} round-trips (avg "
              f"{len(tranche_buys) / len(portfolio.trades):.1f} tranches filled "
              f"per position)")

    stats = generate_tearsheet(
        portfolio.equity_curve,
        OUTPUT_DIR / "tranche_pullback_tearsheet.png",
        title="Tranche pullback ladder on synthetic BTC",
        extra_stats={"friction_paid": f"{sum(f.total_cost for f in portfolio.fills):,.0f}"},
    )
    print(f"Tearsheet: {OUTPUT_DIR / 'tranche_pullback_tearsheet.png'}")

    result = run_truncation_test(
        lambda n: run_backtest(df, truncate_last=n).positions_history,
        n_truncated=30,
    )
    print(result)

    # Scenario 2: engineered black swan to exercise the 4.5% daily breaker.
    # Note the layering: the strategy's own Kaufman-rule stop (low trigger,
    # close fill) exits at the crash bar's close, while the account-level
    # breaker liquidates at the NEXT open and suspends signals — slightly
    # worse fill here, but the breaker is the backstop that still fires when
    # a strategy has no stop, is halted, or the loss comes from other books.
    crash = make_crash_scenario()
    protected = run_backtest(crash, breaker=True)
    unprotected = run_backtest(crash, breaker=False)
    print("\nBlack-swan scenario (-25% day while the ladder stacks in):")
    report(protected, "with 4.5% breaker")
    report(unprotected, "without breaker")
    trip_day_fills = [f for f in protected.fills if f.direction == "SELL"]
    for f in trip_day_fills:
        print(f"  liquidation fill: {f.timestamp.date()} qty={f.quantity:.4f} "
              f"px={f.fill_price:,.2f}")
    print("  while halted the portfolio drops every signal (entries and "
          "exits); the parked liquidation order retries each open until "
          "filled, and the halt clears at the next daily rollover.")


if __name__ == "__main__":
    main()
