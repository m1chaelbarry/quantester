"""Synthetic evaluation: hourly BTC Donchian breakout (SMA200 + ADX + ATR risk).

Builds a trending synthetic hourly path, runs DonchianBreakoutStrategy under
delay=1 with FractionalRiskSizer (2% equity to the 2×ATR stop) and the
ConservativeFrictionCostModel, then prints a short performance summary.

Run from the repo root:  python examples/donchian_breakout/run.py

For real CCXT hourly BTC evaluation see run_donchian_breakout_ccxt.py.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

import pandas as pd

from quantester.analytics.performance import annualized_sharpe, max_drawdown
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import FractionalRiskSizer, PortfolioManager
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv
from quantester.validation.truncation import run_truncation_test

SYMBOL = "BTC/USD"
INITIAL_CAPITAL = 25_000.0
PERIODS = 24 * 365  # hourly crypto calendar
FRICTION = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)


def make_hourly_trend(n: int = 3_000, seed: int = 7) -> pd.DataFrame:
    """GBM path with positive drift so the SMA200 regime can engage."""
    df = make_synthetic_ohlcv(
        SYMBOL, n_bars=n, s0=30_000.0, mu=0.35, sigma=0.55, seed=seed,
    )
    df = df.copy()
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="h", name="datetime")
    return df


def run_backtest(df: pd.DataFrame, truncate_last: int | None = None) -> PortfolioManager:
    data = df.iloc[:-truncate_last] if truncate_last else df
    handler = HistoricCSVDataHandler({SYMBOL: data})
    strategy = DonchianBreakoutStrategy(handler, SYMBOL)
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=FractionalRiskSizer(0.02),
    )
    BacktestEngine(
        handler, strategy, portfolio, SimulatedExecutionHandler(FRICTION),
    ).run_backtest()
    return portfolio


def main():
    print("=" * 72)
    print("Quantester example: BTC hourly Donchian breakout (SMA200/ADX/ATR)")
    print("=" * 72)

    df = make_hourly_trend()
    portfolio = run_backtest(df)
    equity = portfolio.equity_curve
    dd = max_drawdown(equity)
    print(f"bars={len(df)}  trades={len(portfolio.trades)}  fills={len(portfolio.fills)}")
    print(f"total return={equity.iloc[-1] / equity.iloc[0] - 1:.2%}")
    print(f"sharpe_{PERIODS}={annualized_sharpe(equity, periods=PERIODS):.3f}")
    print(f"max_dd={dd['max_drawdown']:.2%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "donchian_breakout_tearsheet.png"
    generate_tearsheet(
        equity, path,
        title="Donchian Breakout (synthetic hourly)",
        extra_stats={"friction_paid": f"{sum(f.total_cost for f in portfolio.fills):,.0f}"},
    )
    print(f"Tearsheet: {path}")

    result = run_truncation_test(
        lambda n: run_backtest(df, truncate_last=n).positions_history,
        n_truncated=30,
    )
    print(result)


if __name__ == "__main__":
    main()
