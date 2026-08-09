"""Real-data evaluation: hourly BTC Donchian breakout on CCXT history.

Downloads BTC/USD 1h bars via CCXTDataHandler (default Bitstamp; caches under
examples/data/), runs DonchianBreakoutStrategy with FractionalRiskSizer (2%
equity to the 2×ATR stop) and ConservativeFrictionCostModel, then reports
net vs gross, buy-and-hold benchmark, and a truncation leak check.

The indicator windows are the notebook/spec hourly values (SMA200 hours,
Donchian 20 hours, ADX/ATR 14 hours) — not calendar-scaled day ports.

Run from the repo root:  python examples/donchian_breakout/run_ccxt.py
"""

from __future__ import annotations

import pandas as pd

from quantester.analytics.tearsheet import generate_tearsheet
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import (
    FractionalRiskSizer,
    PercentEquitySizer,
    PortfolioManager,
)
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.strategy.examples import BuyAndHoldStrategy
from quantester.validation.truncation import run_truncation_test

from _shared import (
    FRICTION,
    INITIAL_CAPITAL,
    OUTPUT_DIR,
    SYMBOL,
    ZERO,
    load_or_fetch,
    metrics,
    report,
)


def run_donchian(
    df: pd.DataFrame,
    cost_model,
    truncate_last: int | None = None,
) -> PortfolioManager:
    data = df.iloc[:-truncate_last] if truncate_last else df
    handler = HistoricCSVDataHandler({SYMBOL: data})
    strategy = DonchianBreakoutStrategy(handler, SYMBOL)
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=FractionalRiskSizer(0.02),
    )
    BacktestEngine(
        handler, strategy, portfolio, SimulatedExecutionHandler(cost_model),
    ).run_backtest()
    return portfolio


def run_buy_and_hold(df: pd.DataFrame, cost_model) -> PortfolioManager:
    handler = HistoricCSVDataHandler({SYMBOL: df})
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=PercentEquitySizer(1.0),
    )
    BacktestEngine(
        handler,
        BuyAndHoldStrategy(handler),
        portfolio,
        SimulatedExecutionHandler(cost_model),
    ).run_backtest()
    return portfolio


def main():
    print("=" * 72)
    print("Quantester: hourly BTC Donchian breakout on CCXT (Bitstamp 1h)")
    print("=" * 72)

    df = load_or_fetch()
    print(f"bars={len(df)}  range={df.index[0]} → {df.index[-1]}")

    net = run_donchian(df, FRICTION)
    gross = run_donchian(df, ZERO)
    bh = run_buy_and_hold(df, FRICTION)

    print("\n-- performance --")
    report(metrics(net.equity_curve, net, "Donchian net (friction)"))
    report(metrics(gross.equity_curve, gross, "Donchian gross (zero cost)"))
    report(metrics(bh.equity_curve, bh, "Buy & hold BTC"))

    print("\n-- leak check --")
    result = run_truncation_test(
        lambda n: run_donchian(df, FRICTION, truncate_last=n).positions_history,
        n_truncate=48,
    )
    print(result)

    path = OUTPUT_DIR / "donchian_breakout_ccxt_tearsheet.png"
    eq = net.equity_curve
    generate_tearsheet(
        eq, path,
        title="Donchian breakout — real BTC/USD hourly (Bitstamp, net)",
        extra_stats={
            "CAGR": f"{metrics(eq)['cagr']:+.2%}",
            "trades": str(len(net.trades)),
        },
    )
    print(f"\nTearsheet: {path}")


if __name__ == "__main__":
    main()
