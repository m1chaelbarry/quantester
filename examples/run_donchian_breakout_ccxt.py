"""Real-data evaluation: hourly BTC Donchian breakout on CCXT history.

Downloads BTC/USD 1h bars via CCXTDataHandler (default Bitstamp; caches under
examples/data/), runs DonchianBreakoutStrategy with FractionalRiskSizer (2%
equity to the 2×ATR stop) and ConservativeFrictionCostModel, then reports
net vs gross, buy-and-hold benchmark, and a truncation leak check.

The indicator windows are the notebook/spec hourly values (SMA200 hours,
Donchian 20 hours, ADX/ATR 14 hours) — not calendar-scaled day ports.

Run from the repo root:  python examples/run_donchian_breakout_ccxt.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantester.analytics.performance import (
    annualized_sharpe,
    calmar_ratio,
    max_drawdown,
)
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel, CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import FractionalRiskSizer, PortfolioManager
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.strategy.examples import BuyAndHoldStrategy
from quantester.validation.truncation import run_truncation_test

DATA_DIR = Path("examples/data")
OUTPUT_DIR = Path("examples/output")
CACHE = DATA_DIR / "BTCUSD_bitstamp_1h.csv"
SYMBOL = "BTC/USD"
INITIAL_CAPITAL = 25_000.0
PERIODS = 24 * 365

FRICTION = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
ZERO = CostModel(
    fixed_commission=0.0, per_share_commission=0.0, spread_pct=0.0,
    slippage_vol_coef=0.0, impact_coef=0.0,
)


def load_or_fetch() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["datetime"], index_col="datetime")
    from quantester.data.ccxt_handler import CCXTDataHandler

    print("Fetching BTC/USD 1h from Bitstamp ...")
    handler = CCXTDataHandler(
        SYMBOL, exchange="bitstamp", timeframe="1h",
        start="2019-01-01", limit=1000,
    )
    df = handler._data[SYMBOL]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index_label="datetime")
    return df


def run(df: pd.DataFrame, cost_model, truncate_last: int | None = None,
        buy_and_hold: bool = False) -> PortfolioManager:
    data = df.iloc[:-truncate_last] if truncate_last else df
    handler = HistoricCSVDataHandler({SYMBOL: data})
    if buy_and_hold:
        strategy = BuyAndHoldStrategy(handler)
        from quantester.portfolio.portfolio import PercentEquitySizer
        portfolio = PortfolioManager(
            handler, INITIAL_CAPITAL, sizer=PercentEquitySizer(1.0),
        )
    else:
        strategy = DonchianBreakoutStrategy(handler, SYMBOL)
        portfolio = PortfolioManager(
            handler, INITIAL_CAPITAL, sizer=FractionalRiskSizer(0.02),
        )
    BacktestEngine(
        handler, strategy, portfolio, SimulatedExecutionHandler(cost_model),
    ).run_backtest()
    return portfolio


def metrics(equity: pd.Series, portfolio: PortfolioManager | None = None,
            label: str = "") -> dict:
    years = max(len(equity) / PERIODS, 1e-12)
    row = {
        "label": label,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "sharpe": annualized_sharpe(equity, periods=PERIODS),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "calmar": calmar_ratio(equity, periods=PERIODS),
    }
    if portfolio is not None:
        row["trades"] = len(portfolio.trades)
        row["friction"] = float(sum(f.total_cost for f in portfolio.fills))
    return row


def report(row: dict) -> None:
    print(
        f"  {row['label']:<28}  ret={row['total_return']:+.2%}  "
        f"cagr={row['cagr']:+.2%}  sharpe={row['sharpe']:.3f}  "
        f"maxDD={row['max_dd']:.2%}  trades={row.get('trades', '-')}"
    )


def main():
    print("=" * 72)
    print("Quantester: hourly BTC Donchian breakout on CCXT (Bitstamp 1h)")
    print("=" * 72)

    df = load_or_fetch()
    print(f"bars={len(df)}  range={df.index[0]} → {df.index[-1]}")

    net = run(df, FRICTION)
    gross = run(df, ZERO)
    bh = run(df, FRICTION, buy_and_hold=True)

    print("\n-- performance --")
    report(metrics(net.equity_curve, net, "Donchian net (friction)"))
    report(metrics(gross.equity_curve, gross, "Donchian gross (zero cost)"))
    report(metrics(bh.equity_curve, bh, "Buy & hold BTC"))

    print("\n-- leak check --")
    result = run_truncation_test(
        lambda n: run(df, FRICTION, truncate_last=n).positions_history,
        n_truncated=48,
    )
    print(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
