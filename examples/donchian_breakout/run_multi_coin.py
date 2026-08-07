"""Multi-coin daily long-only Donchian portfolio backtest.

Runs one DonchianBreakoutStrategy(long_only=True) per major on a shared
handler, with book-level risk budgeting (default: 2% total risk split across
the universe so concurrent breakouts cannot stack full per-name risk).

Universe default: BTC, ETH, LTC, XRP, BCH on Bitstamp daily (cached under
examples/data/). LTC/BCH are retained so the study shows why universe
filtering matters — prefer --universe BTC/USD,ETH/USD,XRP/USD for the
production sleeve.

Run from the repo root:
  python examples/donchian_breakout/run_multi_coin.py
  python examples/donchian_breakout/run_multi_coin.py --risk-budget 0.02 \\
      --universe BTC/USD,ETH/USD,XRP/USD
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "examples" / "data"
OUTPUT_DIR = HERE / "output"

import numpy as np
import pandas as pd

from quantester.analytics.performance import annualized_sharpe, max_drawdown
from quantester.data.ccxt_handler import CCXTDataHandler
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import (
    FractionalRiskSizer,
    PercentEquitySizer,
    PortfolioManager,
)
from quantester.portfolio.risk import spectral_risk_attribution
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.strategy.examples import BuyAndHoldStrategy

FRICTION = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
INITIAL_CAPITAL = 100_000.0
PERIODS = 365
DEFAULT_UNIVERSE = ("BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD", "BCH/USD")


def cache_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.replace('/', '')}_bitstamp_1d.csv"


def load_or_fetch(symbol: str) -> pd.DataFrame:
    path = cache_path(symbol)
    if path.exists():
        return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {symbol} daily from Bitstamp ...")
    handler = CCXTDataHandler(
        symbol, exchange="bitstamp", timeframe="1d",
        start="2017-01-01", limit=1000,
    )
    df = handler._data[symbol]
    df.to_csv(path, index_label="datetime")
    return df


def load_universe(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    frames = {s: load_or_fetch(s) for s in symbols}
    start = max(df.index.min() for df in frames.values())
    end = min(df.index.max() for df in frames.values())
    return {s: df.loc[start:end].copy() for s, df in frames.items()}


def run_portfolio(frames: dict, risk_per_name: float) -> PortfolioManager:
    handler = HistoricCSVDataHandler(frames)
    strategies = [
        DonchianBreakoutStrategy(handler, symbol, long_only=True)
        for symbol in frames
    ]
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=FractionalRiskSizer(risk_per_name),
    )
    BacktestEngine(
        handler, strategies, portfolio, SimulatedExecutionHandler(FRICTION),
    ).run_backtest()
    return portfolio


def run_single(symbol: str, df: pd.DataFrame, risk: float = 0.02) -> PortfolioManager:
    handler = HistoricCSVDataHandler({symbol: df})
    strategy = DonchianBreakoutStrategy(handler, symbol, long_only=True)
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=FractionalRiskSizer(risk),
    )
    BacktestEngine(
        handler, strategy, portfolio, SimulatedExecutionHandler(FRICTION),
    ).run_backtest()
    return portfolio


def run_bh(symbol: str, df: pd.DataFrame) -> PortfolioManager:
    handler = HistoricCSVDataHandler({symbol: df})
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=PercentEquitySizer(1.0),
    )
    BacktestEngine(
        handler, BuyAndHoldStrategy(handler), portfolio,
        SimulatedExecutionHandler(FRICTION),
    ).run_backtest()
    return portfolio


def summarize(equity: pd.Series, portfolio: PortfolioManager | None = None) -> dict:
    years = max(len(equity) / PERIODS, 1e-12)
    row = {
        "ret": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "sharpe": annualized_sharpe(equity, periods=PERIODS),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "trades": len(portfolio.trades) if portfolio is not None else None,
    }
    return row


def report(label: str, row: dict) -> None:
    trades = row["trades"] if row["trades"] is not None else "-"
    print(
        f"  {label:<28} ret={row['ret']:+7.1%}  cagr={row['cagr']:+6.1%}  "
        f"sharpe={row['sharpe']:+.3f}  dd={row['max_dd']:6.1%}  trades={trades}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe",
        default=",".join(DEFAULT_UNIVERSE),
        help="Comma-separated CCXT symbols",
    )
    parser.add_argument(
        "--risk-budget", type=float, default=0.02,
        help="Total book risk fraction split evenly across the universe",
    )
    args = parser.parse_args()
    symbols = tuple(s.strip() for s in args.universe.split(",") if s.strip())
    risk_per_name = args.risk_budget / max(len(symbols), 1)

    print("=" * 72)
    print("Daily long-only Donchian — multi-coin portfolio")
    print("=" * 72)
    frames = load_universe(symbols)
    start = min(df.index.min() for df in frames.values())
    end = max(df.index.max() for df in frames.values())
    print(f"Universe: {', '.join(symbols)}")
    print(f"Window:   {start.date()} → {end.date()}")
    print(f"Risk:     budget={args.risk_budget:.2%}  per-name={risk_per_name:.3%}")

    print("\n-- standalone (2% risk each, for comparison) --")
    singles = {}
    for symbol, df in frames.items():
        port = run_single(symbol, df, risk=0.02)
        singles[symbol] = port
        report(symbol, summarize(port.equity_curve, port))

    print("\n-- combined book --")
    book = run_portfolio(frames, risk_per_name=risk_per_name)
    report(f"portfolio ({risk_per_name:.2%}/name)", summarize(book.equity_curve, book))
    bh_btc = run_bh("BTC/USD", frames["BTC/USD"]) if "BTC/USD" in frames else None
    if bh_btc is not None:
        report("B&H BTC", summarize(bh_btc.equity_curve, bh_btc))

    # Leverage path
    hist = pd.DataFrame(
        book._equity_history, columns=["ts", "equity", "cash", "gross"],
    )
    hist["lev"] = hist["gross"] / hist["equity"].replace(0, np.nan)
    print(
        f"\n  peak leverage={hist['lev'].max():.2f}x  "
        f"median={hist['lev'].median():.2f}x  "
        f"P(lev>1)={(hist['lev'] > 1).mean():.1%}"
    )

    rets = pd.DataFrame(
        {s: p.equity_curve.pct_change() for s, p in singles.items()}
    ).dropna(how="all")
    corr = rets.corr()
    mean_corr = float(
        corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()
    )
    print(f"  mean strategy-return corr={mean_corr:.3f}")
    attr = spectral_risk_attribution(rets.fillna(0.0))
    print(f"  PC1 risk share={attr.iloc[0]['risk_share']:.1%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = OUTPUT_DIR / "multi_coin_summary.txt"
    lines = [
        f"universe={','.join(symbols)}",
        f"start={start}",
        f"end={end}",
        f"risk_budget={args.risk_budget}",
        f"risk_per_name={risk_per_name}",
        f"portfolio_sharpe={summarize(book.equity_curve)['sharpe']:.6f}",
        f"portfolio_ret={summarize(book.equity_curve)['ret']:.6f}",
        f"portfolio_max_dd={summarize(book.equity_curve)['max_dd']:.6f}",
        f"mean_strategy_corr={mean_corr:.6f}",
        f"pc1_risk_share={attr.iloc[0]['risk_share']:.6f}",
        f"peak_leverage={hist['lev'].max():.6f}",
    ]
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {summary}")
    print("Charts:  python examples/donchian_breakout/run_multi_coin_viz.py")


if __name__ == "__main__":
    main()
