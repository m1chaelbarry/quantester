"""Real-data evaluation: BTC tranche pullback ladder on CCXT history.

Downloads the full BTC/USD daily history from a public exchange via the
CCXTDataHandler (default: Bitstamp, which paginates deep history; Binance and
Bybit geo-block this runner's location, and Kraken's OHLC endpoint ignores
distant `since` values), caches it under examples/data/, and answers one
question honestly: is the strategy profitable on real data, net of the
spec's conservative friction, with the 4.5% daily drawdown breaker armed?

Evaluation protocol:
- net vs gross (zero-cost) to isolate friction drag;
- buy-and-hold BTC benchmark over the identical window and cost model;
- per-calendar-year return table (regime robustness, not just the total);
- ATR-spacing sensitivity (informational only -- the spec fixes 1.5x; this
  is NOT a parameter selection sweep, so no PBO/DSR gate is claimed);
- truncation test (look-ahead leak detector);
- crypto-calendar annualization (365 periods/yr, not the equity 252).

NOTE: a single realized path cannot establish statistical significance. The
strategy has no closed-form vectorized twin, so MCPT fast-track validation
is unavailable; treat the results as descriptive evidence, not proof of edge.

Run from the repo root:  python examples/run_tranche_pullback_ccxt.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.portfolio.risk import DailyDrawdownBreaker
from quantester.strategy.examples import BuyAndHoldStrategy
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.validation.truncation import run_truncation_test

DATA_DIR = Path("examples/data")
OUTPUT_DIR = Path("examples/output")
CACHE = DATA_DIR / "BTCUSD_bitstamp_1d.csv"
SYMBOL = "BTC/USD"
INITIAL_CAPITAL = 25_000.0
PERIODS = 365  # crypto trades daily

# Spec friction: 2x (half-spread + taker fee). BTC/USD on a major venue:
# ~2 bps full spread, 4 bps per-side taker fee are representative-worse.
FRICTION = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
ZERO = CostModel(fixed_commission=0.0, per_share_commission=0.0,
                 spread_pct=0.0, slippage_vol_coef=0.0, impact_coef=0.0)


def load_or_fetch() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["datetime"], index_col="datetime")
    from quantester.data.ccxt_handler import CCXTDataHandler

    handler = CCXTDataHandler(SYMBOL, exchange="bitstamp", timeframe="1d",
                              start="2013-01-01", limit=1000)
    df = handler._data[SYMBOL]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index_label="datetime")
    return df


def run(df: pd.DataFrame, cost_model, breaker: bool = True,
        truncate_last: int | None = None, buy_and_hold: bool = False,
        **strategy_overrides) -> PortfolioManager:
    if truncate_last:
        df = df.iloc[:-truncate_last]
    handler = HistoricCSVDataHandler({SYMBOL: df})
    if buy_and_hold:
        strategy = BuyAndHoldStrategy(handler)
    else:
        strategy = TranchePullbackStrategy(handler, SYMBOL, **strategy_overrides)
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=PercentEquitySizer(1.0),
        drawdown_breaker=DailyDrawdownBreaker(0.045) if breaker else None,
    )
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(cost_model))
    engine.run_backtest()
    return portfolio


def metrics(equity: pd.Series, portfolio: PortfolioManager | None = None,
            label: str = "") -> dict:
    years = len(equity) / PERIODS
    row = {
        "label": label,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "sharpe_365": annualized_sharpe(equity, periods=PERIODS),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "calmar_365": calmar_ratio(equity, periods=PERIODS),
    }
    if portfolio is not None:
        pnls = [t["pnl"] for t in portfolio.trades]
        in_market = portfolio.positions_history.abs().sum(axis=1) > 0
        row.update(
            round_trips=len(pnls),
            win_rate=float(np.mean([p > 0 for p in pnls])) if pnls else np.nan,
            time_in_market=float(in_market.mean()),
            friction_paid=sum(f.total_cost for f in portfolio.fills),
            breaker_trips=(portfolio.drawdown_breaker.triggered_count
                           if portfolio.drawdown_breaker else 0),
        )
    return row


def report(row: dict) -> None:
    print(f"{row['label']:>26}: ret {row['total_return']:+9.1%}  "
          f"CAGR {row['cagr']:+7.2%}  sharpe {row['sharpe_365']:+.3f}  "
          f"maxDD {row['max_dd']:+.1%}  calmar {row['calmar_365']:+.3f}", end="")
    if "round_trips" in row:
        print(f"  | trades {row['round_trips']:>3}  win {row['win_rate']:.0%}  "
              f"in-mkt {row['time_in_market']:.0%}  "
              f"friction {row['friction_paid']:,.0f}  "
              f"breaker {row['breaker_trips']}")
    else:
        print()


def yearly_table(strat_eq: pd.Series, bh_eq: pd.Series) -> pd.DataFrame:
    def yearly(eq):
        years = sorted(set(eq.index.year))
        out = {}
        for y in years:
            seg = eq.loc[str(y)]
            out[y] = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        return pd.Series(out)

    table = pd.DataFrame({"strategy_net": yearly(strat_eq),
                          "buy_and_hold": yearly(bh_eq)})
    table["beat"] = table["strategy_net"] > table["buy_and_hold"]
    return table


def main():
    print("=" * 88)
    print("REAL-DATA TEST: tranche pullback ladder on BTC/USD daily (CCXT/Bitstamp)")
    print("=" * 88)
    df = load_or_fetch()
    print(f"Data: {len(df)} daily bars  {df.index[0].date()} -> "
          f"{df.index[-1].date()}  (close {df['close'].iloc[0]:,.2f} -> "
          f"{df['close'].iloc[-1]:,.2f})")

    portfolio = run(df, FRICTION, breaker=True)
    eq = portfolio.equity_curve

    print("\n-- headline (net of 2x spread+fee friction, breaker armed) --")
    report(metrics(eq, portfolio, "strategy net"))
    gross = run(df, ZERO)
    report(metrics(gross.equity_curve, gross, "strategy gross (zero cost)"))

    bh_portfolio = run(df, FRICTION, breaker=False, buy_and_hold=True)
    bh_eq = bh_portfolio.equity_curve
    # Align the benchmark to the strategy's tradable window (post-warmup).
    bh_aligned = bh_eq.loc[eq.index[0]:]
    report(metrics(bh_aligned, None, "buy & hold (same window)"))

    print("\n-- per-calendar-year net returns --")
    table = yearly_table(eq, bh_aligned)
    for year, r in table.iterrows():
        print(f"  {year}: strategy {r['strategy_net']:+8.1%}   "
              f"buy&hold {r['buy_and_hold']:+8.1%}   "
              f"{'beat' if r['beat'] else ''}")
    print(f"  years beating B&H: {int(table['beat'].sum())}/{len(table)}")

    print("\n-- ATR-spacing sensitivity (informational; NOT a selection sweep) --")
    for spacing in (0.75, 1.0, 1.25, 1.5):
        p = run(df, FRICTION, breaker=True, atr_spacing=spacing)
        report(metrics(p.equity_curve, p, f"spacing {spacing}x ATR"))

    print("\n-- leak check --")
    result = run_truncation_test(
        lambda n: run(df, FRICTION, breaker=True,
                      truncate_last=n).positions_history,
        n_truncated=30,
    )
    print(result)

    generate_tearsheet(
        eq, OUTPUT_DIR / "tranche_pullback_ccxt_tearsheet.png",
        title="Tranche pullback ladder — real BTC/USD daily (Bitstamp, net)",
        extra_stats={"CAGR_365": f"{metrics(eq)['cagr']:+.2%}",
                     "breaker_trips": str(metrics(eq, portfolio)["breaker_trips"])},
    )
    print(f"\nTearsheet: {OUTPUT_DIR / 'tranche_pullback_ccxt_tearsheet.png'}")


if __name__ == "__main__":
    main()
