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

Run from the repo root:  python examples/tranche_pullback/run_ccxt.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "examples" / "data"
OUTPUT_DIR = HERE / "output"

import numpy as np
import pandas as pd

from scipy.stats import kurtosis as skurt
from scipy.stats import skew as sskew

from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import (
    annualized_sharpe,
    calmar_ratio,
    carver_cost_drag_sr,
    max_drawdown,
    speed_limit_warning,
)
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel, CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.portfolio.risk import DailyDrawdownBreaker
from quantester.portfolio.sizing import optimal_f
from quantester.strategy.examples import BuyAndHoldStrategy
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.validation.pbo import pbo_cscv
from quantester.validation.truncation import run_truncation_test

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
    df = handler.source_ohlcv(SYMBOL)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index_label="datetime")
    return df


def run(df: pd.DataFrame, cost_model, breaker: bool = True,
        truncate_last: int | None = None, buy_and_hold: bool = False,
        cash_yield_rate: float = 0.0,
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
        cash_yield_rate=cash_yield_rate,  # Kaufman: half the 3M T-bill rate
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

    # Idle-cash yield (notebook-verified: Kaufman accrues half the 3M T-bill
    # rate on flat capital; Carver requires including rf on undeployed cash).
    # 2.0% T-bill assumption -> 1.0% effective on idle cash. Assumption, not data.
    yielded = run(df, FRICTION, breaker=True, cash_yield_rate=0.02)
    print("\n-- idle-cash yield variant (Kaufman half of assumed 2% T-bill) --")
    report(metrics(yielded.equity_curve, yielded, "strategy net + cash yield"))

    # Carver cost audit (notebook-verified): standardized round-trip cost
    # 2C/(16*ICV), turnover drag, and the speed limits (0.13 SR = 1/3 of the
    # 0.40 staunch-systems SR; 0.08 SR stricter repo gate).
    print("\n-- Carver cost audit --")
    daily_sigma = float(df["close"].pct_change().std())
    block_value = float(df["close"].mean())          # 1-BTC block, mean price
    icv = daily_sigma * block_value                  # instrument currency vol
    c_one_way = block_value * FRICTION.friction_multiplier * (
        FRICTION.spread_pct / 2 + FRICTION.fee_rate
    )
    standardized = 2 * c_one_way / (16 * icv)        # SR units per round-trip
    n_years = len(df) / PERIODS
    turnover = len(portfolio.trades) / n_years       # round-trips per year
    drag = carver_cost_drag_sr(turnover, standardized)
    print(f"  daily sigma {daily_sigma:.2%}  ICV/BTC {icv:,.0f}  "
          f"standardized cost {standardized:.5f} SR/round-trip")
    print(f"  turnover {turnover:.1f} round-trips/yr -> cost drag "
          f"{drag:.3f} SR/yr (limits: 0.08 strict, 0.13 = 1/3 of 0.40 SR)")
    warning = speed_limit_warning(drag)
    print(f"  speed limit (0.08): {'EXCEEDED — ' + warning if warning else 'within'}"
          f" | one-third rule (0.13): {'EXCEEDED' if drag > 0.13 else 'within'}")

    # Vince optimal-f on the 1-unit P&L stream (notebook-verified formulas in
    # portfolio/sizing.py; worst loss gap-stressed 1.5x per Vince's caveat).
    print("\n-- Vince optimal-f audit (1-unit basis) --")
    unit_pnl = np.array([t["pnl"] / t["qty"] for t in portfolio.trades])
    worst = float(unit_pnl.min())
    f_star = optimal_f(unit_pnl)                     # gap-stressed by default
    f_plain = optimal_f(unit_pnl, gap_stress=1.0)    # unstressed, for contrast
    stressed_worst = worst * 1.5
    q_dollars = abs(stressed_worst) / f_star if f_star > 0 else np.inf
    k_units = INITIAL_CAPITAL / q_dollars if np.isfinite(q_dollars) else 0.0
    print(f"  worst 1-BTC loss {worst:,.0f} (gap-stressed {stressed_worst:,.0f})")
    print(f"  f* = {f_star:.3f} (unstressed {f_plain:.3f})  ->  Q = "
          f"{q_dollars:,.0f} USD per BTC  ->  K = {k_units:.3f} BTC on "
          f"{INITIAL_CAPITAL:,.0f} equity")
    cap_frac = k_units * block_value / INITIAL_CAPITAL
    print(f"  implied notional cap ~{cap_frac:.0%} of equity at mean price; "
          f"the spec deploys up to 100% (25/35/40 stacking)")

    print("\n-- spacing grid + overfitting audits (CSCV/PBO gate, DSR) --")
    spacings = (0.75, 1.0, 1.25, 1.5)
    registry = TrialsRegistry()
    curves = {}
    for spacing in spacings:
        p = run(df, FRICTION, breaker=True, atr_spacing=spacing)
        eq_s = p.equity_curve
        curves[spacing] = eq_s
        rets = eq_s.pct_change().dropna()
        sr_daily = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0
        registry.log_trial(
            params={"atr_spacing": spacing}, sharpe=sr_daily,
            mean=float(rets.mean()), std=float(rets.std()),
            skew=float(sskew(rets)), kurt=float(skurt(rets, fisher=False)),
            n_obs=len(rets), run_id="spacing_grid",
        )
        report(metrics(eq_s, p, f"spacing {spacing}x ATR"))
    pnl_grid = pd.DataFrame(
        {s: curves[s].diff() for s in spacings}
    ).dropna()
    pbo = pbo_cscv(pnl_grid, n_blocks=16)
    best = registry.best_trial()
    dsr = dsr_from_registry(registry, sr_hat=best["sharpe"], n_obs=best["n_obs"],
                            skew=best["skew"], kurtosis=best["kurt"])
    print(f"  PBO over the {len(spacings)}-point spacing grid: {pbo.pbo:.3f} "
          f"({pbo.n_combinations} CSCV combos) — gate < 0.10: "
          f"{'PASS' if pbo.passes_gate else 'FAIL (overfitting likely)'}")
    print(f"  DSR (N={registry.n_trials()} trials, daily SR units): {dsr:.3f} "
          f"— probability the best config has true skill after selection bias")
    registry.close()

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
