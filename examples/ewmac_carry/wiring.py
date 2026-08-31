"""Five-module wiring for EWMAC + crypto carry."""

from __future__ import annotations

import pandas as pd
from scipy.stats import kurtosis, skew

from quantester.analytics.performance import annualized_sharpe
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import perp_cost_scenario
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.fast_track import fast_backtest
from quantester.portfolio.portfolio import CarverVolTargetSizer, FixedUnitSizer, PortfolioManager
from quantester.strategy.ewmac_carry import EWMACCarryStrategy, combined_forecast_positions

from market import SYMBOL

INITIAL_CAPITAL = 100_000.0
SEED = 8

GRID = {
    "ewmac": ((16, 64), (32, 128)),
    "target_vol": (0.10, 0.15, 0.20),
    "beta": (0.10, 0.15, 0.20),
    "w_carry": (0.30, 0.40, 0.50),
}


def units_for(df: pd.DataFrame) -> float:
    px = float(df["close"].iloc[0])
    return max(INITIAL_CAPITAL * 0.95 / px, 1e-8)


def moments(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    return {
        "sharpe_daily": float(rets.mean() / rets.std()) if float(rets.std()) > 0 else 0.0,
        "sharpe_ann": annualized_sharpe(equity),
        "skew": float(skew(rets, bias=False)) if len(rets) > 3 else 0.0,
        "kurt": float(kurtosis(rets, fisher=True, bias=False)) if len(rets) > 3 else 0.0,
        "n_obs": int(len(rets)),
    }


def daily_pnl(equity: pd.Series) -> pd.Series:
    return equity.diff().fillna(0.0)


def twin_target(df: pd.DataFrame, fast: int, slow: int, w_carry: float) -> pd.Series:
    return combined_forecast_positions(
        df, fast=fast, slow=slow,
        trend_weight=1.0 - w_carry, carry_weight=w_carry,
    )


def fast_equity(df: pd.DataFrame, fast: int, slow: int, w_carry: float) -> pd.Series:
    target = twin_target(df, fast, slow, w_carry)
    return fast_backtest(
        df, target, perp_cost_scenario("BASE"),
        initial_capital=INITIAL_CAPITAL, units=units_for(df),
    ).equity


def fast_sharpe(df: pd.DataFrame, fast: int, slow: int, w_carry: float) -> float:
    return annualized_sharpe(fast_equity(df, fast, slow, w_carry))


def event_backtest(
    df: pd.DataFrame,
    *,
    fast: int = 16,
    slow: int = 64,
    w_carry: float = 0.40,
    target_vol: float = 0.15,
    inertia_beta: float = 0.15,
    sizer=None,
    costs=None,
    book_funding_settlements: bool = True,
) -> PortfolioManager:
    handler = HistoricCSVDataHandler({SYMBOL: df})
    strat = EWMACCarryStrategy(
        handler, SYMBOL, fast=fast, slow=slow,
        trend_weight=1.0 - w_carry, carry_weight=w_carry,
    )
    if sizer is None:
        sizer = CarverVolTargetSizer(
            target_vol=target_vol, inertia_beta=inertia_beta,
        )
    port = PortfolioManager(handler, INITIAL_CAPITAL, sizer=sizer)
    engine = BacktestEngine(
        handler, strat, port,
        SimulatedExecutionHandler(costs or perp_cost_scenario("BASE")),
        book_funding_settlements=book_funding_settlements,
    )
    engine.run_backtest()
    return port
