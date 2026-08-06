"""End-to-end example: synthetic 3-symbol data -> event-driven backtest ->
tearsheet + truncation check + trials-registry DSR.

Run from the repo root:  python examples/run_ma_cross.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import (
    carver_cost_drag_sr,
    speed_limit_warning,
    summarize,
)
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv, write_csvs
from quantester.validation.truncation import run_truncation_test

DATA_DIR = Path("examples/data")
OUTPUT_DIR = Path("examples/output")
INITIAL_CAPITAL = 100_000.0


def build_data() -> dict:
    data = {
        "AAA": make_synthetic_ohlcv("AAA", seed=1, mu=0.10, sigma=0.22),
        "BBB": make_synthetic_ohlcv("BBB", seed=2, mu=0.05, sigma=0.18,
                                    s0=80.0, missing_every=47),
        "CCC": make_synthetic_ohlcv("CCC", seed=3, mu=-0.02, sigma=0.25, s0=120.0),
    }
    return write_csvs(data, DATA_DIR)


def run_backtest(csv_map: dict, symbol: str, fast: int, slow: int,
                 truncate_last: int | None = None):
    if truncate_last:
        trimmed = {
            s: pd_read(p).iloc[:-truncate_last] for s, p in csv_map.items()
        }
        handler = HistoricCSVDataHandler(trimmed)
    else:
        handler = HistoricCSVDataHandler(csv_map)
    strategy = MovingAverageCrossStrategy(handler, symbol, fast=fast, slow=slow,
                                          direction="both")
    portfolio = PortfolioManager(handler, INITIAL_CAPITAL,
                                 sizer=PercentEquitySizer(0.9))
    execution = SimulatedExecutionHandler(CostModel())
    engine = BacktestEngine(handler, strategy, portfolio, execution)
    engine.run_backtest()
    return portfolio


def pd_read(path):
    import pandas as pd

    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")


def main():
    print("=" * 72)
    print("Quantester example: MA-cross event-driven backtest")
    print("=" * 72)
    csv_map = build_data()

    symbol, fast, slow = "AAA", 10, 40

    # Trials registry: a small parameter sweep is logged so DSR sees real N/V.
    registry = TrialsRegistry()
    best_sharpe, best_params, best_portfolio = -np.inf, None, None
    for f, s in [(5, 20), (10, 40), (20, 60)]:
        portfolio = run_backtest(csv_map, symbol, f, s)
        stats = summarize(portfolio.equity_curve)
        rets = portfolio.equity_curve.pct_change().dropna()
        registry.log_trial(
            params={"symbol": symbol, "fast": f, "slow": s},
            sharpe=stats["sharpe"],
            mean=float(rets.mean()),
            std=float(rets.std()),
            skew=float(skew(rets)),
            kurt=float(kurtosis(rets, fisher=False)),
            n_obs=len(rets),
            run_id="ma_cross_sweep",
        )
        print(f"fast={f:>2} slow={s:>2}  sharpe={stats['sharpe']:+.3f}  "
              f"mdd={stats['max_drawdown']:+.2%}  calmar={stats['calmar']:+.3f}")
        if stats["sharpe"] > best_sharpe:
            best_sharpe, best_params, best_portfolio = stats["sharpe"], (f, s), portfolio

    best = registry.best_trial()
    dsr = dsr_from_registry(registry, sr_hat=best["sharpe"], n_obs=best["n_obs"],
                            skew=best["skew"], kurtosis=best["kurt"])
    print(f"\nBest trial: fast={best_params[0]} slow={best_params[1]} "
          f"sharpe={best_sharpe:+.3f}")
    print(f"DSR (N={registry.n_trials()} trials, registry-driven): {dsr:.3f}")

    drag = carver_cost_drag_sr(annual_turnover=4.0, standardized_cost_sr=0.01)
    warning = speed_limit_warning(drag)
    print(f"Carver cost drag: {drag:.3f} SR/yr"
          + (f"  WARNING: {warning}" if warning else " (within speed limit)"))

    stats = generate_tearsheet(
        best_portfolio.equity_curve,
        OUTPUT_DIR / "ma_cross_tearsheet.png",
        title=f"MA({best_params[0]}/{best_params[1]}) cross on {symbol}",
        extra_stats={"DSR": f"{dsr:.3f}", "cost_drag_SR": f"{drag:.3f}"},
    )
    print(f"\nTearsheet: {OUTPUT_DIR / 'ma_cross_tearsheet.png'}")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()
           if not isinstance(v, str)})

    result = run_truncation_test(
        lambda n: run_backtest(csv_map, symbol, *best_params, truncate_last=n)
        .positions_history,
        n_truncated=30,
    )
    print(f"\n{result}")
    registry.close()


if __name__ == "__main__":
    main()
