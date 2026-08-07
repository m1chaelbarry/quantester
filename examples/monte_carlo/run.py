"""Monte Carlo validation example (MC Report section 6 checklist).

Backtest -> trade resampling PDF -> MCPT p-value with Trend/Bias/Skill
partition -> double-bootstrap drawdown bound -> O-U OTR sweep -> diagnostics.

Run from the repo root:  python examples/monte_carlo/run.py
Replication counts are set small for a quick demo; raise N_REPS / N_OUTER /
N_INNER to the checklist values (1,000 / 10,000 / 1,000) for production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.drawdown import double_bootstrap_dd_bound
from quantester.montecarlo.fast_track import fast_backtest
from quantester.montecarlo.permutation import (
    intra_inter_bar_permutation,
    masters_p_value,
    multi_market_permutation,
    permute_log_changes,
    trend_bias_skill,
)
from quantester.montecarlo.synthetic import estimate_ou_params, generate_ou_paths, otr_sweep
from quantester.montecarlo.trade_resampling import (
    ehlers_randomized_equity,
    empirical_resample,
)
from quantester.portfolio.portfolio import FixedUnitSizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv

N_REPS = 200          # checklist: >= 1,000
N_OUTER = 300         # checklist: 10,000
N_INNER = 100         # checklist: 1,000
SEED = 7


def main():
    print("=" * 72)
    print("Quantester example: Monte Carlo validation suite")
    print("=" * 72)
    rng = np.random.default_rng(SEED)

    df = make_synthetic_ohlcv("AAA", seed=1, mu=0.10, sigma=0.22)
    handler = HistoricCSVDataHandler({"AAA": df})
    strategy = MovingAverageCrossStrategy(handler, "AAA", fast=10, slow=40)
    portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(500))
    engine = BacktestEngine(handler, strategy, portfolio, SimulatedExecutionHandler(CostModel()))
    engine.run_backtest()
    rets = portfolio.equity_curve.pct_change().dropna().to_numpy()
    print(f"Backtest: {len(rets)} daily returns, "
          f"total return {portfolio.equity_curve.iloc[-1] / 100_000 - 1:+.2%}")

    print("\n-- 1. Trade-level resampling ---------------------------------")
    hat = empirical_resample(rets, horizon=260, n_sims=5_000, seed=SEED)
    q = hat.quantiles()
    print(f"Hat resampling (260d x 5,000): terminal return "
          f"P5={q[0.05]:+.2%}  median={q[0.5]:+.2%}  P95={q[0.95]:+.2%}")
    trades = pd.DataFrame(portfolio.trades)
    if not trades.empty:
        wins = trades[trades["pnl"] > 0]["pnl"]
        losses = trades[trades["pnl"] < 0]["pnl"]
        win_rate = len(wins) / len(trades)
        pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.inf
        eh = ehlers_randomized_equity(win_rate, pf, abs(losses.mean()) if len(losses) else 1.0,
                                      n_trades=len(trades), n_sims=5_000, e0=100_000.0,
                                      seed=SEED)
        print(f"Ehlers parametric ({len(trades)} trades, p={win_rate:.2f}, PF={pf:.2f}): "
              f"median terminal {np.median(eh[:, -1]) / 100_000 - 1:+.2%}")

    print("\n-- 2. MCPT permutation testing --------------------------------")
    close = df["close"]

    def optimizer(series: pd.Series) -> float:
        ohlc = df.copy()
        ohlc["close"] = series.reindex(ohlc.index).ffill().bfill()
        best = -np.inf
        for fast in (5, 10, 20):
            target = MovingAverageCrossStrategy(
                None, "AAA", fast=fast, slow=40).vectorized_signals({"AAA": ohlc})["AAA"]
            result = fast_backtest(ohlc, target, CostModel())
            best = max(best, result.sharpe)
        return best

    original = optimizer(close)
    permuted = np.array([optimizer(permute_log_changes(close, rng)) for _ in range(N_REPS - 1)])
    p_value = masters_p_value(original, permuted)
    partition = trend_bias_skill(
        r_orig=original, b_orig=float(close.pct_change().mean() * 252),
        r_perm=float(permuted.mean()),
        b_perm=float(np.mean([s.pct_change().mean() * 252 for s in
                              [permute_log_changes(close, rng) for _ in range(20)]])),
    )
    print(f"MCPT p-value ({N_REPS} reps): {p_value:.4f} "
          f"({'SIGNIFICANT p<0.05' if p_value < 0.05 else 'not significant'})")
    print(f"Return partition: trend={partition['trend']:+.3f} "
          f"bias={partition['training_bias']:+.3f} skill={partition['skill']:+.3f}")

    mm = multi_market_permutation(
        pd.DataFrame({"A": close, "B": make_synthetic_ohlcv('B', seed=2)['close']}), rng)
    print(f"Protocol I multi-market permutation OK ({mm.shape})")
    pp = intra_inter_bar_permutation(df, rng)
    valid = bool(((pp["high"] >= pp[["open", "close"]].max(axis=1) - 1e-9)
                  & (pp["low"] <= pp[["open", "close"]].min(axis=1) + 1e-9)).all())
    print(f"Protocol II intra/inter-bar permutation: OHLC physically valid = {valid}")

    print("\n-- 3. Drawdown double bootstrap -------------------------------")
    dd = double_bootstrap_dd_bound(rets, dd_conf=0.95, bound_conf=0.70,
                                   n_outer=N_OUTER, n_inner=N_INNER, seed=SEED)
    print(f"Double-bootstrap DD bound (DD_conf=0.95, Bound_conf=0.70): {dd.bound:.4f}")

    print("\n-- 4. O-U synthetic paths + OTR sweep ---------------------------")
    ou = estimate_ou_params(close)
    paths = generate_ou_paths(ou, p0=float(close.iloc[-1]), n_steps=120,
                              n_paths=2_000, seed=SEED)
    grid = otr_sweep(paths, stop_losses=[0.05, 0.10], take_profits=[0.10, 0.20])
    best = grid.loc[grid["mean_pnl"].idxmax()]
    print(f"OU params: theta={ou.theta:.4f} mu={ou.mu:.2f} sigma={ou.sigma:.3f}")
    print(f"OTR best: stop={best['stop_loss']:.0%} tp={best['take_profit']:.0%} "
          f"mean_pnl={best['mean_pnl']:+.3f} win_rate={best['win_rate']:.2f}")

    print("\n-- 5. Autocorrelation diagnostics gate --------------------------")
    report = autocorrelation_gate(rets)
    print(f"runs p={report.runs_p:.3f}  Ljung-Box p={report.ljung_box_p:.3f}  "
          f"-> {'serial correlation: use ' if report.serial_correlation else 'no serial correlation: '}"
          f"{report.recommended_method}")


if __name__ == "__main__":
    main()
