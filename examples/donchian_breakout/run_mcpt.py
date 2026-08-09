"""Backtest + MCPT for the hourly BTC Donchian breakout.

The strategy has no closed-form vectorized twin (path-dependent protective
stop), so MCPT drives full event-loop re-runs on Protocol II OHLC permutations
(intra-bar geometry shuffled jointly; inter-bar gaps shuffled independently).
That is the correct protocol when the model reads highs/lows (Donchian, ATR,
ADX).

Default window: last ``--bars`` hourly prints (2500 ≈ 3.4 months) so a few
hundred event-driven reps finish in minutes. Raise ``--reps`` toward 1,000 for
production Masters gates; the autocorrelation gate is printed first — if it
flags serial correlation, treat iid-style conclusions with care (block
bootstrap remains available via the parameter-study scripts).

Usage:
  python examples/donchian_breakout/run_mcpt.py
  python examples/donchian_breakout/run_mcpt.py --bars 2500 --reps 200 --workers 4
  python examples/donchian_breakout/run_mcpt.py --full-history
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantester.analytics.performance import annualized_sharpe
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.permutation import (
    intra_inter_bar_permutation,
    masters_p_value,
    trend_bias_skill,
)
from quantester.portfolio.portfolio import (
    FractionalRiskSizer,
    PercentEquitySizer,
    PortfolioManager,
)
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.strategy.examples import BuyAndHoldStrategy

from _shared import (
    FRICTION,
    INITIAL_CAPITAL,
    OUTPUT_DIR,
    PERIODS,
    SYMBOL,
    ZERO,
    load_or_fetch,
    metrics,
    report,
)

# Set in main before the fork pool so workers see the MCPT window via CoW.
_DF: pd.DataFrame | None = None


def backtest(df: pd.DataFrame, cost_model=FRICTION,
             buy_and_hold: bool = False) -> PortfolioManager:
    handler = HistoricCSVDataHandler({SYMBOL: df})
    if buy_and_hold:
        strategy = BuyAndHoldStrategy(handler)
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


def _mcpt_worker(seed: int) -> dict:
    """One Protocol II permutation → event-driven Sharpe / return / B&H."""
    assert _DF is not None
    rng = np.random.default_rng(seed)
    perm = intra_inter_bar_permutation(_DF, rng)
    perm["volume"] = _DF["volume"].to_numpy()
    eq = backtest(perm).equity_curve
    return {
        "sharpe": float(annualized_sharpe(eq, periods=PERIODS)),
        "ret": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "bh": float(perm["close"].iloc[-1] / perm["close"].iloc[0] - 1.0),
    }


def run_window_backtests(window: pd.DataFrame) -> tuple:
    net = backtest(window, FRICTION)
    gross = backtest(window, ZERO)
    bh = backtest(window, FRICTION, buy_and_hold=True)
    return net, gross, bh


def main():
    global _DF
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=2500,
                        help="Trailing hourly bars used for MCPT window")
    parser.add_argument("--reps", type=int, default=200,
                        help="MCPT replications (Masters checklist: >= 1000)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--full-history", action="store_true",
                        help="Also backtest the entire cached hourly file")
    parser.add_argument("--skip-mcpt", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("Donchian breakout — hourly BTC backtest + Protocol II MCPT")
    print("=" * 72)

    full = load_or_fetch()
    print(f"Cache: {len(full)} bars  {full.index[0]} → {full.index[-1]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.full_history:
        print("\n-- full-history backtest --")
        t0 = time.perf_counter()
        net = backtest(full, FRICTION)
        gross = backtest(full, ZERO)
        bh = backtest(full, FRICTION, buy_and_hold=True)
        print(f"  elapsed {time.perf_counter() - t0:.0f}s")
        report(metrics(net.equity_curve, net, "Donchian net"))
        report(metrics(gross.equity_curve, gross, "Donchian gross"))
        report(metrics(bh.equity_curve, bh, "Buy & hold"))
        generate_tearsheet(
            net.equity_curve,
            OUTPUT_DIR / "donchian_breakout_hourly_full_tearsheet.png",
            title="Donchian breakout — BTC/USD 1h full cache (net)",
            extra_stats={"trades": str(len(net.trades))},
        )

    window = full.iloc[-args.bars:].copy()
    print(f"\n-- MCPT window: last {len(window)} bars  "
          f"{window.index[0]} → {window.index[-1]} --")
    t0 = time.perf_counter()
    net_w, gross_w, bh_w = run_window_backtests(window)
    print(f"  elapsed {time.perf_counter() - t0:.0f}s")
    report(metrics(net_w.equity_curve, net_w, "Donchian net (window)"))
    report(metrics(gross_w.equity_curve, gross_w, "Donchian gross (window)"))
    report(metrics(bh_w.equity_curve, bh_w, "Buy & hold (window)"))

    generate_tearsheet(
        net_w.equity_curve,
        OUTPUT_DIR / "donchian_breakout_mcpt_window_tearsheet.png",
        title=f"Donchian breakout — last {len(window)} hourly bars (net)",
        extra_stats={"trades": str(len(net_w.trades))},
    )

    log_rets = np.log(window["close"] / window["close"].shift(1)).dropna()
    gate = autocorrelation_gate(log_rets)
    print("\n-- autocorrelation gate --")
    print(f"  recommended={gate.recommended_method}  "
          f"runs_p={gate.runs_p:.4f}  LB_p={gate.ljung_box_p:.4f}  "
          f"serial_correlation={gate.serial_correlation}")

    if args.skip_mcpt:
        return

    print(f"\n-- Protocol II MCPT ({args.reps} reps, {args.workers} workers) --")
    print("  (event-driven re-runs; no vectorized twin for this strategy)")
    _DF = window
    original_sharpe = float(
        annualized_sharpe(net_w.equity_curve, periods=PERIODS)
    )
    original_ret = float(
        net_w.equity_curve.iloc[-1] / net_w.equity_curve.iloc[0] - 1.0
    )
    bh_orig = float(window["close"].iloc[-1] / window["close"].iloc[0] - 1.0)

    seeds = [args.seed + 1 + i for i in range(args.reps - 1)]
    t0 = time.perf_counter()
    if args.workers <= 1:
        rows = [_mcpt_worker(s) for s in seeds]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            rows = pool.map(_mcpt_worker, seeds)
    elapsed = time.perf_counter() - t0

    perm_sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    perm_rets = np.array([r["ret"] for r in rows], dtype=float)
    perm_bh = np.array([r["bh"] for r in rows], dtype=float)
    p_value = masters_p_value(original_sharpe, perm_sharpes)
    partition = trend_bias_skill(
        r_orig=original_ret,
        b_orig=bh_orig,
        r_perm=float(np.mean(perm_rets)),
        b_perm=float(np.mean(perm_bh)),
    )

    verdict = "SIGNIFICANT p<0.05" if p_value < 0.05 else "not significant"
    print(f"  elapsed {elapsed:.0f}s")
    print(f"  original Sharpe={original_sharpe:+.3f}  return={original_ret:+.2%}")
    print(f"  perm Sharpe: median={np.median(perm_sharpes):+.3f}  "
          f"p5={np.percentile(perm_sharpes, 5):+.3f}  "
          f"p95={np.percentile(perm_sharpes, 95):+.3f}")
    print(f"  MCPT p-value ({args.reps} reps): {p_value:.4f}  ({verdict})")
    print(f"  Masters partition: trend={partition['trend']:+.3f}  "
          f"bias={partition['training_bias']:+.3f}  "
          f"skill={partition['skill']:+.3f}")
    print(f"  P(perm Sharpe >= original)="
          f"{(perm_sharpes >= original_sharpe).mean():.3f}  "
          f"P(perm Sharpe > 0)={(perm_sharpes > 0).mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(net_w.equity_curve.index, net_w.equity_curve.values, lw=1.2)
    axes[0].set_title("Equity (MCPT window, net of friction)")
    axes[0].set_ylabel("Equity")
    axes[1].hist(perm_sharpes, bins=24, alpha=0.75, label="permuted")
    axes[1].axvline(original_sharpe, color="r", lw=1.5,
                    label=f"original {original_sharpe:+.3f}")
    axes[1].axvline(0.0, color="k", lw=0.8)
    axes[1].set_title(f"Protocol II MCPT  p={p_value:.3f} ({verdict})")
    axes[1].set_xlabel(f"Sharpe (periods={PERIODS})")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = OUTPUT_DIR / "donchian_breakout_mcpt.png"
    fig.savefig(out, dpi=120)
    print(f"\nFigure: {out}")

    summary = OUTPUT_DIR / "donchian_breakout_mcpt_summary.txt"
    summary.write_text(
        "\n".join([
            f"window_bars={len(window)}",
            f"window_start={window.index[0]}",
            f"window_end={window.index[-1]}",
            f"original_sharpe={original_sharpe:.6f}",
            f"original_return={original_ret:.6f}",
            f"bh_return={bh_orig:.6f}",
            f"trades={len(net_w.trades)}",
            f"n_reps={args.reps}",
            f"p_value={p_value:.6f}",
            f"verdict={verdict}",
            f"trend={partition['trend']:.6f}",
            f"training_bias={partition['training_bias']:.6f}",
            f"skill={partition['skill']:.6f}",
            f"perm_sharpe_median={float(np.median(perm_sharpes)):.6f}",
            f"autocorr_method={gate.recommended_method}",
            f"serial_correlation={gate.serial_correlation}",
        ]) + "\n"
    )
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
