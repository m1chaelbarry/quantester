"""Wide parameter study + governance gates for the BTC tranche pullback ladder.

Extends the single-point real-data evaluation (run_tranche_pullback_ccxt.py)
into a proper study, per the repo's validation rules and the Gemini-notebook
cross-reference:

1. GRID: atr_spacing x stop_atr_mult x exit_window (54 valid trials; the
   stop must sit wider than the deepest tranche, 3x spacing). Each trial is a
   full event-engine run on real BTC/USD daily data (cached CCXT/Bitstamp),
   net of the 2x friction model with the 4.5% daily breaker armed. Workers
   run in forked processes; trial records come back over IPC and are logged
   single-threaded to the TrialsRegistry (its parallel-safe pattern).
2. CSCV/PBO (Bailey-de Prado, notebook-verified algorithm in validation/pbo.py)
   over the full trial matrix — the gate is PBO < 0.10.
3. DSR from the registry (daily SR units so sqrt(T-1) scaling is consistent).
4. Block-bootstrap MC harness: the spec config is re-run through the event
   engine on stationary-block-bootstrapped OHLC paths (Politis-Romano;
   Masters-endorsed protocol per the cross-ref) after the autocorrelation
   gate. iid resampling is invalid for BTC returns — the gate confirms it.
   The MC null scrambles the long-run regime ordering while preserving
   within-block dependence, answering: "does the strategy's mechanics-driven
   edge survive markets with BTC-like short-run structure but shuffled
   regimes?" — NOT "will 2013-2026 repeat".

Run from the repo root:  python examples/tranche_pullback/run_parameter_study.py
(~4 min on 4 cores; set WORKERS=1 to force sequential)
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import annualized_sharpe, max_drawdown
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.synthetic import bootstrap_ohlcv
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.portfolio.risk import DailyDrawdownBreaker
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.validation.pbo import PBO_GATE, pbo_cscv
from run_ccxt import (
    FRICTION,
    INITIAL_CAPITAL,
    PERIODS,
    SYMBOL,
    load_or_fetch,
)

GRID_SPACINGS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
GRID_STOPS = (4.5, 5.0, 6.0, 7.5)
GRID_EXITS = (3, 5, 8)
SPEC = {"atr_spacing": 1.5, "stop_atr_mult": 5.0, "exit_window": 5}

N_MC_REPS = 64
MEAN_BLOCK = 20          # ~one trading month of preserved local structure
MC_SEED_BASE = 10_000
WORKERS = 4
WARMUP_BARS = 200        # strategy warmup; B&H benchmark aligns to it

_DF = None  # set before forking workers (fork inherits the parent's memory)


def grid():
    for spacing in GRID_SPACINGS:
        for stop in GRID_STOPS:
            if stop <= 3.0 * spacing:
                continue  # stop must sit wider than the deepest tranche
            for exit_window in GRID_EXITS:
                yield {"atr_spacing": spacing, "stop_atr_mult": stop,
                       "exit_window": exit_window}


def backtest(params: dict, df: pd.DataFrame) -> PortfolioManager:
    handler = HistoricCSVDataHandler({SYMBOL: df})
    strategy = TranchePullbackStrategy(handler, SYMBOL, **params)
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=PercentEquitySizer(1.0),
        drawdown_breaker=DailyDrawdownBreaker(0.045),
    )
    BacktestEngine(handler, strategy, portfolio,
                   SimulatedExecutionHandler(FRICTION)).run_backtest()
    return portfolio


def _trial_worker(params: dict) -> dict:
    portfolio = backtest(params, _DF)
    equity = portfolio.equity_curve
    rets = equity.pct_change().dropna()
    from scipy.stats import kurtosis, skew

    years = len(equity) / PERIODS
    return {
        "params": params,
        "sharpe365": annualized_sharpe(equity, periods_per_year=PERIODS),
        "sharpe_daily": float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0,
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "trades": len(portfolio.trades),
        "skew": float(skew(rets)),
        "kurt": float(kurtosis(rets, fisher=False)),
        "n_obs": len(rets),
        "daily_pnl": equity.diff().dropna().to_numpy(),
    }


def _mc_worker(seed: int) -> dict:
    frame = bootstrap_ohlcv(_DF, mean_block=MEAN_BLOCK, seed=seed)
    portfolio = backtest(SPEC, frame)
    equity = portfolio.equity_curve
    years = len(equity) / PERIODS
    bh_rets = frame["close"].iloc[WARMUP_BARS:].pct_change().dropna()
    bh_sharpe = (
        float(bh_rets.mean() / bh_rets.std() * np.sqrt(PERIODS))
        if bh_rets.std() > 0 else 0.0
    )
    return {
        "seed": seed,
        "sharpe365": annualized_sharpe(equity, periods_per_year=PERIODS),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "trades": len(portfolio.trades),
        "bh_sharpe365": bh_sharpe,
    }


def main():
    global _DF
    print("=" * 88)
    print("PARAMETER STUDY + GOVERNANCE: tranche pullback ladder on real BTC/USD")
    print("=" * 88)
    _DF = load_or_fetch()
    print(f"Data: {len(_DF)} daily bars  {_DF.index[0].date()} -> "
          f"{_DF.index[-1].date()}")
    trials = list(grid())
    print(f"Grid: {len(trials)} valid trials "
          f"(spacing x stop x exit, stop > 3x spacing); workers={WORKERS}")

    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS) as pool:
        results = pool.map(_trial_worker, trials)
    print(f"Grid complete in {time.perf_counter() - t0:.0f}s")

    registry = TrialsRegistry()
    for r in results:
        registry.log_trial(params=r["params"], sharpe=r["sharpe_daily"],
                           mean=None, std=None, skew=r["skew"], kurt=r["kurt"],
                           n_obs=r["n_obs"], run_id="wide_grid")
    sharpes = np.array([r["sharpe365"] for r in results])
    order = np.argsort(sharpes)[::-1]
    print(f"\nSharpe(365) across trials: median {np.median(sharpes):+.3f}  "
          f"IQR [{np.percentile(sharpes, 25):+.3f}, "
          f"{np.percentile(sharpes, 75):+.3f}]  max {sharpes.max():+.3f}")
    print("Top 5 trials:")
    for i in order[:5]:
        p = results[i]["params"]
        print(f"  sharpe {results[i]['sharpe365']:+.3f}  "
              f"cagr {results[i]['cagr']:+.2%}  maxDD {results[i]['max_dd']:+.1%}  "
              f"trades {results[i]['trades']:>3}  <- spacing "
              f"{p['atr_spacing']}, stop {p['stop_atr_mult']}, exit {p['exit_window']}")
    spec_rank = next(
        i for i, r in enumerate(results) if r["params"] == SPEC
    )
    spec_pos = int((sharpes > sharpes[spec_rank]).sum()) + 1
    print(f"Spec config (1.5/5.0/5): sharpe {results[spec_rank]['sharpe365']:+.3f} "
          f"— rank {spec_pos}/{len(results)} (in-sample)")

    # ------------------------------------------------- CSCV / PBO gate
    pnl = pd.DataFrame(
        {f" trial{i}": r["daily_pnl"] for i, r in enumerate(results)}
    )
    pbo = pbo_cscv(pnl, n_blocks=16)
    print(f"\nPBO over N={pbo.n_trials} trials "
          f"({pbo.n_combinations} CSCV combinations): {pbo.pbo:.3f}")
    print(f"Gate PBO < {PBO_GATE}: "
          f"{'PASS' if pbo.passes_gate else 'FAIL — grid selection is overfit-prone'}")

    # ------------------------------------------------------------- DSR
    best = registry.best_trial()
    dsr = dsr_from_registry(registry, sr_hat=best["sharpe"], n_obs=best["n_obs"],
                            skew=best["skew"], kurtosis=best["kurt"])
    print(f"DSR (N={registry.n_trials()} trials, daily SR units): {dsr:.3f} "
          f"(probability the selected config has true skill)")
    registry.close()

    # --------------------------------------------- block-bootstrap MC harness
    log_rets = np.log(_DF["close"] / _DF["close"].shift(1)).dropna()
    gate = autocorrelation_gate(log_rets)
    print(f"\nAutocorrelation gate: runs p={gate.runs_p:.4f}  "
          f"LB p={gate.ljung_box_p:.4f}  ->  {gate.recommended_method}")
    print(f"MC harness: {N_MC_REPS} stationary-block-bootstrap paths "
          f"(mean block {MEAN_BLOCK} bars), event engine re-run per path, "
          f"spec config net of friction + breaker")
    t0 = time.perf_counter()
    with ctx.Pool(WORKERS) as pool:
        mc = pool.map(_mc_worker, [MC_SEED_BASE + i for i in range(N_MC_REPS)])
    print(f"MC complete in {time.perf_counter() - t0:.0f}s")

    mc_sharpe = np.array([r["sharpe365"] for r in mc])
    mc_cagr = np.array([r["cagr"] for r in mc])
    mc_dd = np.array([r["max_dd"] for r in mc])
    mc_bh = np.array([r["bh_sharpe365"] for r in mc])
    realized = results[spec_rank]["sharpe365"]
    p_no_edge = float((mc_sharpe <= 0).mean())
    pbeat_bh = float((mc_sharpe > mc_bh).mean())
    pct = np.percentile(mc_sharpe, [5, 25, 50, 75, 95])
    print(f"\nMC strategy Sharpe(365): p5 {pct[0]:+.3f}  p25 {pct[1]:+.3f}  "
          f"median {pct[2]:+.3f}  p75 {pct[3]:+.3f}  p95 {pct[4]:+.3f}")
    print(f"MC median CAGR {np.median(mc_cagr):+.2%}  "
          f"median maxDD {np.median(mc_dd):+.1%}  "
          f"median trades {int(np.median([r['trades'] for r in mc]))}")
    print(f"P(Sharpe <= 0 | regime-scrambled BTC): {p_no_edge:.3f} "
          f"(MC no-edge probability; lower = stronger)")
    print(f"P(strategy Sharpe > same-path B&H Sharpe): {pbeat_bh:.3f}")
    pct_of_realized = float((mc_sharpe < realized).mean())
    print(f"Realized Sharpe {realized:+.3f} sits at MC percentile "
          f"{pct_of_realized:.1%} of regime-scrambled outcomes")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sub = [(r["params"]["atr_spacing"], r["params"]["stop_atr_mult"],
            r["sharpe365"]) for r in results if r["params"]["exit_window"] == 5]
    sp = sorted(set(s for s, _, _ in sub))
    st = sorted(set(x for _, x, _ in sub))
    grid_sharpe = np.full((len(sp), len(st)), np.nan)
    for s, x, v in sub:
        grid_sharpe[sp.index(s), st.index(x)] = v
    im = axes[0].imshow(grid_sharpe, aspect="auto", cmap="RdYlGn",
                        origin="lower")
    axes[0].set_xticks(range(len(st)), st)
    axes[0].set_yticks(range(len(sp)), sp)
    axes[0].set_xlabel("stop ATR mult")
    axes[0].set_ylabel("ATR spacing")
    axes[0].set_title("Grid Sharpe(365), exit_window=5 (blank = invalid)")
    for i in range(len(sp)):
        for j in range(len(st)):
            if np.isfinite(grid_sharpe[i, j]):
                axes[0].text(j, i, f"{grid_sharpe[i, j]:.2f}", ha="center",
                             va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0])
    axes[1].hist(mc_sharpe, bins=20, alpha=0.7, label="MC bootstrap paths")
    axes[1].axvline(0.0, color="k", lw=1)
    axes[1].axvline(realized, color="r", lw=1.5,
                    label=f"realized {realized:+.3f}")
    axes[1].set_title(f"Block-bootstrap MC Sharpe (P(no edge)={p_no_edge:.3f})")
    axes[1].set_xlabel("Sharpe (365)")
    axes[1].legend()
    fig.tight_layout()
    out = OUTPUT_DIR / "parameter_study_ccxt.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nFigure: {out}")

    print("\nGOVERNANCE VERDICT")
    print(f"  PBO gate (<{PBO_GATE}): {'PASS' if pbo.passes_gate else 'FAIL'} "
          f"({pbo.pbo:.3f});  DSR {dsr:.3f};  MC P(no edge) {p_no_edge:.3f};  "
          f"P(beat B&H risk-adj) {pbeat_bh:.3f}")


if __name__ == "__main__":
    main()
