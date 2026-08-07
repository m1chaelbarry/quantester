"""Higher-resolution BTC parameter study (4h / 1h) with PBO/DSR + bootstrap MC.

Ports the daily tranche ladder to intraday bars with calendar-equivalent
windows (SMA200 days → SMA200·bars_per_day, etc.) and daily operational
cadence (`reanchor_every=bpd`, `cooldown_bars=bpd-1`) so fills and
Kaufman stops resolve on every bar while the ladder only refreshes once
per calendar day and cannot re-arm same-day after an exit.

Naive every-bar re-anchoring on 1h wiped the account (−100%, 18k trades);
daily cadence cuts that to ~1.7k trades but the calendar-scaled SPEC
(1.5/5.0/5) is still deeply unprofitable on both 1h and 4h — this study
asks whether a wider spacing/stop grid recovers an edge, under the same
governance gates as the daily study.

Usage:
  python examples/tranche_pullback/run_parameter_study_intraday.py           # 4h default
  python examples/tranche_pullback/run_parameter_study_intraday.py --tf 1h   # hourly
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "examples" / "data"
OUTPUT_DIR = HERE / "output"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import annualized_sharpe, max_drawdown
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.synthetic import bootstrap_ohlcv
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.portfolio.risk import DailyDrawdownBreaker
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.validation.pbo import PBO_GATE, pbo_cscv

SYMBOL = "BTC/USD"
INITIAL_CAPITAL = 25_000.0
FRICTION = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
WORKERS = 4

# Wider than the daily grid: intraday sampling hits the same ATR distance
# more often, so cost-survivable configs sit at larger spacing/stop.
GRID_SPACINGS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
GRID_STOPS = (5.0, 6.0, 7.5, 10.0, 12.0, 15.0, 18.0)
GRID_EXIT_DAYS = (3, 5, 8)
SPEC_DAYS = {"atr_spacing": 1.5, "stop_atr_mult": 5.0, "exit_days": 5}

TF_CONFIG = {
    "4h": {"cache": DATA_DIR / "BTCUSD_bitstamp_4h.csv", "bpd": 6,
           "ccxt_tf": "4h", "n_mc": 48, "mean_block_days": 20},
    "1h": {"cache": DATA_DIR / "BTCUSD_bitstamp_1h.csv", "bpd": 24,
           "ccxt_tf": "1h", "n_mc": 24, "mean_block_days": 20},
}

_DF = None
_BPD = None
_PERIODS = None
_MEAN_BLOCK = None
_BASE_PARAMS = None


def load_or_fetch(tf: str) -> pd.DataFrame:
    cfg = TF_CONFIG[tf]
    if cfg["cache"].exists():
        return pd.read_csv(cfg["cache"], parse_dates=["datetime"],
                           index_col="datetime")
    from quantester.data.ccxt_handler import CCXTDataHandler

    print(f"Fetching BTC/USD {tf} from Bitstamp ...")
    handler = CCXTDataHandler(SYMBOL, exchange="bitstamp",
                              timeframe=cfg["ccxt_tf"], start="2017-01-01",
                              limit=1000)
    df = handler._data[SYMBOL]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["cache"], index_label="datetime")
    return df


def calendar_params(spacing, stop, exit_days, bpd: int) -> dict:
    return {
        "regime_window": 200 * bpd,
        "peak_window": 20 * bpd,
        "atr_window": 14 * bpd,
        "atr_spacing": float(spacing),
        "exit_window": int(exit_days) * bpd,
        "stop_atr_mult": float(stop),
        "reanchor_every": bpd,
        "cooldown_bars": max(bpd - 1, 0),
    }


def grid(bpd: int):
    for spacing in GRID_SPACINGS:
        for stop in GRID_STOPS:
            if stop <= 3.0 * spacing:
                continue
            for exit_days in GRID_EXIT_DAYS:
                yield {
                    "atr_spacing": spacing,
                    "stop_atr_mult": stop,
                    "exit_days": exit_days,
                    **calendar_params(spacing, stop, exit_days, bpd),
                }


def backtest(params: dict, df: pd.DataFrame) -> PortfolioManager:
    strat_keys = ("regime_window", "peak_window", "atr_window", "atr_spacing",
                  "exit_window", "stop_atr_mult", "reanchor_every",
                  "cooldown_bars")
    strat_params = {k: params[k] for k in strat_keys}
    handler = HistoricCSVDataHandler({SYMBOL: df})
    strategy = TranchePullbackStrategy(handler, SYMBOL, **strat_params)
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

    years = len(equity) / _PERIODS
    return {
        "params": {
            "atr_spacing": params["atr_spacing"],
            "stop_atr_mult": params["stop_atr_mult"],
            "exit_days": params["exit_days"],
        },
        "full_params": params,
        "sharpe": annualized_sharpe(equity, periods=_PERIODS),
        "sharpe_bar": float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0,
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "trades": len(portfolio.trades),
        "skew": float(skew(rets)),
        "kurt": float(kurtosis(rets, fisher=False)),
        "n_obs": len(rets),
        "bar_pnl": equity.diff().dropna().to_numpy(),
    }


def _mc_worker(seed: int) -> dict:
    frame = bootstrap_ohlcv(_DF, mean_block=_MEAN_BLOCK, seed=seed)
    portfolio = backtest(_BASE_PARAMS, frame)
    equity = portfolio.equity_curve
    years = len(equity) / _PERIODS
    warmup = _BASE_PARAMS["regime_window"]
    bh = frame["close"].iloc[warmup:].pct_change().dropna()
    bh_sharpe = (
        float(bh.mean() / bh.std() * np.sqrt(_PERIODS)) if bh.std() > 0 else 0.0
    )
    return {
        "sharpe": annualized_sharpe(equity, periods=_PERIODS),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "trades": len(portfolio.trades),
        "bh_sharpe": bh_sharpe,
    }


def main():
    global _DF, _BPD, _PERIODS, _MEAN_BLOCK, _BASE_PARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", choices=sorted(TF_CONFIG), default="4h")
    args = parser.parse_args()
    cfg = TF_CONFIG[args.tf]
    _BPD = cfg["bpd"]
    _PERIODS = 365 * _BPD
    _MEAN_BLOCK = cfg["mean_block_days"] * _BPD
    _DF = load_or_fetch(args.tf)
    _BASE_PARAMS = calendar_params(
        SPEC_DAYS["atr_spacing"], SPEC_DAYS["stop_atr_mult"],
        SPEC_DAYS["exit_days"], _BPD,
    )

    print("=" * 88)
    print(f"INTRADAY STUDY: tranche ladder on BTC/USD {args.tf} "
          f"(bpd={_BPD}, daily cadence)")
    print("=" * 88)
    print(f"Data: {len(_DF)} bars  {_DF.index[0]} -> {_DF.index[-1]}  "
          f"close {_DF['close'].iloc[0]:,.2f} -> {_DF['close'].iloc[-1]:,.2f}")

    # Spec headline
    t0 = time.perf_counter()
    spec_port = backtest(_BASE_PARAMS, _DF)
    eq = spec_port.equity_curve
    print(f"\n-- SPEC (1.5/5.0/5d) on {args.tf}, daily cadence --")
    print(f"  ret {eq.iloc[-1]/eq.iloc[0]-1:+.1%}  "
          f"CAGR {(eq.iloc[-1]/eq.iloc[0])**(1.0/(len(eq)/_PERIODS))-1:+.2%}  "
          f"sharpe {annualized_sharpe(eq, periods=_PERIODS):+.3f}  "
          f"maxDD {max_drawdown(eq)['max_drawdown']:+.1%}  "
          f"trades {len(spec_port.trades)}  "
          f"breaker {spec_port.drawdown_breaker.triggered_count}  "
          f"({time.perf_counter()-t0:.0f}s)")

    trials = list(grid(_BPD))
    print(f"\nGrid: {len(trials)} valid trials; workers={WORKERS}")
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS) as pool:
        results = pool.map(_trial_worker, trials)
    print(f"Grid complete in {time.perf_counter()-t0:.0f}s")

    registry = TrialsRegistry()
    for r in results:
        registry.log_trial(params=r["params"], sharpe=r["sharpe_bar"],
                           skew=r["skew"], kurt=r["kurt"], n_obs=r["n_obs"],
                           run_id=f"intraday_{args.tf}")
    sharpes = np.array([r["sharpe"] for r in results])
    order = np.argsort(sharpes)[::-1]
    print(f"\nSharpe across trials: median {np.median(sharpes):+.3f}  "
          f"IQR [{np.percentile(sharpes,25):+.3f}, "
          f"{np.percentile(sharpes,75):+.3f}]  max {sharpes.max():+.3f}  "
          f"P(sharpe>0)={(sharpes>0).mean():.1%}")
    print("Top 5:")
    for i in order[:5]:
        p = results[i]["params"]
        print(f"  sharpe {results[i]['sharpe']:+.3f}  "
              f"cagr {results[i]['cagr']:+.2%}  maxDD {results[i]['max_dd']:+.1%}  "
              f"trades {results[i]['trades']:>4}  <- spacing {p['atr_spacing']}, "
              f"stop {p['stop_atr_mult']}, exit {p['exit_days']}d")
    spec_idx = next(
        i for i, r in enumerate(results)
        if r["params"] == {"atr_spacing": 1.5, "stop_atr_mult": 5.0, "exit_days": 5}
    )
    print(f"Spec rank: {int((sharpes > sharpes[spec_idx]).sum())+1}/{len(results)} "
          f"(sharpe {sharpes[spec_idx]:+.3f})")

    pnl = pd.DataFrame({f"t{i}": r["bar_pnl"] for i, r in enumerate(results)})
    # Align lengths (warmup makes them equal already, but be safe)
    min_len = min(map(len, pnl.values.T))
    pnl = pnl.iloc[-min_len:]
    pbo = pbo_cscv(pnl, n_blocks=16)
    best = registry.best_trial()
    dsr = dsr_from_registry(registry, sr_hat=best["sharpe"], n_obs=best["n_obs"],
                            skew=best["skew"], kurtosis=best["kurt"])
    print(f"\nPBO N={pbo.n_trials}: {pbo.pbo:.3f}  "
          f"gate <{PBO_GATE}: {'PASS' if pbo.passes_gate else 'FAIL'}")
    print(f"DSR (N={registry.n_trials()}, bar SR units): {dsr:.3f}")
    registry.close()

    # MC on BEST config if it beats 0, else on SPEC (honest about failure)
    best_i = int(order[0])
    mc_params = results[best_i]["full_params"]
    mc_label = results[best_i]["params"]
    _BASE_PARAMS = mc_params  # for _mc_worker
    log_rets = np.log(_DF["close"] / _DF["close"].shift(1)).dropna()
    gate = autocorrelation_gate(log_rets)
    print(f"\nAutocorr gate: {gate.recommended_method} "
          f"(runs p={gate.runs_p:.4f}, LB p={gate.ljung_box_p:.4f})")
    print(f"MC: {cfg['n_mc']} block-bootstrap paths on BEST "
          f"{mc_label}, mean_block={_MEAN_BLOCK} bars")
    t0 = time.perf_counter()
    with ctx.Pool(WORKERS) as pool:
        mc = pool.map(_mc_worker,
                      [10_000 + i for i in range(cfg["n_mc"])])
    print(f"MC complete in {time.perf_counter()-t0:.0f}s")
    mc_s = np.array([r["sharpe"] for r in mc])
    mc_bh = np.array([r["bh_sharpe"] for r in mc])
    realized = results[best_i]["sharpe"]
    print(f"MC Sharpe: median {np.median(mc_s):+.3f}  "
          f"p5 {np.percentile(mc_s,5):+.3f}  p95 {np.percentile(mc_s,95):+.3f}")
    print(f"P(no edge)={(mc_s<=0).mean():.3f}  "
          f"P(beat B&H Sharpe)={(mc_s>mc_bh).mean():.3f}  "
          f"realized at MC pct {(mc_s<realized).mean():.1%}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # heatmap: spacing vs stop at exit_days=5
    sub = [(r["params"]["atr_spacing"], r["params"]["stop_atr_mult"],
            r["sharpe"]) for r in results if r["params"]["exit_days"] == 5]
    sp = sorted(set(s for s, _, _ in sub))
    st = sorted(set(x for _, x, _ in sub))
    mat = np.full((len(sp), len(st)), np.nan)
    for s, x, v in sub:
        mat[sp.index(s), st.index(x)] = v
    im = axes[0].imshow(mat, aspect="auto", cmap="RdYlGn", origin="lower")
    axes[0].set_xticks(range(len(st)), st)
    axes[0].set_yticks(range(len(sp)), sp)
    axes[0].set_xlabel("stop ATR mult")
    axes[0].set_ylabel("ATR spacing")
    axes[0].set_title(f"{args.tf} Sharpe, exit=5d (blank=invalid)")
    for i in range(len(sp)):
        for j in range(len(st)):
            if np.isfinite(mat[i, j]):
                axes[0].text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                             fontsize=7)
    fig.colorbar(im, ax=axes[0])
    axes[1].hist(mc_s, bins=16, alpha=0.7, label="MC paths")
    axes[1].axvline(0.0, color="k", lw=1)
    axes[1].axvline(realized, color="r", lw=1.5,
                    label=f"best {realized:+.3f}")
    axes[1].set_title(f"Block-bootstrap MC  P(no edge)={(mc_s<=0).mean():.3f}")
    axes[1].set_xlabel("Sharpe (annualized)")
    axes[1].legend()
    fig.tight_layout()
    out = OUTPUT_DIR / f"parameter_study_{args.tf}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nFigure: {out}")
    print(f"\nGOVERNANCE ({args.tf}): PBO {pbo.pbo:.3f} "
          f"({'PASS' if pbo.passes_gate else 'FAIL'})  DSR {dsr:.3f}  "
          f"MC P(no edge) {(mc_s<=0).mean():.3f}")


if __name__ == "__main__":
    main()
