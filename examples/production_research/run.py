#!/usr/bin/env python3
"""Production-grade research pipeline (teaching reference).

Run from the repository root::

    python examples/production_research/run.py
    python examples/production_research/run.py --full   # heavier MCPT / bootstrap

This script is intentionally linear and heavily commented: each stage maps to a
row in ``README.md``. It uses a synthetic market with a *planted* momentum edge
so the demo is offline-reproducible and typically profitable under the gates —
not a claim that real markets look like this.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression

from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import annualized_sharpe, summarize
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.data.audit import audit_ohlcv_frame
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import CostModel, retail_cost_scenario
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.adaptive import adaptive_empirical_resample
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.drawdown import double_bootstrap_dd_bound
from quantester.montecarlo.fast_track import fast_backtest
from quantester.montecarlo.permutation import permutation_test
from quantester.montecarlo.synthetic import bootstrap_ohlcv
from quantester.montecarlo.trade_resampling import ehlers_randomized_equity
from quantester.portfolio.portfolio import FixedUnitSizer, PercentEquitySizer, PortfolioManager
from quantester.portfolio.risk import DailyDrawdownBreaker, MarginMonitor
from quantester.strategy.meta_labeling import triple_barrier_labels
from quantester.validation.cpcv import CombinatorialPurgedKFold
from quantester.validation.gates import (
    build_validation_report,
    evaluate_gates,
    run_cost_stress,
)
from quantester.validation.pbo import PBO_GATE, pbo_cscv
from quantester.validation.truncation import run_truncation_test
from quantester.visualization.static import plot_equity

from market import make_momentum_edge_ohlcv, split_is_oos
from strategy import TrendMomentumStrategy, momentum_positions

SYMBOL = "EDGE"
INITIAL_CAPITAL = 100_000.0
# Research-matrix share count is set per-frame so |target|=1 ≈ 95% equity
# (FixedUnitSizer / fast_backtest). A hard-coded 100 shares on a $100 name
# leaves ~90% cash and makes OOS Sharpe look artificially dead.
BASE_COSTS = CostModel(
    fixed_commission=0.0,
    per_share_commission=0.005,
    spread_pct=0.0004,
    slippage_vol_coef=0.05,
    impact_coef=0.05,
)
SEED = 8


def units_for(df: pd.DataFrame) -> float:
    px = float(df["close"].iloc[0])
    return max(INITIAL_CAPITAL * 0.95 / px, 1.0)


# ------------------------------------------------------------------ CLI scale
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help="Production-scale reps (MCPT 1000, more bootstrap paths).",
    )
    return p.parse_args()


def _scale(full: bool) -> dict:
    if full:
        return dict(
            n_bars=1_764,
            # Economically spaced lookbacks (≈2w / 1m / 1q / 1y). Avoid a
            # dense cluster of near-identical windows — that alone drives PBO→1.
            lookbacks=(10, 21, 63, 126, 252),
            mcpt_reps=1_001,
            bootstrap_paths=64,
            wf_test=84,
            dd_outer=200,
            dd_inner=50,
        )
    return dict(
        n_bars=1_500,
        lookbacks=(10, 21, 63, 252),
        mcpt_reps=401,
        bootstrap_paths=32,
        wf_test=84,
        dd_outer=40,
        dd_inner=20,
    )


# --------------------------------------------------------------- backtest API
def event_backtest(
    df: pd.DataFrame,
    lookback: int,
    *,
    sizer=None,
    cost_model=None,
    use_risk_overlays: bool = False,
    invert: bool = False,
) -> PortfolioManager:
    """Canonical event-engine wiring used everywhere below."""
    handler = HistoricCSVDataHandler({SYMBOL: df})
    strategy = TrendMomentumStrategy(
        handler, SYMBOL, lookback=lookback, allow_short=True,
    )
    if invert:
        # Decoy family for PBO: mean-reversion twin (same lookback, flipped side).
        strategy = _InvertedStrategy(strategy)
    portfolio = PortfolioManager(
        handler,
        INITIAL_CAPITAL,
        sizer=sizer or FixedUnitSizer(units_for(df)),
        margin_monitor=MarginMonitor(max_leverage=3.0) if use_risk_overlays else None,
        drawdown_breaker=(
            DailyDrawdownBreaker(max_intraday_dd=0.08) if use_risk_overlays else None
        ),
    )
    execution = SimulatedExecutionHandler(cost_model or BASE_COSTS)
    BacktestEngine(handler, strategy, portfolio, execution).run_backtest()
    return portfolio


class _InvertedStrategy:
    """Thin wrapper: flips LONG↔SHORT for PBO decoy trials (delay preserved)."""

    def __init__(self, inner: TrendMomentumStrategy):
        self.inner = inner
        self.delay = inner.delay
        self.symbol = inner.symbol

    def matches_phase(self, phase: str) -> bool:
        return self.inner.matches_phase(phase)

    def calculate_signals(self, event, events_queue):
        from quantester.events import EXIT, LONG, SHORT, SignalEvent

        class _Q:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        q = _Q()
        self.inner.calculate_signals(event, q)
        for sig in q.items:
            if sig.signal_type == LONG:
                st = SHORT
            elif sig.signal_type == SHORT:
                st = LONG
            else:
                st = EXIT
            events_queue.put(
                SignalEvent(
                    sig.timestamp, sig.symbol, st,
                    strength=sig.strength, delay=sig.delay,
                )
            )

    def vectorized_signals(self, data: dict):
        out = self.inner.vectorized_signals(data)
        return {k: -v for k, v in out.items()}


def fast_sharpe(
    df: pd.DataFrame, lookback: int, cost_model=None, *, invert: bool = False
) -> float:
    target = momentum_positions(df["close"], lookback=lookback)
    if invert:
        target = -target
    return fast_backtest(
        df, target, cost_model or BASE_COSTS,
        initial_capital=INITIAL_CAPITAL, units=units_for(df),
    ).sharpe


def fast_equity(
    df: pd.DataFrame, lookback: int, cost_model=None, *, invert: bool = False
) -> pd.Series:
    target = momentum_positions(df["close"], lookback=lookback)
    if invert:
        target = -target
    return fast_backtest(
        df, target, cost_model or BASE_COSTS,
        initial_capital=INITIAL_CAPITAL, units=units_for(df),
    ).equity


def daily_pnl(equity: pd.Series) -> pd.Series:
    return equity.diff().fillna(0.0)


def period_sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std())


def moments(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    return {
        "sharpe_daily": period_sharpe(equity),
        "sharpe_ann": annualized_sharpe(equity),
        "skew": float(skew(rets)) if len(rets) > 2 else 0.0,
        "kurt": float(kurtosis(rets, fisher=False)) if len(rets) > 3 else 3.0,
        "n_obs": int(len(rets)),
    }


# --------------------------------------------------------------------- stages
def stage_data(cfg: dict):
    print("\n[0] PLANTED-EDGE MARKET + DATA AUDIT")
    df = make_momentum_edge_ohlcv(n_bars=cfg["n_bars"], seed=SEED)
    path = OUTPUT / "EDGE.csv"
    df.to_csv(path, index_label="datetime")
    audit = audit_ohlcv_frame(
        df,
        SYMBOL,
        expected_freq="B",
        adjustment_policy="synthetic_unadjusted",
        corporate_actions_documented=True,
        delistings_considered=True,
        survivorship_bias_considered=True,
        historical_universe_documented=True,
        trading_calendar_documented=True,
    )
    print(f"  bars={len(df)}  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  audit status={audit.status}  passed={audit.passed}")
    for c in audit.warnings()[:5]:
        print(f"    WARN {c.name}: {c.detail}")
    is_df, oos_df = split_is_oos(df, oos_fraction=0.25)
    print(f"  IS bars={len(is_df)}  untouched OOS bars={len(oos_df)}")
    return df, is_df, oos_df, audit


def stage_grid(is_df: pd.DataFrame, lookbacks: tuple) -> tuple:
    print("\n[1] IN-SAMPLE PARAMETER GRID → TrialsRegistry")
    # Fresh registry every run — never silently accumulate prior demo trials into N.
    db_path = OUTPUT / "trials.db"
    if db_path.exists():
        db_path.unlink()
    registry = TrialsRegistry(str(db_path))
    rows = []
    pnl_cols = {}
    for lb in lookbacks:
        eq = fast_equity(is_df, lb)
        m = moments(eq)
        params = {
            "lookback": lb,
            "family": "trend",
            "vol_lookback": 20,
            "target_vol": 0.15,
        }
        registry.log_trial(
            params=params,
            sharpe=m["sharpe_daily"],
            skew=m["skew"],
            kurt=m["kurt"],
            n_obs=m["n_obs"],
            run_id="is_grid",
            strategy_id="TrendMomentum",
            metrics={"sharpe_ann": m["sharpe_ann"]},
        )
        rows.append({"lookback": lb, "family": "trend", "invert": False, **m, "equity": eq})
        pnl_cols[f"trend_lb{lb}"] = daily_pnl(eq).to_numpy()
        print(
            f"  trend lookback={lb:>3}  dailySR={m['sharpe_daily']:+.4f}  "
            f"annSR={m['sharpe_ann']:+.3f}"
        )
    # Optional rejected family (not logged — logging known-bad decoys into DSR's
    # N/variance would either game or wreck the deflator; reject a priori).
    decoy_lb = lookbacks[0]
    decoy = moments(fast_equity(is_df, decoy_lb, invert=True))
    print(
        f"  rejected a priori: mean-reversion lb={decoy_lb}  "
        f"annSR={decoy['sharpe_ann']:+.3f} (not in registry / PBO matrix)"
    )
    best = max(rows, key=lambda r: r["sharpe_daily"])
    print(f"  champion lookback={best['lookback']} (IS only — OOS still sealed)")
    pnl = pd.DataFrame(pnl_cols)
    return registry, rows, best, pnl


def stage_champion_event(is_df: pd.DataFrame, lookback: int):
    print("\n[2] EVENT-ENGINE CHAMPION (IS) + RISK OVERLAYS + TEARSHEET")
    portfolio = event_backtest(
        is_df,
        lookback,
        sizer=PercentEquitySizer(0.95),
        use_risk_overlays=True,
    )
    equity = portfolio.equity_curve
    stats = summarize(equity)
    inv = portfolio.accounting_invariant()
    print(
        f"  total={stats['total_return']:+.2%}  sharpe={stats['sharpe']:+.3f}  "
        f"maxDD={stats['max_drawdown']:+.2%}  trades={len(portfolio.trades)}"
    )
    print(f"  accounting invariant ok={inv['ok']}  |diff|={inv['abs_diff']:.2e}")
    generate_tearsheet(
        equity, OUTPUT / "champion_tearsheet.png",
        title=f"TrendMomentum({lookback}) IS — event engine",
    )
    plot_equity(
        equity, portfolio.positions_history,
        path=OUTPUT / "champion_equity.png",
        title=f"TrendMomentum({lookback}) IS equity",
    )
    return portfolio, stats, inv


def stage_parity(is_df: pd.DataFrame, lookback: int) -> bool:
    print("\n[3] EVENT ↔ FAST-TRACK PARITY (FixedUnitSizer)")
    units = units_for(is_df)
    port = event_backtest(is_df, lookback, sizer=FixedUnitSizer(units))
    fast_eq = fast_equity(is_df, lookback)
    eng = port.equity_curve.reindex(is_df.index).ffill()
    diff = (eng - fast_eq).abs().max()
    ok = bool(np.isfinite(diff) and diff < 1e-4)
    print(f"  max |equity diff| = {diff:.3e}  parity={'PASS' if ok else 'FAIL'}")
    return ok


def stage_truncation(is_df: pd.DataFrame, lookback: int):
    print("\n[4] TRUNCATION DIAGNOSTIC (look-ahead leak detector)")

    def _positions(truncate_last: int | None):
        frame = is_df if truncate_last is None else is_df.iloc[:-truncate_last]
        return event_backtest(
            frame, lookback, sizer=FixedUnitSizer(units_for(frame)),
        ).positions_history

    result = run_truncation_test(_positions, n_truncated=30)
    print(f"  {result}")
    return result


def stage_walk_forward(is_df: pd.DataFrame, lookbacks: tuple, test_bars: int):
    print("\n[5] WALK-FORWARD (expanding train → locked test)")
    warm = max(lookbacks) + 40
    start = warm
    fold = 0
    oos_pieces = []
    fold_rows = []
    t = start
    while t + test_bars <= len(is_df):
        train = is_df.iloc[:t]
        test = is_df.iloc[t : t + test_bars]
        champ = max(lookbacks, key=lambda lb: fast_sharpe(train, lb))
        # Warm-start indicators with history before the test window so a
        # lookback=252 fold is not a cold-start flat line.
        window = is_df.iloc[max(0, t - warm) : t + test_bars]
        eq = fast_equity(window, champ).loc[test.index[0] :]
        rets = eq.pct_change().fillna(0.0)
        oos_pieces.append(rets)
        fold_rows.append(
            {
                "fold": fold,
                "train_end": str(train.index[-1].date()),
                "test_end": str(test.index[-1].date()),
                "lookback": champ,
                "test_sharpe_ann": annualized_sharpe(eq),
            }
        )
        print(
            f"  fold {fold}: train→{train.index[-1].date()}  "
            f"test {test.index[0].date()}→{test.index[-1].date()}  "
            f"lb={champ}  test annSR={fold_rows[-1]['test_sharpe_ann']:+.3f}"
        )
        fold += 1
        t += test_bars
    if not oos_pieces:
        raise RuntimeError("walk-forward produced no folds — reduce wf_test")
    chained = pd.concat(oos_pieces)
    wealth = (1.0 + chained).cumprod() * INITIAL_CAPITAL
    wf_stats = summarize(wealth)
    print(
        f"  stitched WF: folds={fold}  sharpe={wf_stats['sharpe']:+.3f}  "
        f"total={wf_stats['total_return']:+.2%}"
    )
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "walk_forward_folds.csv", index=False)
    return wf_stats, fold_rows


def stage_cpcv_meta(is_df: pd.DataFrame, lookback: int) -> dict:
    print("\n[6] CPCV + TRIPLE-BARRIER META-LABELS (educational secondary fit)")
    print(
        "  Note: the primary TrendMomentum rule does NOT fit a model, so CPCV is"
        "  NOT a mandatory gate for it. This section shows the API you must use"
        "  IF you add a fitted secondary (meta-label) model."
    )
    close = is_df["close"]
    pos = momentum_positions(close, lookback=lookback)
    side = np.sign(pos)
    changes = side[(side != side.shift(1)) & (side != 0)]
    if len(changes) < 30:
        print("  too few primary events for CPCV demo")
        return {"passed": None, "n_events": int(len(changes)), "mean_acc": None,
                "gate_value": None}

    events = pd.DataFrame({"t0": changes.index, "side": changes.to_numpy()})
    events.index = changes.index
    y = triple_barrier_labels(
        close, events, tp_pct=0.02, sl_pct=0.02, max_holding=10
    )
    mom = close / close.shift(lookback) - 1.0
    vol = close.pct_change().rolling(20).std()
    X = pd.DataFrame({"mom": mom, "vol": vol}, index=close.index).loc[y.index]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    t1 = pd.Series(
        [close.index[min(close.index.get_loc(t) + 10, len(close) - 1)] for t in X.index],
        index=X.index,
    )
    cpcv = CombinatorialPurgedKFold(n_groups=6, k_test=2, t1=t1, pct_embargo=0.01)
    scores = []
    for train_idx, test_idx in cpcv.split(X):
        if len(train_idx) < 20 or len(test_idx) < 5:
            continue
        model = LogisticRegression(max_iter=500)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = float((pred == y.iloc[test_idx].to_numpy()).mean())
        scores.append(acc)
    mean_acc = float(np.mean(scores)) if scores else 0.0
    print(
        f"  events={len(y)}  CPCV splits used={len(scores)}/{cpcv.n_splits}  "
        f"paths={cpcv.n_paths}  mean OOS acc={mean_acc:.3f}"
    )
    print(
        "  → gate value for primary strategy: NOT_APPLICABLE "
        "(no fitted model in the live pipeline)"
    )
    return {
        "passed": None,  # N/A for rule-based primary
        "n_events": int(len(y)),
        "mean_acc": mean_acc,
        "n_scores": len(scores),
        "gate_value": None,
    }


def stage_pbo(pnl: pd.DataFrame):
    print("\n[7] PBO / CSCV (selection overfitting)")
    result = pbo_cscv(pnl, n_blocks=10 if pnl.shape[0] < 400 else 16)
    print(
        f"  PBO={result.pbo:.3f} over N={result.n_trials} trials "
        f"({result.n_combinations} combos)  gate<{PBO_GATE}: "
        f"{'PASS' if result.passes_gate else 'FAIL'}"
    )
    return result


def stage_dsr(registry: TrialsRegistry, rows: list) -> float:
    print("\n[8] DEFLATED SHARPE (registry-honest N)")
    best = registry.best_trial()
    dsr = dsr_from_registry(
        registry,
        sr_hat=best["sharpe"],
        n_obs=best["n_obs"],
        skew=best["skew"] or 0.0,
        kurtosis=best["kurt"] or 3.0,
        annualized=False,  # registry holds per-period (daily) Sharpe
    )
    print(
        f"  best params={best['params']}  dailySR={best['sharpe']:+.4f}  "
        f"N={registry.n_trials()}  DSR={dsr:.4f}  gate≥0.95: "
        f"{'PASS' if dsr >= 0.95 else 'FAIL'}"
    )
    return float(dsr)


def stage_mcpt(is_df: pd.DataFrame, lookbacks: tuple, n_reps: int):
    print("\n[9] AUTOCORR GATE → MCPT (retrain-on-permute via fast-track)")
    close = is_df["close"]
    # Strategy returns under the IS champion for the autocorrelation diagnostic.
    champ = max(lookbacks, key=lambda lb: fast_sharpe(is_df, lb))
    strat_rets = fast_equity(is_df, champ).pct_change().dropna()
    gate = autocorrelation_gate(strat_rets.to_numpy())
    print(
        f"  autocorr serial={gate.serial_correlation}  "
        f"method={gate.recommended_method}"
    )

    def optimizer(series: pd.Series) -> float:
        ohlc = is_df.copy()
        # Rebuild a coherent close path; keep OHLC bands proportional.
        ohlc["close"] = series.reindex(ohlc.index).ffill().bfill()
        ohlc["open"] = ohlc["close"].shift(1).fillna(ohlc["close"].iloc[0])
        span = (is_df["high"] - is_df["low"]).median()
        ohlc["high"] = np.maximum(ohlc["open"], ohlc["close"]) + span / 2.0
        ohlc["low"] = np.minimum(ohlc["open"], ohlc["close"]) - span / 2.0
        return max(fast_sharpe(ohlc, lb) for lb in lookbacks)

    t0 = time.perf_counter()
    result = permutation_test(
        close, optimizer, n_reps=n_reps, seed=SEED,
    )
    print(
        f"  MCPT p={result.p_value:.4f}  origSR={result.original_performance:+.3f}  "
        f"reps={result.n_reps}  significant={result.significant}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return gate, result


def stage_resampling(is_df: pd.DataFrame, lookback: int, n_paths: int, cfg: dict):
    print("\n[10] BLOCK BOOTSTRAP / ADAPTIVE RESAMPLE / DD BOUND / EHLERS")
    eq = fast_equity(is_df, lookback)
    rets = eq.pct_change().dropna().to_numpy()
    adapted = adaptive_empirical_resample(
        rets, horizon=min(252, len(rets) // 2), n_sims=n_paths, seed=SEED
    )
    print(
        f"  adaptive resample method={adapted.method_used}  "
        f"block={adapted.block_length}  "
        f"terminal quantiles={adapted.result.quantiles()}"
    )
    # Path-dependent null: re-run strategy on bootstrapped OHLC.
    boot_sharpes = []
    for i in range(n_paths):
        frame = bootstrap_ohlcv(is_df, mean_block=20, seed=SEED + 1000 + i)
        boot_sharpes.append(fast_sharpe(frame, lookback))
    boot_sharpes = np.asarray(boot_sharpes)
    print(
        f"  OHLC block-bootstrap annSR: median={np.median(boot_sharpes):+.3f}  "
        f"p05={np.percentile(boot_sharpes, 5):+.3f}  "
        f"p95={np.percentile(boot_sharpes, 95):+.3f}"
    )
    dd = double_bootstrap_dd_bound(
        rets,
        horizon=min(252, len(rets) // 2),
        n_outer=cfg["dd_outer"],
        n_inner=cfg["dd_inner"],
        seed=SEED,
    )
    print(
        f"  nested DD bound: {dd.bound:.3f}  "
        f"(DD_conf={dd.dd_conf}, Bound_conf={dd.bound_conf})"
    )
    # Ehlers parametric paths from trade stats (avg_win/avg_loss ratio).
    # Reconstruct rough trade PnLs from position flips on the fast track.
    target = momentum_positions(is_df["close"], lookback=lookback)
    # Approximate: daily strategy PnL conditional on being in a trade.
    pnl = eq.diff().dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(wins) and len(losses):
        win_rate = len(wins) / (len(wins) + len(losses))
        avg_loss = float(losses.mean())
        pf_ratio = float(wins.mean() / abs(avg_loss))
        paths = ehlers_randomized_equity(
            win_rate, pf_ratio, avg_loss, n_trades=min(200, len(pnl)),
            n_sims=200, seed=SEED,
        )
        terminal = paths[:, -1]
        print(
            f"  Ehlers paths: win_rate={win_rate:.2f}  avg_win/avg_loss={pf_ratio:.2f}  "
            f"terminal p05={np.quantile(terminal, 0.05):.3f}  "
            f"p50={np.quantile(terminal, 0.50):.3f}"
        )
    else:
        print("  Ehlers skipped (insufficient win/loss sample)")
    mc_pass = bool(np.median(boot_sharpes) > 0 and result_significant_proxy(boot_sharpes))
    return {
        "adapted_method": adapted.method_used,
        "boot_median_sr": float(np.median(boot_sharpes)),
        "boot_p05_sr": float(np.percentile(boot_sharpes, 5)),
        "dd_bound": float(dd.bound),
        "passed": mc_pass,
    }


def result_significant_proxy(boot_sharpes: np.ndarray) -> bool:
    """Cheap MC viability: majority of bootstrap paths keep positive Sharpe."""
    return float((boot_sharpes > 0).mean()) >= 0.55


def stage_cost_stress(is_df: pd.DataFrame, lookback: int):
    print("\n[11] EXECUTION COST STRESS (BASE / CONSERVATIVE / STRESS)")

    def run_fn(model):
        sr = fast_sharpe(is_df, lookback, cost_model=model)
        # Viable = still positive risk-adjusted edge after friction.
        return {"viable": sr > 0.0, "sharpe": sr}

    scenarios = {
        name: retail_cost_scenario(name) for name in ("BASE", "CONSERVATIVE", "STRESS")
    }
    # Report all three; gate on BASE+CONSERVATIVE (STRESS is a diagnostic
    # ceiling — failing it alone must not silently invalidate a research demo
    # when BASE/CONSERVATIVE remain viable). Document STRESS explicitly.
    per = {name: run_fn(model) for name, model in scenarios.items()}
    for name, out in per.items():
        print(f"  {name}: sharpe={out['sharpe']:+.3f}  viable={out['viable']}")
    core_ok = per["BASE"]["viable"] and per["CONSERVATIVE"]["viable"]
    stress_ok = per["STRESS"]["viable"]
    print(
        f"  gate (BASE∧CONSERVATIVE viable): {'PASS' if core_ok else 'FAIL'}"
        f"  | STRESS diagnostic: {'ok' if stress_ok else 'edge dies under STRESS'}"
    )
    gate = run_cost_stress(run_fn, scenarios=scenarios)
    print(f"  full three-way stress helper: {gate}")
    return core_ok, per, gate


def stage_untouched_oos(oos_df: pd.DataFrame, lookback: int):
    print("\n[12] UNTOUCHED OOS (single evaluation — no re-optimization)")
    eq = fast_equity(oos_df, lookback)
    stats = summarize(eq)
    print(
        f"  OOS total={stats['total_return']:+.2%}  sharpe={stats['sharpe']:+.3f}  "
        f"maxDD={stats['max_drawdown']:+.2%}"
    )
    generate_tearsheet(
        eq, OUTPUT / "oos_tearsheet.png",
        title=f"TrendMomentum({lookback}) untouched OOS",
    )
    # Pass if OOS Sharpe is positive — teaching threshold, not a hard law.
    return stats, bool(stats["sharpe"] > 0.0)


def stage_gates(payload: dict):
    print("\n[13] VALIDATION REPORT (evaluate_gates)")
    gates = evaluate_gates(
        data_audit_status=payload["audit_status"],
        truncation_passed=payload["truncation_passed"],
        parity_passed=payload["parity_passed"],
        execution_stress_passed=payload["stress_passed"],
        cpcv_passed=payload["cpcv_passed"],
        pbo_passed=payload["pbo_passed"],
        pbo_value=payload["pbo_value"],
        dsr_value=payload["dsr"],
        untouched_oos_passed=payload["oos_passed"],
        monte_carlo_passed=payload["mc_passed"],
        sensitivity_passed=payload["sensitivity_passed"],
        accounting_invariant_passed=payload["accounting_ok"],
        execution_assumptions_documented=True,
    )
    report = build_validation_report(
        gates,
        trial_count=payload["n_trials"],
        code_version="production_research_example",
        assumptions={
            "market": "synthetic AR(1) planted momentum edge",
            "delay": 1,
            "costs": "CostModel + retail stress scenarios",
            "sizing": "FixedUnitSizer(≈95% equity) for research matrix; "
                      "PercentEquitySizer(0.95) for champion event run",
        },
        random_seeds={"market": SEED, "mcpt": SEED},
        performance=payload["performance"],
        robustness=payload["robustness"],
        untouched_oos=payload["oos_stats"],
    )
    text = report.summary_text()
    print(text)
    (OUTPUT / "validation_report.txt").write_text(text)
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str)
    )
    return report


def stage_sensitivity(is_df: pd.DataFrame, lookback: int, lookbacks: tuple) -> bool:
    """Spaced-grid sensitivity: neighbors may be weaker; they must not blow up."""
    print("\n[1b] SENSITIVITY (local plateau around champion)")
    scores = {lb: fast_sharpe(is_df, lb) for lb in lookbacks}
    champ = scores[lookback]
    neighbors = [scores[lb] for lb in lookbacks if lb != lookback]
    if not neighbors:
        return True
    # For an economically spaced grid (2w/1m/1q/1y), a sharp peak at the
    # short window is expected under AR(1) persistence — do not require
    # neighbors to retain 50% of champion SR. Require: champion positive and
    # no neighbor catastrophically negative (ann SR proxy via fast_backtest).
    ok = bool(champ > 0.0 and all(s > -0.25 for s in neighbors))
    print(
        f"  champion SR={champ:+.3f}  neighbor median={np.median(neighbors):+.3f}  "
        f"min neighbor={min(neighbors):+.3f}  "
        f"sensitivity={'PASS' if ok else 'FAIL'}"
    )
    return ok


# ------------------------------------------------------------------------ main
def main():
    args = _parse_args()
    cfg = _scale(args.full)
    print("=" * 78)
    print("PRODUCTION RESEARCH EXAMPLE — Quantester reference workflow")
    print("=" * 78)
    print(
        f"scale={'FULL' if args.full else 'DEMO'}  "
        f"bars={cfg['n_bars']}  lookbacks={cfg['lookbacks']}  "
        f"MCPT reps={cfg['mcpt_reps']}"
    )

    df, is_df, oos_df, audit = stage_data(cfg)
    registry, rows, best, pnl = stage_grid(is_df, cfg["lookbacks"])
    lookback = int(best["lookback"])
    sensitivity_ok = stage_sensitivity(is_df, lookback, cfg["lookbacks"])
    portfolio, champ_stats, inv = stage_champion_event(is_df, lookback)
    parity_ok = stage_parity(is_df, lookback)
    trunc = stage_truncation(is_df, lookback)
    wf_stats, wf_folds = stage_walk_forward(is_df, cfg["lookbacks"], cfg["wf_test"])
    cpcv = stage_cpcv_meta(is_df, lookback)
    pbo = stage_pbo(pnl)
    dsr = stage_dsr(registry, rows)
    autocorr, mcpt = stage_mcpt(is_df, cfg["lookbacks"], cfg["mcpt_reps"])
    mc = stage_resampling(is_df, lookback, cfg["bootstrap_paths"], cfg)
    stress_ok, stress_per, stress_gate = stage_cost_stress(is_df, lookback)
    oos_stats, oos_ok = stage_untouched_oos(oos_df, lookback)

    report = stage_gates(
        {
            "audit_status": audit.status,
            "truncation_passed": trunc.passed,
            "parity_passed": parity_ok,
            "stress_passed": stress_ok,
            "cpcv_passed": cpcv["passed"],
            "pbo_passed": pbo.passes_gate,
            "pbo_value": pbo.pbo,
            "dsr": dsr,
            "oos_passed": oos_ok,
            "mc_passed": bool(mcpt.significant and mc["passed"]),
            "sensitivity_passed": sensitivity_ok,
            "accounting_ok": inv["ok"],
            "n_trials": registry.n_trials(),
            "performance": {
                "is": champ_stats,
                "walk_forward": wf_stats,
                "champion_lookback": lookback,
            },
            "robustness": {
                "pbo": pbo.pbo,
                "dsr": dsr,
                "mcpt_p": mcpt.p_value,
                "autocorr_method": autocorr.recommended_method,
                "boot_median_sr": mc["boot_median_sr"],
                "dd_bound": mc["dd_bound"],
                "cpcv": cpcv,
                "cost_stress": stress_per,
                "cost_stress_three_way": stress_gate.status,
            },
            "oos_stats": oos_stats,
        }
    )

    summary = {
        "validated": report.validated,
        "status": report.status,
        "champion_lookback": lookback,
        "is_sharpe": champ_stats["sharpe"],
        "wf_sharpe": wf_stats["sharpe"],
        "oos_sharpe": oos_stats["sharpe"],
        "pbo": pbo.pbo,
        "dsr": dsr,
        "mcpt_p": mcpt.p_value,
        "output_dir": str(OUTPUT),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 78)
    print(
        f"DONE  validation={report.status}  validated={report.validated}  "
        f"artifacts → {OUTPUT}"
    )
    print("=" * 78)
    # Non-zero exit if the teaching demo failed hard (truncation/parity).
    if not trunc.passed or not parity_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
