#!/usr/bin/env python3
"""Production-grade research pipeline — the place to learn Quantester.

How to read this file
---------------------
1. Skim ``strategy.py`` (the rule) and ``market.py`` (the fake data).
2. Skim ``wiring.py`` (how the five modules connect).
3. Read ``main()`` at the bottom — it is a numbered checklist.
4. Jump into any ``stage_*`` function for the detail of that checklist item.

Run from the repository root::

    python examples/production_research/run.py          # demo scale (~20s)
    python examples/production_research/run.py --full   # heavier MCPT / bootstrap

Artifacts land in ``examples/production_research/output/`` (gitignored).

The market is fiction with a known answer (planted AR(1) momentum). A green
``VALIDATED`` here means the *workflow* is intact — not permission to size
real capital.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make this folder importable so ``from market import ...`` works when you
# run the script from the repo root.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import annualized_sharpe, summarize
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.data.audit import audit_ohlcv_frame
from quantester.execution.costs import retail_cost_scenario
from quantester.montecarlo.adaptive import adaptive_empirical_resample
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.drawdown import double_bootstrap_dd_bound
from quantester.montecarlo.permutation import permutation_test
from quantester.montecarlo.synthetic import bootstrap_ohlcv
from quantester.montecarlo.trade_resampling import ehlers_randomized_equity
from quantester.portfolio.portfolio import FixedUnitSizer, PercentEquitySizer
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
from strategy import momentum_positions
from wiring import (
    INITIAL_CAPITAL,
    SEED,
    SYMBOL,
    daily_pnl,
    event_backtest,
    fast_equity,
    fast_sharpe,
    moments,
    units_for,
)


# ===========================================================================
# CLI — demo vs full scale
# ===========================================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="Quantester production-research teaching pipeline.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Production-scale reps (MCPT 1000, more bootstrap paths).",
    )
    return p.parse_args()


def _scale(full: bool) -> dict:
    """Knob set for demo (~20s) vs heavier validation.

    Lookbacks are *economically spaced* (≈2w / 1m / 1q / 1y). A dense cluster
    of near-identical windows drives PBO → 1 by construction — avoid that.
    """
    if full:
        return dict(
            n_bars=1_764,
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


def _banner(stage: str, title: str) -> None:
    print(f"\n[{stage}] {title}")


# ===========================================================================
# STAGE [0] — Data audit
# WHAT: Build the synthetic market and prove the frame is research-ready.
# WHY:  Never optimize on garbage (tz, NaNs, undocumented adjustments).
# NEED: market.make_momentum_edge_ohlcv, quantester.data.audit.audit_ohlcv_frame
# ===========================================================================
def stage_data(cfg: dict):
    _banner("0", "PLANTED-EDGE MARKET + DATA AUDIT")

    # Build fiction OHLCV with a known AR(1) momentum edge (see market.py).
    df = make_momentum_edge_ohlcv(n_bars=cfg["n_bars"], seed=SEED)
    df.to_csv(OUTPUT / "EDGE.csv", index_label="datetime")

    # Document every assumption the auditor asks about. WARN ≠ pass.
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

    # Seal the final 25% — we will not look at it until stage [12].
    is_df, oos_df = split_is_oos(df, oos_fraction=0.25)
    print(f"  IS bars={len(is_df)}  untouched OOS bars={len(oos_df)}")
    return df, is_df, oos_df, audit


# ===========================================================================
# STAGE [1] — In-sample grid → TrialsRegistry
# WHAT: Try economically spaced lookbacks on IS only; log every real trial.
# WHY:  DSR needs an honest N. Do not stuff junk decoys into the registry.
# NEED: wiring.fast_equity, quantester.analytics.trials_registry.TrialsRegistry
# ===========================================================================
def stage_grid(is_df: pd.DataFrame, lookbacks: tuple) -> tuple:
    _banner("1", "IN-SAMPLE PARAMETER GRID → TrialsRegistry")

    # Fresh DB every run — never silently accumulate prior demo trials into N.
    db_path = OUTPUT / "trials.db"
    if db_path.exists():
        db_path.unlink()
    registry = TrialsRegistry(str(db_path))

    rows = []
    pnl_cols = {}  # columns become the PBO trial matrix in stage [7]

    for lb in lookbacks:
        eq = fast_equity(is_df, lb)
        m = moments(eq)
        params = {
            "lookback": lb,
            "family": "trend",
            "vol_lookback": 20,
            "target_vol": 0.15,
        }
        # Log what you would actually consider — including losers.
        registry.log_trial(
            params=params,
            sharpe=m["sharpe_daily"],  # per-period; DSR needs annualized=False
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

    # Rejected a priori (mean-reversion): print it, do NOT log it.
    # Logging known-bad decoys games or wrecks the DSR deflator.
    decoy_lb = lookbacks[0]
    decoy = moments(fast_equity(is_df, decoy_lb, invert=True))
    print(
        f"  rejected a priori: mean-reversion lb={decoy_lb}  "
        f"annSR={decoy['sharpe_ann']:+.3f} (not in registry / PBO matrix)"
    )

    best = max(rows, key=lambda r: r["sharpe_daily"])
    print(f"  champion lookback={best['lookback']} (IS only — OOS still sealed)")
    return registry, rows, best, pd.DataFrame(pnl_cols)


# ===========================================================================
# STAGE [1b] — Sensitivity around the champion
# WHAT: Neighbors on the spaced grid may be weaker; they must not blow up.
# WHY:  A needle-thin peak is usually noise, not an edge.
# ===========================================================================
def stage_sensitivity(is_df: pd.DataFrame, lookback: int, lookbacks: tuple) -> bool:
    _banner("1b", "SENSITIVITY (local plateau around champion)")

    scores = {lb: fast_sharpe(is_df, lb) for lb in lookbacks}
    champ = scores[lookback]
    neighbors = [scores[lb] for lb in lookbacks if lb != lookback]
    if not neighbors:
        return True

    # Spaced grid (2w/1m/1q/1y): a sharp short-window peak is expected under
    # AR(1). Require champion > 0 and no neighbor catastrophically negative.
    ok = bool(champ > 0.0 and all(s > -0.25 for s in neighbors))
    print(
        f"  champion SR={champ:+.3f}  neighbor median={np.median(neighbors):+.3f}  "
        f"min neighbor={min(neighbors):+.3f}  "
        f"sensitivity={'PASS' if ok else 'FAIL'}"
    )
    return ok


# ===========================================================================
# STAGE [2] — Event-engine champion
# WHAT: Full stack run of the locked lookback with risk overlays + tearsheet.
# WHY:  Grids used the fast-track; the champion must survive the real engine.
# NEED: wiring.event_backtest, PercentEquitySizer, MarginMonitor, DD breaker
# ===========================================================================
def stage_champion_event(is_df: pd.DataFrame, lookback: int):
    _banner("2", "EVENT-ENGINE CHAMPION (IS) + RISK OVERLAYS + TEARSHEET")

    portfolio = event_backtest(
        is_df,
        lookback,
        sizer=PercentEquitySizer(0.95),  # size off live equity each bar
        use_risk_overlays=True,          # margin + intraday DD breaker
    )
    equity = portfolio.equity_curve
    stats = summarize(equity)
    inv = portfolio.accounting_invariant()  # cash + MTM must reconcile

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


# ===========================================================================
# STAGE [3] — Event ↔ fast-track parity
# WHAT: Same FixedUnitSizer sizing in both paths; equity must match.
# WHY:  No parity → MCPT on the fast-track is lying about the event engine.
# ===========================================================================
def stage_parity(is_df: pd.DataFrame, lookback: int) -> bool:
    _banner("3", "EVENT ↔ FAST-TRACK PARITY (FixedUnitSizer)")

    units = units_for(is_df)
    port = event_backtest(is_df, lookback, sizer=FixedUnitSizer(units))
    fast_eq = fast_equity(is_df, lookback)

    eng = port.equity_curve.reindex(is_df.index).ffill()
    diff = (eng - fast_eq).abs().max()
    ok = bool(np.isfinite(diff) and diff < 1e-4)

    print(f"  max |equity diff| = {diff:.3e}  parity={'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# STAGE [4] — Truncation diagnostic
# WHAT: Chop the last N bars, re-run, compare overlapping positions.
# WHY:  Mismatch ⇒ look-ahead leak. Everything else is then void.
# NEED: quantester.validation.truncation.run_truncation_test
# ===========================================================================
def stage_truncation(is_df: pd.DataFrame, lookback: int):
    _banner("4", "TRUNCATION DIAGNOSTIC (look-ahead leak detector)")

    def _positions(truncate_last: int | None):
        # truncate_last=None → full series; else drop the final N bars.
        frame = is_df if truncate_last is None else is_df.iloc[:-truncate_last]
        return event_backtest(
            frame, lookback, sizer=FixedUnitSizer(units_for(frame)),
        ).positions_history

    result = run_truncation_test(_positions, n_truncated=30)
    print(f"  {result}")
    return result


# ===========================================================================
# STAGE [5] — Walk-forward
# WHAT: Expanding train → lock lookback → score the next test block.
# WHY:  A single IS Sharpe does not prove the rule survives parameter churn.
# NOTE: Quantester has purged-CV helpers but no WalkForward class — compose it.
# ===========================================================================
def stage_walk_forward(is_df: pd.DataFrame, lookbacks: tuple, test_bars: int):
    _banner("5", "WALK-FORWARD (expanding train → locked test)")

    warm = max(lookbacks) + 40  # history so lookback=252 is not a cold start
    fold = 0
    oos_pieces = []
    fold_rows = []
    t = warm

    while t + test_bars <= len(is_df):
        train = is_df.iloc[:t]
        test = is_df.iloc[t : t + test_bars]

        # Re-select the champion on *train only*.
        champ = max(lookbacks, key=lambda lb: fast_sharpe(train, lb))

        # Warm-start indicators with history before the test window.
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

    # Stitch fold returns into one OOS wealth path.
    chained = pd.concat(oos_pieces)
    wealth = (1.0 + chained).cumprod() * INITIAL_CAPITAL
    wf_stats = summarize(wealth)
    print(
        f"  stitched WF: folds={fold}  sharpe={wf_stats['sharpe']:+.3f}  "
        f"total={wf_stats['total_return']:+.2%}"
    )
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "walk_forward_folds.csv", index=False)
    return wf_stats, fold_rows


# ===========================================================================
# STAGE [6] — CPCV + triple-barrier (educational)
# WHAT: Show the API you must use the moment a secondary model is fitted.
# WHY:  Our primary rule is rule-based → CPCV gate is NOT_APPLICABLE.
# NEED: CombinatorialPurgedKFold, triple_barrier_labels, LogisticRegression
# ===========================================================================
def stage_cpcv_meta(is_df: pd.DataFrame, lookback: int) -> dict:
    _banner("6", "CPCV + TRIPLE-BARRIER META-LABELS (educational secondary fit)")
    print(
        "  Note: the primary TrendMomentum rule does NOT fit a model, so CPCV is"
        "  NOT a mandatory gate for it. This section shows the API you must use"
        "  IF you add a fitted secondary (meta-label) model."
    )

    close = is_df["close"]
    pos = momentum_positions(close, lookback=lookback)
    side = np.sign(pos)

    # Primary "events" = bars where the signed side changes and is non-flat.
    changes = side[(side != side.shift(1)) & (side != 0)]
    if len(changes) < 30:
        print("  too few primary events for CPCV demo")
        return {
            "passed": None, "n_events": int(len(changes)),
            "mean_acc": None, "gate_value": None,
        }

    events = pd.DataFrame({"t0": changes.index, "side": changes.to_numpy()})
    events.index = changes.index

    # Labels: did the side hit TP before SL within max_holding bars?
    y = triple_barrier_labels(
        close, events, tp_pct=0.02, sl_pct=0.02, max_holding=10
    )

    # Tiny feature set for the educational logistic meta-labeler.
    mom = close / close.shift(lookback) - 1.0
    vol = close.pct_change().rolling(20).std()
    X = pd.DataFrame({"mom": mom, "vol": vol}, index=close.index).loc[y.index]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    # Label end-times for purge/embargo (overlap-aware CV).
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


# ===========================================================================
# STAGE [7] — PBO / CSCV
# WHAT: Probability of backtest overfitting from the IS trial PnL matrix.
# WHY:  Selection across many looks makes the best IS Sharpe look fake.
# GATE: PBO < 0.10
# ===========================================================================
def stage_pbo(pnl: pd.DataFrame):
    _banner("7", "PBO / CSCV (selection overfitting)")

    result = pbo_cscv(pnl, n_blocks=10 if pnl.shape[0] < 400 else 16)
    print(
        f"  PBO={result.pbo:.3f} over N={result.n_trials} trials "
        f"({result.n_combinations} combos)  gate<{PBO_GATE}: "
        f"{'PASS' if result.passes_gate else 'FAIL'}"
    )
    return result


# ===========================================================================
# STAGE [8] — Deflated Sharpe Ratio
# WHAT: Deflate the champion's daily Sharpe by registry-honest N + variance.
# WHY:  Multiple testing inflates the best Sharpe; DSR undoes that.
# GATE: DSR ≥ 0.95
# ===========================================================================
def stage_dsr(registry: TrialsRegistry) -> float:
    _banner("8", "DEFLATED SHARPE (registry-honest N)")

    best = registry.best_trial()
    dsr = dsr_from_registry(
        registry,
        sr_hat=best["sharpe"],
        n_obs=best["n_obs"],
        skew=best["skew"] or 0.0,
        kurtosis=best["kurt"] or 3.0,
        annualized=False,  # we stored daily (per-period) Sharpes
    )
    print(
        f"  best params={best['params']}  dailySR={best['sharpe']:+.4f}  "
        f"N={registry.n_trials()}  DSR={dsr:.4f}  gate≥0.95: "
        f"{'PASS' if dsr >= 0.95 else 'FAIL'}"
    )
    return float(dsr)


# ===========================================================================
# STAGE [9] — Autocorr gate → MCPT
# WHAT: Check serial correlation, then permute chronology and re-optimize.
# WHY:  If the edge is only the return *distribution*, permutations win too.
# GATE: Masters p-value < 0.05  (n_reps includes the original)
# ===========================================================================
def stage_mcpt(is_df: pd.DataFrame, lookbacks: tuple, n_reps: int):
    _banner("9", "AUTOCORR GATE → MCPT (retrain-on-permute via fast-track)")

    # Autocorr first: iid resampling is invalid when returns are serially linked.
    champ = max(lookbacks, key=lambda lb: fast_sharpe(is_df, lb))
    strat_rets = fast_equity(is_df, champ).pct_change().dropna()
    gate = autocorrelation_gate(strat_rets.to_numpy())
    print(
        f"  autocorr serial={gate.serial_correlation}  "
        f"method={gate.recommended_method}"
    )

    def optimizer(series: pd.Series) -> float:
        """On each permuted close path: rebuild OHLC and re-pick lookback."""
        ohlc = is_df.copy()
        ohlc["close"] = series.reindex(ohlc.index).ffill().bfill()
        ohlc["open"] = ohlc["close"].shift(1).fillna(ohlc["close"].iloc[0])
        span = (is_df["high"] - is_df["low"]).median()
        ohlc["high"] = np.maximum(ohlc["open"], ohlc["close"]) + span / 2.0
        ohlc["low"] = np.minimum(ohlc["open"], ohlc["close"]) - span / 2.0
        return max(fast_sharpe(ohlc, lb) for lb in lookbacks)

    t0 = time.perf_counter()
    result = permutation_test(is_df["close"], optimizer, n_reps=n_reps, seed=SEED)
    print(
        f"  MCPT p={result.p_value:.4f}  origSR={result.original_performance:+.3f}  "
        f"reps={result.n_reps}  significant={result.significant}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return gate, result


# ===========================================================================
# STAGE [10] — Path-risk suite
# WHAT: Block bootstrap, nested DD bound, Ehlers trade-resampling paths.
# WHY:  A point Sharpe ignores path dependence and drawdown risk.
# ===========================================================================
def stage_resampling(is_df: pd.DataFrame, lookback: int, n_paths: int, cfg: dict):
    _banner("10", "BLOCK BOOTSTRAP / ADAPTIVE RESAMPLE / DD BOUND / EHLERS")

    eq = fast_equity(is_df, lookback)
    rets = eq.pct_change().dropna().to_numpy()

    # Adaptive chooser: iid vs block vs other, based on the return series.
    adapted = adaptive_empirical_resample(
        rets, horizon=min(252, len(rets) // 2), n_sims=n_paths, seed=SEED
    )
    print(
        f"  adaptive resample method={adapted.method_used}  "
        f"block={adapted.block_length}  "
        f"terminal quantiles={adapted.result.quantiles()}"
    )

    # Re-run the locked rule on bootstrapped OHLC paths.
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

    # Ehlers paths from win rate + avg_win/avg_loss (not gross profit factor).
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

    # Cheap MC viability: majority of bootstrap paths keep positive Sharpe.
    majority_positive = float((boot_sharpes > 0).mean()) >= 0.55
    mc_pass = bool(np.median(boot_sharpes) > 0 and majority_positive)
    return {
        "adapted_method": adapted.method_used,
        "boot_median_sr": float(np.median(boot_sharpes)),
        "boot_p05_sr": float(np.percentile(boot_sharpes, 5)),
        "dd_bound": float(dd.bound),
        "passed": mc_pass,
    }


# ===========================================================================
# STAGE [11] — Cost stress
# WHAT: Re-run under BASE / CONSERVATIVE / STRESS retail cost scenarios.
# WHY:  An edge that dies under modest friction is not tradeable.
# GATE: BASE ∧ CONSERVATIVE still viable (STRESS is a diagnostic ceiling)
# ===========================================================================
def stage_cost_stress(is_df: pd.DataFrame, lookback: int):
    _banner("11", "EXECUTION COST STRESS (BASE / CONSERVATIVE / STRESS)")

    def run_fn(model):
        sr = fast_sharpe(is_df, lookback, cost_model=model)
        return {"viable": sr > 0.0, "sharpe": sr}

    scenarios = {
        name: retail_cost_scenario(name) for name in ("BASE", "CONSERVATIVE", "STRESS")
    }
    per = {name: run_fn(model) for name, model in scenarios.items()}
    for name, out in per.items():
        print(f"  {name}: sharpe={out['sharpe']:+.3f}  viable={out['viable']}")

    core_ok = per["BASE"]["viable"] and per["CONSERVATIVE"]["viable"]
    stress_ok = per["STRESS"]["viable"]
    print(
        f"  gate (BASE∧CONSERVATIVE viable): {'PASS' if core_ok else 'FAIL'}"
        f"  | STRESS diagnostic: {'ok' if stress_ok else 'edge dies under STRESS'}"
    )

    # Library helper reports the full three-way stress object for the paper trail.
    gate = run_cost_stress(run_fn, scenarios=scenarios)
    print(f"  full three-way stress helper: {gate}")
    return core_ok, per, gate


# ===========================================================================
# STAGE [12] — Untouched out-of-sample
# WHAT: One evaluation of the locked champion on the sealed OOS slice.
# WHY:  Everything above was IS. This is the only honest holdout score.
# RULE: No re-optimization. Ever.
# ===========================================================================
def stage_untouched_oos(oos_df: pd.DataFrame, lookback: int):
    _banner("12", "UNTOUCHED OOS (single evaluation — no re-optimization)")

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
    # Teaching threshold: positive OOS Sharpe. Not a universal hard law.
    return stats, bool(stats["sharpe"] > 0.0)


# ===========================================================================
# STAGE [13] — ValidationReport
# WHAT: Fold every gate into one machine-readable verdict.
# WHY:  VALIDATED only when mandatory gates PASS (or N/A) with ≥1 actionable PASS.
# NEED: evaluate_gates, build_validation_report
# ===========================================================================
def stage_gates(payload: dict):
    _banner("13", "VALIDATION REPORT (evaluate_gates)")

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


# ===========================================================================
# main — the checklist. Read top-to-bottom; each line is one research step.
# ===========================================================================
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
    print("Reading order: strategy.py → market.py → wiring.py → this file.")

    # --- Build data & seal OOS ------------------------------------------------
    df, is_df, oos_df, audit = stage_data(cfg)

    # --- Optimize on IS only --------------------------------------------------
    registry, _rows, best, pnl = stage_grid(is_df, cfg["lookbacks"])
    lookback = int(best["lookback"])
    sensitivity_ok = stage_sensitivity(is_df, lookback, cfg["lookbacks"])

    # --- Prove the event engine + twins are trustworthy -----------------------
    portfolio, champ_stats, inv = stage_champion_event(is_df, lookback)
    parity_ok = stage_parity(is_df, lookback)
    trunc = stage_truncation(is_df, lookback)

    # --- Robustness under selection / time / nulls / friction -----------------
    wf_stats, wf_folds = stage_walk_forward(is_df, cfg["lookbacks"], cfg["wf_test"])
    cpcv = stage_cpcv_meta(is_df, lookback)
    pbo = stage_pbo(pnl)
    dsr = stage_dsr(registry)
    autocorr, mcpt = stage_mcpt(is_df, cfg["lookbacks"], cfg["mcpt_reps"])
    mc = stage_resampling(is_df, lookback, cfg["bootstrap_paths"], cfg)
    stress_ok, stress_per, stress_gate = stage_cost_stress(is_df, lookback)

    # --- One-shot sealed holdout ----------------------------------------------
    oos_stats, oos_ok = stage_untouched_oos(oos_df, lookback)

    # --- Paper trail ----------------------------------------------------------
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
                "walk_forward_folds": len(wf_folds),
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

    # Hard fail the teaching demo if look-ahead or twin parity is broken.
    if not trunc.passed or not parity_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
