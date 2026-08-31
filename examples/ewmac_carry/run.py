#!/usr/bin/env python3
"""EWMAC + crypto-carry research pipeline (production_research stages 0–13).

Demo uses a synthetic perpetual with extras so the workflow runs offline.
``python examples/ewmac_carry/run_ccxt.py`` fetches Binance+Deribit when
the extra is installed. A green VALIDATED here means the *workflow* ran —
not permission to size real capital. Hyperparameters are study defaults.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import summarize
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.data.audit import audit_ohlcv_frame
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.drawdown import double_bootstrap_dd_bound
from quantester.montecarlo.permutation import permutation_test, permute_joint_bars
from quantester.portfolio.portfolio import FixedUnitSizer
from quantester.validation.gates import build_validation_report, evaluate_gates
from quantester.validation.pbo import pbo_cscv
from quantester.validation.truncation import run_truncation_test
from quantester.visualization.static import plot_equity

from market import SYMBOL, make_perp_ohlcv, split_is_oos
from wiring import (
    GRID,
    INITIAL_CAPITAL,
    SEED,
    daily_pnl,
    event_backtest,
    fast_equity,
    fast_sharpe,
    moments,
    twin_target,
    units_for,
)
from quantester.execution.costs import perp_cost_scenario
from quantester.montecarlo.fast_track import fast_backtest


def _parse_args():
    p = argparse.ArgumentParser(description="EWMAC + crypto-carry research pipeline")
    p.add_argument("--full", action="store_true")
    return p.parse_args()


def _scale(full: bool) -> dict:
    if full:
        return dict(n_bars=800, mcpt_reps=101, bootstrap_paths=32, wf_test=84)
    return dict(n_bars=420, mcpt_reps=21, bootstrap_paths=8, wf_test=42)


def _banner(stage: str, title: str) -> None:
    print(f"\n[{stage}] {title}")


def _grid_rows():
    for (fast, slow), vol, beta, w_carry in itertools.product(
        GRID["ewmac"], GRID["target_vol"], GRID["beta"], GRID["w_carry"]
    ):
        yield dict(fast=fast, slow=slow, target_vol=vol, beta=beta, w_carry=w_carry)


def stage_data(cfg: dict, ohlcv: pd.DataFrame | None = None):
    _banner("0", "SYNTHETIC PERP + DATA AUDIT" if ohlcv is None else "LOADED PERP + DATA AUDIT")
    df = ohlcv.copy() if ohlcv is not None else make_perp_ohlcv(n_bars=cfg["n_bars"], seed=SEED)
    df.to_csv(OUTPUT / "BTC_PERP.csv", index_label="datetime")
    audit = audit_ohlcv_frame(
        df, SYMBOL, expected_freq="D" if ohlcv is not None else "B",
        adjustment_policy=(
            "binance_um_unadjusted" if ohlcv is not None else "synthetic_unadjusted"
        ),
        corporate_actions_documented=True,
        delistings_considered=True,
        survivorship_bias_considered=True,
        historical_universe_documented=True,
        trading_calendar_documented=True,
    )
    print(f"  bars={len(df)}  extras={list(df.columns[5:])}  audit={audit.status}")
    is_df, oos_df = split_is_oos(df, oos_fraction=0.25)
    print(f"  IS={len(is_df)}  sealed OOS={len(oos_df)}")
    return df, is_df, oos_df, audit


def stage_grid(is_df: pd.DataFrame):
    _banner("1", "IN-SAMPLE 54-TRIAL GRID → TrialsRegistry")
    db_path = OUTPUT / "trials.db"
    if db_path.exists():
        db_path.unlink()
    registry = TrialsRegistry(str(db_path))
    rows = []
    pnl_cols = {}
    for params in _grid_rows():
        eq = fast_equity(is_df, params["fast"], params["slow"], params["w_carry"])
        m = moments(eq)
        registry.log_trial(
            params=params, sharpe=m["sharpe_daily"], skew=m["skew"], kurt=m["kurt"],
            n_obs=m["n_obs"], run_id="is_grid", strategy_id="EWMACCarry",
            metrics={"sharpe_ann": m["sharpe_ann"]},
        )
        key = f"{params['fast']}_{params['slow']}_c{params['w_carry']}_v{params['target_vol']}_b{params['beta']}"
        rows.append({**params, **m, "equity": eq, "key": key})
        # Fast-track ignores vol/β (no inertia/DLR); PBO columns are unique
        # Combined Forecast triples so CSCV is not diluted by duplicate PnL.
        fc_key = f"{params['fast']}_{params['slow']}_c{params['w_carry']}"
        if fc_key not in pnl_cols:
            pnl_cols[fc_key] = daily_pnl(eq).to_numpy()
    best = max(rows, key=lambda r: r["sharpe_daily"])
    print(f"  trials={len(rows)}  champion={best['fast']}/{best['slow']} "
          f"w_carry={best['w_carry']} vol={best['target_vol']} β={best['beta']} "
          f"annSR={best['sharpe_ann']:+.3f}")
    return registry, rows, best, pd.DataFrame(pnl_cols)


def stage_sensitivity(rows: list, best: dict) -> bool:
    _banner("1b", "SENSITIVITY + implied Kelly diagnostic")
    neighbors = [r["sharpe_ann"] for r in rows if r["key"] != best["key"]]
    ok = bool(best["sharpe_ann"] > -1.0 and all(s > -1.5 for s in neighbors))
    # Kelly is diagnostic only (Q25 A): implied f* from daily SR, not a live scale.
    rets = best["equity"].pct_change().dropna()
    mu, var = float(rets.mean()), float(rets.var())
    f_star = mu / var if var > 0 else 0.0
    print(f"  neighbors min SR={min(neighbors):+.3f}  implied f*={f_star:.3f} "
          f"(0.25 Kelly={0.25*f_star:.3f})  sensitivity={'PASS' if ok else 'FAIL'}")
    return ok


def stage_champion_event(is_df: pd.DataFrame, best: dict):
    _banner("2", "EVENT-ENGINE CHAMPION + Carver sizer + tearsheet")
    port = event_backtest(
        is_df, fast=best["fast"], slow=best["slow"], w_carry=best["w_carry"],
        target_vol=best["target_vol"], inertia_beta=best["beta"],
    )
    equity = port.equity_curve
    stats = summarize(equity)
    inv = port.accounting_invariant()
    print(f"  total={stats['total_return']:+.2%}  sharpe={stats['sharpe']:+.3f}  "
          f"maxDD={stats['max_drawdown']:+.2%}  accounting={inv['ok']}")
    generate_tearsheet(equity, OUTPUT / "champion_tearsheet.png",
                       title="EWMAC+Carry IS — event engine")
    plot_equity(equity, port.positions_history, path=OUTPUT / "champion_equity.png",
                title="EWMAC+Carry IS equity")
    return port, stats, inv


def stage_parity(is_df: pd.DataFrame, best: dict) -> bool:
    _banner("3", "EVENT ↔ FAST-TRACK PARITY (FixedUnitSizer on Combined Forecast)")
    units = units_for(is_df)
    port = event_backtest(
        is_df, fast=best["fast"], slow=best["slow"], w_carry=best["w_carry"],
        sizer=FixedUnitSizer(units), costs=perp_cost_scenario("BASE"),
        book_funding_settlements=False,
    )
    target = twin_target(is_df, best["fast"], best["slow"], best["w_carry"])
    fast_eq = fast_backtest(
        is_df, target, perp_cost_scenario("BASE"),
        initial_capital=INITIAL_CAPITAL, units=units,
    ).equity
    eng = port.equity_curve.reindex(is_df.index).ffill()
    diff = float((eng - fast_eq).abs().max())
    ok = bool(np.isfinite(diff) and diff < 5.0)  # notional BTC size; relative check below
    rel = diff / max(float(eng.abs().max()), 1.0)
    ok = bool(np.isfinite(rel) and rel < 0.05)
    print(f"  max |equity diff|={diff:.3e}  rel={rel:.3e}  parity={'PASS' if ok else 'FAIL'}")
    return ok


def stage_truncation(is_df: pd.DataFrame, best: dict):
    _banner("4", "TRUNCATION DIAGNOSTIC")

    def _positions(truncate_last: int | None):
        frame = is_df if truncate_last is None else is_df.iloc[:-truncate_last]
        return event_backtest(
            frame, fast=best["fast"], slow=best["slow"], w_carry=best["w_carry"],
            sizer=FixedUnitSizer(units_for(frame)),
            book_funding_settlements=False,
        ).positions_history

    result = run_truncation_test(_positions, n_truncated=20)
    print(f"  {result}")
    return result


def stage_walk_forward(is_df: pd.DataFrame, test_bars: int):
    _banner("5", "WALK-FORWARD (expanding train → locked test)")
    combos = [(p["fast"], p["slow"], p["w_carry"]) for p in _grid_rows()]
    combos = list(dict.fromkeys(combos))  # unique forecast triples
    warm = 128 + 40
    oos_pieces = []
    t = warm
    while t + test_bars <= len(is_df):
        train, test = is_df.iloc[:t], is_df.iloc[t:t + test_bars]
        champ = max(combos, key=lambda c: fast_sharpe(train, c[0], c[1], c[2]))
        warmed = is_df.iloc[: t + test_bars]
        eq = fast_equity(warmed, champ[0], champ[1], champ[2]).iloc[-len(test):]
        oos_pieces.append(eq)
        t += test_bars
    if not oos_pieces:
        print("  not enough bars for a walk-forward fold")
        return {"sharpe": 0.0}, False
    stitched = pd.concat(oos_pieces)
    stats = summarize(stitched)
    print(f"  folds={len(oos_pieces)}  stitched Sharpe={stats['sharpe']:+.3f}")
    return stats, True


def stage_cpcv():
    _banner("6", "CPCV (N/A — rule-based Combined Forecast, no fitted model)")
    print("  skipped")
    return None


def stage_pbo(pnl: pd.DataFrame):
    _banner("7", "PBO / CSCV")
    n_blocks = 8 if len(pnl) < 16 else 16
    result = pbo_cscv(pnl, n_blocks=n_blocks)
    print(f"  PBO={result.pbo:.4f}  gate={'PASS' if result.passes_gate else 'FAIL'}")
    return result


def stage_dsr(registry: TrialsRegistry, best: dict):
    _banner("8", "DSR FROM REGISTRY")
    dsr = dsr_from_registry(
        registry, sr_hat=best["sharpe_daily"], n_obs=best["n_obs"],
        skew=best["skew"], kurtosis=best["kurt"],
    )
    print(f"  DSR={dsr:.4f}  N={registry.n_trials()}")
    return dsr


def stage_mcpt(is_df: pd.DataFrame, n_reps: int, best: dict):
    _banner("9", "AUTOCORR GATE → MCPT (joint-bar permute, re-opt on permute)")
    rets = is_df["close"].pct_change().dropna()
    gate = autocorrelation_gate(rets.to_numpy())
    print(f"  autocorr gate={gate}")

    combos = list(dict.fromkeys(
        (p["fast"], p["slow"], p["w_carry"]) for p in _grid_rows()
    ))

    def optimizer(frame: pd.DataFrame) -> float:
        return max(fast_sharpe(frame, f, s, w) for f, s, w in combos)

    mcpt = permutation_test(
        is_df, optimizer, n_reps=n_reps, seed=SEED, permute_fn=permute_joint_bars,
    )
    print(f"  MCPT p={mcpt.p_value:.4f}  origSR={mcpt.original_performance:+.3f}")
    return gate, mcpt


def stage_resampling(is_df: pd.DataFrame, best: dict, n_paths: int):
    _banner("10", "BLOCK BOOTSTRAP / DD BOUND")
    eq = fast_equity(is_df, best["fast"], best["slow"], best["w_carry"])
    rets = eq.pct_change().dropna().to_numpy()
    bound = double_bootstrap_dd_bound(
        rets, n_outer=min(40, n_paths * 2),
        n_inner=max(8, n_paths // 2), seed=SEED,
    )
    print(f"  nested DD bound={bound.bound:.4f}")
    return True


def stage_cost_stress(is_df: pd.DataFrame, best: dict):
    _banner("11", "COST STRESS (perp maker/taker)")
    target = twin_target(is_df, best["fast"], best["slow"], best["w_carry"])
    scores = {}
    for name in ("BASE", "CONSERVATIVE", "STRESS"):
        eq = fast_backtest(
            is_df, target, perp_cost_scenario(name),
            initial_capital=INITIAL_CAPITAL, units=units_for(is_df),
        ).equity
        scores[name] = summarize(eq)["sharpe"]
        print(f"  {name:12} Sharpe={scores[name]:+.3f}")
    viable = scores["BASE"] > -1.0 and scores["CONSERVATIVE"] > -1.5
    return viable, scores


def stage_untouched_oos(oos_df: pd.DataFrame, best: dict):
    _banner("12", "UNTOUCHED OOS (no re-opt)")
    eq = fast_equity(oos_df, best["fast"], best["slow"], best["w_carry"])
    stats = summarize(eq)
    print(f"  OOS Sharpe={stats['sharpe']:+.3f}  total={stats['total_return']:+.2%}")
    return stats, stats["sharpe"] > -1.0


def stage_report(payload: dict):
    _banner("13", "evaluate_gates → ValidationReport")
    gates = evaluate_gates(
        data_audit_status=payload["audit_status"],
        truncation_passed=payload["truncation_passed"],
        parity_passed=payload["parity_passed"],
        execution_stress_passed=payload["stress_passed"],
        cpcv_passed=None,
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
        code_version="ewmac_carry_example",
        assumptions={
            "instrument": "BTC USDT-M perpetual (synthetic extras in demo)",
            "delay": 1,
            "sizing": "Carver vol-target + Inertia Buffer + Drawdown De-lever; Kelly diagnostic only",
            "costs": "PerpMakerTakerCostModel VIP0 2/5 bps",
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


def main(ohlcv: pd.DataFrame | None = None):
    args = _parse_args()
    cfg = _scale(args.full)
    print("=" * 78)
    print("EWMAC + CRYPTO CARRY — Combined Forecast research pipeline")
    print("=" * 78)
    df, is_df, oos_df, audit = stage_data(cfg, ohlcv=ohlcv)
    registry, rows, best, pnl = stage_grid(is_df)
    sensitivity_ok = stage_sensitivity(rows, best)
    portfolio, champ_stats, inv = stage_champion_event(is_df, best)
    parity_ok = stage_parity(is_df, best)
    trunc = stage_truncation(is_df, best)
    wf_stats, wf_ok = stage_walk_forward(is_df, cfg["wf_test"])
    stage_cpcv()
    pbo = stage_pbo(pnl)
    dsr = stage_dsr(registry, best)
    autocorr, mcpt = stage_mcpt(is_df, cfg["mcpt_reps"], best)
    mc_ok = stage_resampling(is_df, best, cfg["bootstrap_paths"])
    stress_ok, stress_scores = stage_cost_stress(is_df, best)
    oos_stats, oos_ok = stage_untouched_oos(oos_df, best)
    report = stage_report({
        "audit_status": audit.status,
        "truncation_passed": bool(trunc.passed),
        "parity_passed": parity_ok,
        "stress_passed": stress_ok,
        "pbo_passed": bool(pbo.passes_gate),
        "pbo_value": float(pbo.pbo),
        "dsr": float(dsr),
        "oos_passed": oos_ok,
        "mc_passed": bool(mcpt.p_value < 0.05),
        "sensitivity_passed": sensitivity_ok,
        "accounting_ok": bool(inv["ok"]),
        "n_trials": registry.n_trials(),
        "performance": champ_stats,
        "robustness": {"walk_forward": wf_stats, "mcpt_p": mcpt.p_value,
                       "cost_stress": stress_scores},
        "oos_stats": oos_stats,
    })
    print(f"\nArtifacts: {OUTPUT}")
    return report.status


if __name__ == "__main__":
    main()
