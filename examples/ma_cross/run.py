"""Tier-1 example: MA cross sweep + tearsheet + truncation + DSR.

Uses the one-call ``run_backtest`` API for each trial, then shows the
research extras (trials registry / DSR) once you outgrow hello_trader.

Run from the repo root:  python examples/ma_cross/run.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

from quantester import (
    MovingAverageCrossStrategy,
    generate_tearsheet,
    make_synthetic_ohlcv,
    run_backtest,
    summarize,
)
from quantester.analytics.dsr import dsr_from_registry
from quantester.analytics.performance import carver_cost_drag_sr, speed_limit_warning
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.utils.synthetic import write_csvs

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "examples" / "data"
OUTPUT_DIR = HERE / "output"


def build_data() -> dict:
    data = {
        "AAA": make_synthetic_ohlcv("AAA", seed=1, mu=0.10, sigma=0.22),
        "BBB": make_synthetic_ohlcv(
            "BBB", seed=2, mu=0.05, sigma=0.18, s0=80.0, missing_every=47,
        ),
        "CCC": make_synthetic_ohlcv("CCC", seed=3, mu=-0.02, sigma=0.25, s0=120.0),
    }
    return write_csvs(data, DATA_DIR)


def main():
    print("=" * 72)
    print("Quantester example: MA-cross event-driven backtest")
    print("=" * 72)
    csv_map = build_data()
    symbol = "AAA"

    registry = TrialsRegistry()
    best_sharpe, best_params, best_result = -np.inf, None, None
    for fast, slow in [(5, 20), (10, 40), (20, 60)]:
        result = run_backtest(
            csv_map,
            MovingAverageCrossStrategy,
            symbol=symbol,
            fast=fast,
            slow=slow,
            direction="both",
        )
        stats = result.stats
        rets = result.equity.pct_change().dropna()
        registry.log_trial(
            params={"symbol": symbol, "fast": fast, "slow": slow},
            sharpe=stats["sharpe"],
            mean=float(rets.mean()),
            std=float(rets.std()),
            skew=float(skew(rets)),
            kurt=float(kurtosis(rets, fisher=False)),
            n_obs=len(rets),
            run_id="ma_cross_sweep",
        )
        print(
            f"fast={fast:>2} slow={slow:>2}  sharpe={stats['sharpe']:+.3f}  "
            f"mdd={stats['max_drawdown']:+.2%}  calmar={stats['calmar']:+.3f}"
        )
        if stats["sharpe"] > best_sharpe:
            best_sharpe, best_params, best_result = stats["sharpe"], (fast, slow), result

    best = registry.best_trial()
    dsr = dsr_from_registry(
        registry,
        sr_hat=best["sharpe"],
        n_obs=best["n_obs"],
        skew=best["skew"],
        kurtosis=best["kurt"],
    )
    print(
        f"\nBest trial: fast={best_params[0]} slow={best_params[1]} "
        f"sharpe={best_sharpe:+.3f}"
    )
    print(f"DSR (N={registry.n_trials()} trials, registry-driven): {dsr:.3f}")

    drag = carver_cost_drag_sr(annual_turnover=4.0, standardized_cost_sr=0.01)
    warning = speed_limit_warning(drag)
    print(
        f"Carver cost drag: {drag:.3f} SR/yr"
        + (f"  WARNING: {warning}" if warning else " (within speed limit)")
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = generate_tearsheet(
        best_result.equity,
        OUTPUT_DIR / "ma_cross_tearsheet.png",
        title=f"MA({best_params[0]}/{best_params[1]}) cross on {symbol}",
        extra_stats={"DSR": f"{dsr:.3f}", "cost_drag_SR": f"{drag:.3f}"},
    )
    print(f"\nTearsheet: {OUTPUT_DIR / 'ma_cross_tearsheet.png'}")
    print(
        {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in stats.items()
            if not isinstance(v, str)
        }
    )

    print(f"\n{best_result.check_lookahead(n_truncate=30)}")
    registry.close()


if __name__ == "__main__":
    main()
