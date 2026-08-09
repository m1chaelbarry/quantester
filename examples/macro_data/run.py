"""Macro overlays: World Bank / NBP aligned onto a price calendar.

Run from the repo root:
    pip install "quantester[data]"
    python examples/macro_data/run.py

Without network the script degrades gracefully. Uses synthetic OHLCV locally
and (when online) real NBP + World Bank series as exogenous features — not
as StreamingDataHandler bars.
"""

from __future__ import annotations

import pandas as pd

from quantester import MovingAverageCrossStrategy, make_synthetic_ohlcv, run_backtest
from quantester.macro import as_daily_reindex, load_nbp_fx, load_world_bank


def _synthetic_backtest() -> None:
    # Use a calendar that overlaps free NBP/World Bank windows in the demo.
    data = make_synthetic_ohlcv(
        "AAA", n_bars=260, seed=7, start="2023-01-01",
    )
    result = run_backtest(
        data, MovingAverageCrossStrategy, symbol="AAA", fast=10, slow=40,
    )
    print(
        f"  synthetic MA(10/40): sharpe={result.sharpe:+.3f}  "
        f"mdd={result.max_drawdown:+.2%}  bars={len(data)}"
    )
    return data


def _macro_overlay(calendar: pd.DatetimeIndex) -> None:
    print("\n[macro] NBP USD/PLN mid + World Bank US CPI (aligned to bars)")
    start = calendar[0].strftime("%Y-%m-%d")
    end = calendar[-1].strftime("%Y-%m-%d")

    fx = None
    cpi = None
    try:
        fx = load_nbp_fx("USD", start=start, end=end)
    except Exception as exc:
        print(f"  NBP skipped ({type(exc).__name__}: {exc})")
    try:
        cpi = load_world_bank(
            "FP.CPI.TOTL.ZG", "USA", start=2015, end=2025, timeout=60.0,
        )
    except Exception as exc:
        print(f"  World Bank skipped ({type(exc).__name__}: {exc})")

    if fx is None and cpi is None:
        print(
            "  no live macro fetch; install quantester[data] and retry with network."
        )
        toy = pd.Series(
            [1.0, 1.1],
            index=pd.to_datetime([calendar[0], calendar[min(60, len(calendar) - 1)]]),
            name="toy",
        )
        if toy.index.tz is None:
            toy.index = toy.index.tz_localize("UTC")
        aligned = as_daily_reindex(calendar[:60], toy)
        print(
            f"  toy align demo: calendar={len(calendar[:60])}  "
            f"ffilled_non_null={int(aligned.notna().sum())}"
        )
        return

    if fx is not None:
        fx_daily = as_daily_reindex(calendar, fx)
        fx_tail = fx_daily.dropna()
        print(
            f"  NBP USD mid: obs={len(fx)}  aligned_non_null="
            f"{int(fx_daily.notna().sum())}"
            + (f"  last={float(fx_tail.iloc[-1]):.4f}" if len(fx_tail) else "")
        )
        trail = fx_daily.rolling(20, min_periods=5).mean()
        gap = (fx_daily / trail - 1.0).dropna()
        if len(gap):
            print(
                f"  FX vs 20d mean: mean_gap={float(gap.mean()):+.4%}  "
                f"last={float(gap.iloc[-1]):+.4%}"
            )
    if cpi is not None:
        cpi_daily = as_daily_reindex(calendar, cpi)
        cpi_tail = cpi_daily.dropna()
        print(
            f"  WB US CPI YoY: obs={len(cpi)}  aligned_non_null="
            f"{int(cpi_daily.notna().sum())}"
            + (f"  last={float(cpi_tail.iloc[-1]):.3f}" if len(cpi_tail) else "")
        )


def main() -> None:
    print("=" * 72)
    print("Quantester example: macro overlays (World Bank + NBP)")
    print("=" * 72)
    print("\n[bars] synthetic equity for the engine (macro is exogenous)")
    data = _synthetic_backtest()
    _macro_overlay(data.index)


if __name__ == "__main__":
    main()
