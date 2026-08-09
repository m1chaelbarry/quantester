"""Shared constants and helpers for Donchian hourly BTC examples (DRY)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from quantester.analytics.performance import (
    annualized_sharpe,
    calmar_ratio,
    max_drawdown,
)
from quantester.execution.costs import ConservativeFrictionCostModel, CostModel
from quantester.portfolio.portfolio import PortfolioManager

# Script-dir import: ``python examples/donchian_breakout/run_*.py`` puts THIS
# folder on sys.path[0]; parent ``examples/`` is added so ``_common`` resolves.
_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))
from _common import DATA_DIR, output_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = output_dir(__file__)
CACHE = DATA_DIR / "BTCUSD_bitstamp_1h.csv"
SYMBOL = "BTC/USD"
INITIAL_CAPITAL = 25_000.0
PERIODS = 24 * 365

FRICTION = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)
ZERO = CostModel(
    fixed_commission=0.0,
    per_share_commission=0.0,
    spread_pct=0.0,
    slippage_vol_coef=0.0,
    impact_coef=0.0,
)


def load_or_fetch() -> pd.DataFrame:
    """Load cached Bitstamp 1h BTC/USD, or fetch once via CCXT and cache."""
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["datetime"], index_col="datetime")
    from quantester.data.ccxt_handler import CCXTDataHandler

    print("Fetching BTC/USD 1h from Bitstamp ...")
    handler = CCXTDataHandler(
        SYMBOL, exchange="bitstamp", timeframe="1h",
        start="2019-01-01", limit=1000,
    )
    df = handler.source_ohlcv(SYMBOL)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index_label="datetime")
    return df


def metrics(
    equity: pd.Series,
    portfolio: PortfolioManager | None = None,
    label: str = "",
) -> dict:
    years = max(len(equity) / PERIODS, 1e-12)
    row = {
        "label": label,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "sharpe": annualized_sharpe(equity, periods=PERIODS),
        "max_dd": max_drawdown(equity)["max_drawdown"],
        "calmar": calmar_ratio(equity, periods=PERIODS),
    }
    if portfolio is not None:
        row["trades"] = len(portfolio.trades)
        row["friction"] = float(sum(f.total_cost for f in portfolio.fills))
    return row


def report(row: dict) -> None:
    print(
        f"  {row['label']:<28}  ret={row['total_return']:+.2%}  "
        f"cagr={row['cagr']:+.2%}  sharpe={row['sharpe']:.3f}  "
        f"maxDD={row['max_dd']:.2%}  trades={row.get('trades', '-')}"
    )
