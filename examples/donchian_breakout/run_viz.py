"""Visualization suite for the hourly BTC Donchian breakout strategy.

Runs the event-driven backtest on cached Bitstamp 1h BTC/USD, then renders:
  1. Candles with SMA200/SMA20, Donchian(20), trail(10), fills & round-trips
  2. Same window with ADX + ATR subpanels and held position
  3. Equity / drawdown / position history
  4. Round-trip trade analysis
  5. Monthly returns heatmap
  6. Rolling Sharpe / vol / drawdown
  7. Interactive viewer headless snapshot (zoom on recent activity)

Run from the repo root:
  python examples/donchian_breakout/run_viz.py
  python examples/donchian_breakout/run_viz.py --bars 2500 --candle-bars 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "examples" / "data"
OUTPUT_DIR = HERE / "output"

import pandas as pd

from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import ConservativeFrictionCostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import FractionalRiskSizer, PortfolioManager
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.visualization import (
    indicators,
    interactive_view,
    plot_candles,
    plot_equity,
    plot_monthly_returns,
    plot_rolling_metrics,
    plot_trade_analysis,
    trade_stats,
)

CACHE = DATA_DIR / "BTCUSD_bitstamp_1h.csv"
SYMBOL = "BTC/USD"
INITIAL_CAPITAL = 25_000.0
FRICTION = ConservativeFrictionCostModel(spread_pct=0.0002, fee_rate=0.0004)


def load_data() -> pd.DataFrame:
    if not CACHE.exists():
        from quantester.data.ccxt_handler import CCXTDataHandler

        print("Fetching BTC/USD 1h from Bitstamp ...")
        handler = CCXTDataHandler(
            SYMBOL, exchange="bitstamp", timeframe="1h",
            start="2024-01-01", limit=1000,
        )
        df = handler._data[SYMBOL]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index_label="datetime")
        return df
    return pd.read_csv(CACHE, parse_dates=["datetime"], index_col="datetime")


def run_backtest(df: pd.DataFrame) -> PortfolioManager:
    handler = HistoricCSVDataHandler({SYMBOL: df})
    strategy = DonchianBreakoutStrategy(handler, SYMBOL)
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=FractionalRiskSizer(0.02),
    )
    BacktestEngine(
        handler, strategy, portfolio, SimulatedExecutionHandler(FRICTION),
    ).run_backtest()
    return portfolio


def _in_window(rows: list, start, end) -> list:
    out = []
    for row in rows:
        if isinstance(row, dict):
            t0, t1 = row.get("t0"), row.get("t1")
            if t0 is not None and t1 is not None and t1 >= start and t0 <= end:
                out.append(row)
        else:
            # FillEvent
            if start <= row.timestamp <= end:
                out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=2500,
                        help="Trailing hourly bars for the backtest window")
    parser.add_argument("--candle-bars", type=int, default=600,
                        help="Trailing bars shown on the candlestick charts")
    args = parser.parse_args()

    print("=" * 72)
    print("Donchian breakout — visualization suite (BTC/USD 1h)")
    print("=" * 72)

    full = load_data()
    # Keep extra history so SMA200 / ADX are warm at the window start.
    warmup = 250
    need = args.bars + warmup
    data = full.iloc[-need:].copy() if len(full) > need else full.copy()
    window = data.iloc[-args.bars:].copy()

    print(f"Cache {len(full)} bars; backtest window {len(window)}  "
          f"{window.index[0]} → {window.index[-1]}")
    portfolio = run_backtest(data)
    # Restrict artifacts to the analysis window (drop pure-warmup fills).
    equity = portfolio.equity_curve.loc[window.index[0]:]
    positions = portfolio.positions_history
    if not positions.empty:
        positions = positions.loc[window.index[0]:]
    trades = _in_window(portfolio.trades, window.index[0], window.index[-1])
    fills = _in_window(portfolio.fills, window.index[0], window.index[-1])
    print(f"Backtest done: {len(fills)} fills, {len(trades)} round trips "
          f"in window (equity bars={len(equity)}).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close = data["close"]
    high, low = data["high"], data["low"]

    # Indicators on the warm series; slice to the candle display window.
    sma200 = indicators.sma(close, 200)
    sma20 = indicators.sma(close, 20)
    channel = indicators.donchian(high, low, window=20, shift=1)
    trail = indicators.donchian(high, low, window=10, shift=1)
    adx = indicators.adx(high, low, close, window=14)
    atr = indicators.atr(high, low, close, window=14)

    candle = data.iloc[-args.candle_bars:].copy()
    c0, c1 = candle.index[0], candle.index[-1]
    candle_trades = _in_window(trades, c0, c1)
    candle_fills = _in_window(fills, c0, c1)
    held = None
    if not positions.empty and SYMBOL in positions.columns:
        held = positions[SYMBOL].reindex(candle.index).ffill().fillna(0.0)

    overlays = {
        "SMA(200)": sma200,
        "SMA(20)": sma20,
        "Donchian(20)": channel,
        "Trail(10)": trail,
    }
    subpanels = {
        "ADX(14)": adx["adx"],
        "ATR(14)": atr,
    }

    out = OUTPUT_DIR / "donchian_candles_trades.png"
    plot_candles(
        candle, overlays=overlays,
        trades=candle_trades, fills=candle_fills,
        title=f"BTC/USD 1h — Donchian breakout (last {len(candle)} bars)",
        path=out, figsize=(14, 8),
    )
    print(f"1. {out}")

    out = OUTPUT_DIR / "donchian_candles_adx_position.png"
    plot_candles(
        candle, overlays=overlays, subpanels=subpanels,
        positions=held, trades=candle_trades, fills=candle_fills,
        title="Donchian breakout — ADX/ATR filters + held position",
        path=out, figsize=(14, 11),
    )
    print(f"2. {out}")

    out = OUTPUT_DIR / "donchian_equity_positions.png"
    plot_equity(
        equity, positions_history=positions,
        title="Donchian breakout — equity, drawdown, exposure",
        path=out,
    )
    print(f"3. {out}")

    out = OUTPUT_DIR / "donchian_trade_analysis.png"
    if trades:
        stats = trade_stats(trades)
        print(f"   trade_stats: n={stats.get('n_trades')}  "
              f"win_rate={stats.get('win_rate', float('nan')):.1%}  "
              f"expectancy={stats.get('expectancy', float('nan')):.2f}  "
              f"pf={stats.get('profit_factor', float('nan')):.2f}")
        plot_trade_analysis(trades, title="Donchian breakout — round trips",
                            path=out)
    else:
        print("   no round trips in window — skipping trade analysis chart")
    print(f"4. {out}")

    out = OUTPUT_DIR / "donchian_monthly_returns.png"
    plot_monthly_returns(equity, title="Donchian breakout — monthly returns (%)",
                         path=out)
    print(f"5. {out}")

    # ~2 weeks of hourly bars for rolling window on this sample.
    roll = min(336, max(48, len(equity) // 5))
    out = OUTPUT_DIR / "donchian_rolling_metrics.png"
    plot_rolling_metrics(
        equity, window=roll, periods=24 * 365,
        title=f"Donchian breakout — rolling metrics (window={roll}h)",
        path=out,
    )
    print(f"6. {out}")

    viewer = interactive_view(
        candle,
        overlays={"SMA(200)": sma200, "SMA(20)": sma20, "Donchian(20)": channel},
        subpanels={"ADX(14)": adx["adx"]},
        equity=equity.reindex(candle.index).dropna(),
        positions=held,
        trades=candle_trades,
        fills=candle_fills,
        title="Donchian breakout — interactive snapshot",
    )
    out = OUTPUT_DIR / "donchian_interactive_snapshot.png"
    viewer.zoom(0.45)
    viewer.pan(-0.25)
    viewer.save(out)
    print(f"7. {out} (headless snapshot)")
    print("\nCharts written under examples/donchian_breakout/output/. "
          "For live scroll/zoom, run locally with QtAgg/TkAgg and viewer.show().")


if __name__ == "__main__":
    main()
