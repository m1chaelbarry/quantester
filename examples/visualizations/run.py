"""Visualization example: static charts + scrollable interactive viewer.

Runs one MA-cross backtest, then renders:
  1. candlestick chart with SMA overlays, RSI/MACD subpanels, fills & trades
  2. strategy-development view: indicators + the vectorized twin's target
     positions (what the strategy WANTED to hold, bar by bar)
  3. equity / drawdown / per-symbol positions
  4. round-trip trade analysis
  5. monthly returns heatmap
  6. rolling Sharpe / volatility / drawdown
  7. Monte Carlo fan chart from O-U synthetic paths fitted to the series
  8. interactive viewer snapshot (headless); run with an interactive backend
     (Qt/Tk, or `%matplotlib widget` in a notebook) to scroll, zoom, and pan

Run from the repo root:  python examples/visualizations/run.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.synthetic import estimate_ou_params, generate_ou_paths
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy
from quantester.utils.synthetic import make_synthetic_ohlcv
from quantester.visualization import (
    indicators,
    interactive_view,
    plot_candles,
    plot_equity,
    plot_monthly_returns,
    plot_path_distribution,
    plot_rolling_metrics,
    plot_trade_analysis,
)

INITIAL_CAPITAL = 100_000.0


def main():
    print("=" * 72)
    print("Quantester example: backtest visualization suite")
    print("=" * 72)

    bars = make_synthetic_ohlcv("AAA", seed=1, mu=0.10, sigma=0.22)
    close = bars["close"]

    handler = HistoricCSVDataHandler({"AAA": bars})
    strategy = MovingAverageCrossStrategy(handler, "AAA", fast=10, slow=40,
                                          direction="both")
    portfolio = PortfolioManager(handler, INITIAL_CAPITAL,
                                 sizer=PercentEquitySizer(0.9))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(CostModel()))
    engine.run_backtest()
    print(f"Backtest done: {len(portfolio.fills)} fills, "
          f"{len(portfolio.trades)} round trips.")

    overlays = {
        "SMA(10)": indicators.sma(close, 10),
        "SMA(40)": indicators.sma(close, 40),
        "Bollinger(20,2)": indicators.bollinger_bands(close, 20, 2.0),
    }
    subpanels = {
        "RSI(14)": indicators.rsi(close, 14),
        "MACD(12,26,9)": indicators.macd(close),
    }

    out = OUTPUT_DIR / "candles_indicators_trades.png"
    plot_candles(bars, overlays=overlays, subpanels=subpanels,
                 trades=portfolio.trades, fills=portfolio.fills,
                 title="AAA — MA(10/40) cross with fills & round trips",
                 path=out)
    print(f"1. {out}")

    targets = strategy.vectorized_signals({"AAA": bars})["AAA"]
    out = OUTPUT_DIR / "strategy_targets.png"
    plot_candles(bars, overlays=overlays, subpanels=subpanels,
                 positions=targets,
                 title="Strategy development: indicators + target positions",
                 path=out)
    print(f"2. {out}")

    out = OUTPUT_DIR / "equity_positions.png"
    plot_equity(portfolio.equity_curve,
                positions_history=portfolio.positions_history,
                title="Equity, drawdown, and held quantity", path=out)
    print(f"3. {out}")

    out = OUTPUT_DIR / "trade_analysis.png"
    plot_trade_analysis(portfolio.trades, title="AAA round-trip analysis",
                        path=out)
    print(f"4. {out}")

    out = OUTPUT_DIR / "monthly_returns.png"
    plot_monthly_returns(portfolio.equity_curve, path=out)
    print(f"5. {out}")

    out = OUTPUT_DIR / "rolling_metrics.png"
    plot_rolling_metrics(portfolio.equity_curve, window=63, path=out)
    print(f"6. {out}")

    ou = estimate_ou_params(close.to_numpy())
    paths = generate_ou_paths(ou, p0=float(close.iloc[-1]), n_steps=252,
                              n_paths=500, seed=7)
    out = OUTPUT_DIR / "mc_path_distribution.png"
    plot_path_distribution(paths, title="O-U synthetic paths from last close",
                           path=out)
    print(f"7. {out}")

    viewer = interactive_view(
        bars, overlays=overlays, subpanels=subpanels,
        equity=portfolio.equity_curve, positions=targets,
        trades=portfolio.trades, fills=portfolio.fills,
        title="AAA — scroll / drag / arrow keys / hover crosshair",
    )
    out = OUTPUT_DIR / "interactive_view_snapshot.png"
    viewer.zoom(0.35)  # snapshot a zoomed window, not the full squeeze
    viewer.pan(-0.5)
    viewer.save(out)
    print(f"8. {out} (headless snapshot)")
    print("\nTo explore interactively: run this script locally with an\n"
          "interactive matplotlib backend (QtAgg/TkAgg, or `%matplotlib widget`\n"
          "in a notebook) and call viewer.show() — wheel zooms, drag pans,\n"
          "arrows/home/end navigate, hovering shows the OHLCV readout.")


if __name__ == "__main__":
    main()
