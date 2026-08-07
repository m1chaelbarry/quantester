# Quantester

Institutional-grade, event-driven quantitative backtesting engine in pure Python.

## Architecture

Five decoupled modules communicating through a strict four-event lifecycle
(`MarketEvent -> SignalEvent -> OrderEvent -> FillEvent`) via a centralized
synchronous queue:

| Module | Package | Role |
| --- | --- | --- |
| Data Handler | `quantester/data` | Point-in-time market data stream (look-ahead firewall) |
| Strategy | `quantester/strategy` | Signal generation; meta-labeling scaffold |
| Portfolio Manager | `quantester/portfolio` | Ledger, sizing (Kelly / vol-parity / optimal-f), spectral risk, margin |
| Execution Simulator | `quantester/execution` | Temporal-firewall fills, cost models (Kyle lambda, Kaufman, spread) |
| Performance Analytics | `quantester/analytics` | Sharpe/MDD/Calmar, cost drag, trials registry, DSR, tearsheets |

Plus `quantester/validation` (truncation test, CPCV, CSCV PBO) and
`quantester/montecarlo` (vectorized fast-track, trade resampling, MCPT
permutation testing, drawdown double bootstrap, O-U synthetic paths).

## Key guarantees

- **State-Based Temporal Firewall**: strategies declare `delay` (0 or 1 bars);
  orders carry `earliest_fill_time` enforced by the execution ledger. `delay=0`
  fills at the bar's open under an intra-bar visibility guard (data strictly
  before the fill timestamp).
- **No silent history rewrites**: multi-symbol data aligns on an outer-join
  timestamp union; missing bars mark the asset untradeable, never erased.
- **Anti-overfitting gates**: Purged/embargoed combinatorial CV, CSCV PBO gate
  (< 0.10), registry-driven Deflated Sharpe Ratio, truncation regression test.
- **Monte Carlo at scale**: 10,000-rep permutation tests run on a vectorized
  fast-track; a parity test proves it matches the event engine exactly.

## Quick start

```bash
pip install -e .[dev]
pytest
python examples/run_ma_cross.py         # backtest + tearsheet + truncation check
python examples/run_monte_carlo.py      # Monte Carlo validation suite
python examples/run_visualizations.py   # chart gallery + interactive viewer demo
```

## Visualization

`quantester/visualization` renders post-run artifacts (bars, indicators,
equity, fills, trades) — display tooling only, never inside the event loop.

- **Static charts**: candlesticks with indicator overlays/subpanels, fill and
  round-trip markers (`plot_candles`); equity + drawdown + held quantities
  (`plot_equity`); round-trip PnL breakdown (`plot_trade_analysis`); monthly
  returns heatmap (`plot_monthly_returns`); rolling Sharpe/vol/drawdown
  (`plot_rolling_metrics`); Monte Carlo percentile fan + terminal histogram
  (`plot_path_distribution`).
- **Interactive viewer** (`interactive_view`): a scrollable matplotlib chart —
  mouse wheel zooms around the cursor, left-drag pans, arrow keys navigate,
  hovering shows an OHLCV crosshair readout. Works on any interactive backend
  (Qt/Tk/notebook); headless runs save snapshots via `viewer.save(path)`.
- **Indicator helpers** (`visualization.indicators`): SMA, EMA, RSI, MACD,
  Bollinger Bands, ATR, rolling volatility for overlay/subpanel series.

```python
from quantester.visualization import indicators, interactive_view, plot_candles

plot_candles(bars, overlays={"SMA(10)": indicators.sma(bars["close"], 10)},
             subpanels={"RSI(14)": indicators.rsi(bars["close"])},
             trades=portfolio.trades, fills=portfolio.fills,
             path="chart.png")

viewer = interactive_view(bars, equity=portfolio.equity_curve,
                          trades=portfolio.trades)
viewer.show()   # interactive backend: scroll/drag/keys
```

## Source verification status

Core formulas (ETF trick, CPCV purging, CSCV PBO, Masters permutation p-value
and Trend/Bias/Skill partition, Vince optimal-f, imbalance bars, meta-labeling,
Carver cost drag) were verified against specialist quant literature. Items
flagged in module docstrings as not covered by that literature (DSR's exact
form, Kyle lambda estimation, Protocol II details, Ehlers randomization) follow
their canonical paper/report sources.
