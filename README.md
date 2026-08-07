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
python examples/run_ma_cross.py      # backtest + tearsheet + truncation check
python examples/run_monte_carlo.py   # Monte Carlo validation suite
```

## Market data providers

All feeds share one streaming implementation, so the temporal firewall and
availability-mask semantics are identical regardless of source:

- `HistoricCSVDataHandler` — local `datetime,open,high,low,close,volume` CSVs.
- `YFinanceDataHandler` — free Yahoo Finance OHLCV
  (`pip install "quantester[yfinance]"`).
- `CCXTDataHandler` — free OHLCV from 100+ crypto exchanges via ccxt
  (`pip install "quantester[ccxt]"`, or `quantester[data]` for both).

```bash
python examples/run_market_data.py   # live-data backtest via yfinance + ccxt
```

## Source verification status

Core formulas (ETF trick, CPCV purging, CSCV PBO, Masters permutation p-value
and Trend/Bias/Skill partition, Vince optimal-f, imbalance bars, meta-labeling,
Carver cost drag) were verified against specialist quant literature. Items
flagged in module docstrings as not covered by that literature (DSR's exact
form, Kyle lambda estimation, Protocol II details, Ehlers randomization) follow
their canonical paper/report sources.
