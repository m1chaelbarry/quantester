# Getting Started

This page gets Quantester installed and running on your machine, and walks you
through the two bundled examples so you can see a full backtest — and a full
Monte Carlo validation suite — before writing any code yourself.

## Prerequisites

- **Python 3.12 or newer** (`python --version` to check).
- A virtual environment is strongly recommended:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## Installation

From the repository root:

```bash
pip install -e .[dev]
```

This installs Quantester in editable mode plus the dev tools (`pytest`).
Runtime dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn`, and
`matplotlib` — nothing else.

## Verify the install

Run the test suite (should finish in a few seconds):

```bash
pytest
```

Every module ships with its own tests under `tests/`. A green run means the
engine, ledger, cost models, and validation gates all behave as specified.

## Run the bundled examples

Examples are grouped **one folder per strategy/demo** — see
[`examples/README.md`](../examples/README.md).

### 1. Event-driven backtest: `examples/ma_cross/run.py`

```bash
python examples/ma_cross/run.py
```

This script is the canonical "hello world" of the engine. It:

1. Generates three synthetic OHLCV symbols (`AAA`, `BBB`, `CCC`) and writes
   them as CSVs to `examples/data/` — `BBB` has deliberately missing bars so
   you can see the availability mask in action.
2. Sweeps a moving-average crossover strategy over three parameter pairs,
   logging each trial to the **Trials Registry**.
3. Computes the **Deflated Sharpe Ratio (DSR)** from the registry so the
   "best" result is honestly deflated by the number of trials tried.
4. Checks the **Carver cost-drag speed limit**.
5. Renders a tearsheet to `examples/ma_cross/output/ma_cross_tearsheet.png`.
6. Runs the **truncation test** — the engine's look-ahead leak detector — and
   prints `PASS`.

Expected output looks like:

```
fast= 5 slow=20  sharpe=...  mdd=...  calmar=...
fast=10 slow=40  sharpe=...  mdd=...  calmar=...
fast=20 slow=60  sharpe=...  mdd=...  calmar=...
Best trial: ...
DSR (N=3 trials, registry-driven): ...
Carver cost drag: 0.040 SR/yr (within speed limit)
Tearsheet: examples/ma_cross/output/ma_cross_tearsheet.png
Truncation test [PASS]: compared ... rows after truncating 30 bars; 0 mismatch(es).
```

### 2. Monte Carlo validation: `examples/monte_carlo/run.py`

```bash
python examples/monte_carlo/run.py
```

This runs the five-step Monte Carlo checklist on a backtest:

1. Trade-level resampling (empirical "hat" bootstrap + Ehlers parametric
   randomization).
2. MCPT permutation testing (Masters' p-value + Trend/Bias/Skill partition,
   Protocol I & II permutations).
3. Double-bootstrap maximum drawdown bound.
4. Ornstein-Uhlenbeck synthetic paths + Optimal Trading Rules sweep.
5. Autocorrelation diagnostics gate (runs test + Ljung-Box).

> **Note:** the example uses small replication counts so it finishes quickly.
> For production conclusions, raise `N_REPS` to ≥ 1,000, `N_OUTER` to 10,000,
> and `N_INNER` to 1,000 (the checklist values printed in the script header).

### 3. Multi-coin Donchian dashboard: `examples/donchian_breakout/run_multi_coin_viz.py`

```bash
python examples/donchian_breakout/run_multi_coin_viz.py \
  --universe BTC/USD,ETH/USD,XRP/USD --risk-budget 0.02
```

Daily long-only Donchian across majors with book-level risk budgeting.
Writes `examples/donchian_breakout/output/multi_coin_dashboard.png`.

## Project layout

```
quantester/
├── engine.py            # the synchronous event loop
├── events.py            # MarketEvent / SignalEvent / OrderEvent / FillEvent
├── data/                # DataHandler interface, CSV handler, info-driven bars
├── strategy/            # Strategy interface, examples, meta-labeling
├── portfolio/           # ledger, sizers, spectral risk, margin monitor
├── execution/           # execution simulator, transaction cost models
├── analytics/           # performance stats, tearsheets, DSR, trials registry
├── validation/          # truncation test, PurgedKFold/CPCV, PBO
├── montecarlo/          # fast-track, MCPT, resampling, DD bounds, OU paths
└── utils/               # ETF trick, synthetic OHLCV generator
examples/                # one folder per strategy/demo (+ shared data/)
tests/                   # pytest suite mirroring the modules
docs/                    # this documentation
```

## Next step

Head to [Architecture & Core Concepts](architecture.md) for a ten-minute tour
of how the engine thinks, then build your first strategy in the
[step-by-step tutorial](tutorials/creating-a-strategy.md).
