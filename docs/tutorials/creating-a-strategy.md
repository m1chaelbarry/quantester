# Creating a Strategy & Backtesting It

This tutorial takes you from an empty file to a fully validated backtest in
about ten steps. It is beginner-friendly — every concept is explained as we
meet it — but it follows the same workflow a desk quant would use, including
the anti-overfitting gates at the end.

**What we will build:** a long/flat momentum strategy on a single symbol.
If the close is higher than it was 20 bars ago, we go long; when momentum
fades, we exit. Simple enough to understand completely, real enough to show
every part of the engine.

**Companion script:** every snippet below is assembled, ready to run, in
[`examples/custom_strategy/run.py`](../../examples/custom_strategy/run.py).
Run `python examples/custom_strategy/run.py` from the repo root at any point
to check your work.

**Prerequisites:** [installation done and tests green](../getting-started.md).

---

## Step 0 — Understand the contract

A strategy in Quantester is a class with one job: when the engine says "new
market data," decide whether to trade and, if so, say *what direction* — never
*how much* (sizing is the portfolio's job) and never *at what price* (that is
the execution simulator's job).

Three rules keep every strategy honest:

1. **Only look through the firewall.** Read market data exclusively via
   `data_handler.get_latest_bars(...)` / `get_current_open(...)`. Never read a
   raw DataFrame — the firewall guarantees you cannot see the future.
2. **Only speak in events.** Communicate by putting `SignalEvent`s on the
   queue passed to you. Never call the portfolio or execution handler directly.
3. **Emit only on changes.** Carry your current position on the instance and
   send a signal only when your *target* changes. Re-emitting the same target
   every bar would spam redundant orders.

## Step 1 — Get some data

The engine consumes per-symbol OHLCV data: either CSV files with the header
`datetime,open,high,low,close,volume`, or pre-loaded DataFrames indexed by
datetime. For this tutorial we generate deterministic synthetic data with the
bundled helper — no downloads needed:

```python
from quantester.utils.synthetic import make_synthetic_ohlcv

df = make_synthetic_ohlcv("AAA", n_bars=750, s0=100.0,
                          mu=0.10, sigma=0.22, seed=1)
```

This creates 750 business days of geometric-Brownian-motion prices starting at
$100. To use real data instead, save one CSV per symbol and pass paths:

```python
from quantester.data.csv_handler import HistoricCSVDataHandler

handler = HistoricCSVDataHandler({
    "AAPL": "data/AAPL.csv",     # or a pre-loaded DataFrame
    "MSFT": "data/MSFT.csv",
})
```

Multiple symbols are aligned on the union of all timestamps; a symbol with a
missing bar is simply untradeable at that timestamp (your strategy must
tolerate `None` bars — see Step 3).

## Step 2 — Create the data handler

The data handler is the strategy's *only* window onto the market:

```python
handler = HistoricCSVDataHandler({"AAA": df})
```

It streams bars in time order and enforces the temporal firewall: at a bar's
open phase you can see all prior bars plus the current open print; the current
high/low/close only become visible at the close phase.

## Step 3 — Write the strategy class

Create the class piece by piece.

### 3a. Subclass `Strategy` and configure it

```python
from quantester.events import EXIT, LONG, SignalEvent
from quantester.strategy.base import Strategy


class MomentumStrategy(Strategy):
    """Long when the lookback-bar momentum is positive; flat otherwise."""

    def __init__(self, data_handler, symbol: str, lookback: int = 20):
        self.data_handler = data_handler   # firewall-respecting data window
        self.symbol = symbol               # the one symbol we trade
        self.lookback = lookback           # bars over which momentum is measured
        self.delay = 1                     # signal at close T -> fill at open T+1
        self._position = 0.0               # current target: 1 long, 0 flat
```

About `delay` — this is the temporal-firewall dial:

- `delay = 1` (recommended): signals are computed at bar T's **close** and
  filled at bar T+1's **open**. Safe and realistic.
- `delay = 0`: signals computed at bar T's **open** fill at bar T's open. The
  engine automatically restricts your data view to strictly-before-the-open,
  so you still cannot cheat — but `delay = 1` is the right default while
  learning.

### 3b. Implement `calculate_signals`

This method is called once per bar (close phase, because `delay = 1`):

```python
    def calculate_signals(self, event, events_queue):
        # Rule 1: a missing bar means "untradeable right now" — do nothing.
        if event.bars.get(self.symbol) is None:
            return

        # Ask the firewall for the trailing window it is safe to see.
        bars = self.data_handler.get_latest_bars(self.symbol, self.lookback + 1)
        if len(bars) < self.lookback + 1:
            return  # not enough history yet (start of the series)

        # The signal: close-to-close momentum over the lookback.
        momentum = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0

        # Rule 3: only emit when the TARGET changes.
        if momentum > 0 and self._position <= 0:
            events_queue.put(SignalEvent(event.timestamp, self.symbol,
                                         LONG, strength=1.0, delay=self.delay))
            self._position = 1.0
        elif momentum <= 0 and self._position > 0:
            events_queue.put(SignalEvent(event.timestamp, self.symbol,
                                         EXIT, strength=1.0, delay=self.delay))
            self._position = 0.0
```

That is a complete strategy. Note what it does **not** do: it never says how
many shares to buy, and never touches prices beyond what the firewall served.

### 3c. (Optional, needed for Monte Carlo) the vectorized twin

Monte Carlo validation runs thousands of backtests through a vectorized
fast-track — the event loop is too slow for that. To opt in, implement
`vectorized_signals(data)` returning the target position after every close,
and make it numerically identical to the event form:

```python
    def vectorized_signals(self, data: dict):
        close = data[self.symbol]["close"]
        momentum = close / close.shift(self.lookback) - 1.0
        return {self.symbol: (momentum > 0).astype(float)}
```

A parity test (`tests/test_montecarlo.py`) proves the two forms produce the
same equity curve, so Monte Carlo results actually describe your strategy.
If you skip this, everything in this tutorial still works except the MCPT
section at the end.

## Step 4 — Choose how much to trade (sizing)

Signals say *direction*; **sizers** say *how much*. The portfolio manager
ships with two, and you can write your own:

| Sizer | Behavior |
| --- | --- |
| `PercentEquitySizer(pct)` | Target position worth `pct * equity * strength` — compounds and scales with account size. |
| `FixedUnitSizer(units)` | Fixed number of shares per unit of strength — useful for parity checks and Monte Carlo. |

```python
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager

portfolio = PortfolioManager(
    handler,
    initial_capital=100_000.0,
    sizer=PercentEquitySizer(0.9),       # invest up to 90% of equity
)
```

Advanced sizing engines — Kelly, Gaussian Kelly, volatility parity, Vince's
gap-stressed optimal-f, and the Kakushadze cost-aware adjustment — live in
`quantester/portfolio/sizing.py` and are covered in the
[portfolio reference](../modules/portfolio.md).

## Step 5 — Choose how fills happen (execution & costs)

```python
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler

costs = CostModel(
    fixed_commission=1.0,        # $ per order
    per_share_commission=0.005,  # $ per share
    spread_pct=0.0005,           # full bid-ask spread, 5 bps
    slippage_vol_coef=0.1,       # Kaufman volatility slippage
    impact_coef=0.1,             # Kyle lambda market impact
)
execution = SimulatedExecutionHandler(costs)
```

Every fill price is the bar's open **plus an adverse adjustment** (half-spread
+ volatility slippage + impact) — you always buy a little higher and sell a
little lower than the printed price. This is deliberate: optimistic fills are
the most common way backtests lie. The default `CostModel()` is a reasonable
starting point for liquid US equities.

## Step 6 — Assemble the engine and run

```python
from quantester.engine import BacktestEngine

strategy = MomentumStrategy(handler, "AAA", lookback=20)
engine = BacktestEngine(handler, strategy, portfolio, execution)
engine.run_backtest()
```

`run_backtest()` returns the portfolio, which now holds the complete record of
the run. (Running multiple strategies at once is supported too — pass a list.)

## Step 7 — Read the results

```python
from quantester.analytics.performance import summarize

equity = portfolio.equity_curve            # pd.Series of daily equity
print(summarize(equity))
# {'total_return': ..., 'sharpe': ..., 'max_drawdown': ...,
#  'max_drawdown_duration_days': ..., 'calmar': ...}
```

Useful objects on the portfolio after a run:

| Attribute | Contents |
| --- | --- |
| `equity_curve` | Daily mark-to-market equity (`pd.Series`). |
| `positions_history` | Position per symbol per bar (`pd.DataFrame`). |
| `fills` | Every `FillEvent` — prices, commissions, slippage. |
| `trades` | Completed round-trips with entry/exit prices and realized PnL. |
| `cash`, `positions` | Final ledger state. |

The headline numbers, in words: **total return** is what you made;
**Sharpe** is return per unit of daily volatility (annualized, ×√252);
**max drawdown** is the worst peak-to-trough loss; **Calmar** divides the
annualized return by that drawdown.

## Step 8 — Render a tearsheet

```python
from quantester.analytics.tearsheet import generate_tearsheet

stats = generate_tearsheet(equity, "examples/custom_strategy/output/momentum_tearsheet.png",
                           title="Momentum(20) on AAA")
```

This writes a PNG with the equity curve, the underwater (drawdown) plot, the
return histogram, and a stats box, and returns the stats dict.

## Step 9 — Prove there is no look-ahead (truncation test)

Before trusting *any* backtest, run Ernest Chan's truncation test: re-run the
identical program with the last N bars chopped off. The overlapping positions
must be **bit-identical**; any difference means future data leaked into past
decisions.

```python
from quantester.validation.truncation import run_truncation_test


def run(n=None):
    data = {"AAA": df.iloc[:-n] if n else df}
    h = HistoricCSVDataHandler(data)
    p = PortfolioManager(h, 100_000.0, sizer=PercentEquitySizer(0.9))
    e = BacktestEngine(h, MomentumStrategy(h, "AAA", 20), p,
                       SimulatedExecutionHandler(CostModel()))
    e.run_backtest()
    return p.positions_history


result = run_truncation_test(run, n_truncated=30)
print(result)   # Truncation test [PASS]: ...
```

Because `MomentumStrategy` only reads through the firewall, it passes by
construction — but run it anyway. The day it fails, you have found a leak.

## Step 10 — Is it skill or luck? (validation gates)

A profitable backtest proves nothing by itself. Quantester's full validation
battery is documented in the [Validation Workflow](validation-workflow.md);
here is the short version you can run today:

1. **Parameter sweep + Trials Registry + DSR.** If you tried several lookbacks
   and picked the best, that best is inflated by selection. Log every trial to
   the `TrialsRegistry` and report the **Deflated Sharpe Ratio**, which
   deflates the observed Sharpe by the number of things you tried. See
   `examples/ma_cross/run.py` for a complete sweep-log-DSR loop.
2. **PBO gate.** After any parameter sweep, the CSCV Probability of Backtest
   Overfitting must be `< 0.10`.
3. **MCPT.** Retrain the strategy on thousands of permuted price paths; the
   original must beat at least 95% of them (`p < 0.05`). This uses the
   vectorized twin from Step 3c — the companion script shows the full loop.
4. **Autocorrelation gate first.** If returns are serially correlated, iid
   resampling is invalid — use the block bootstrap or O-U paths instead.
   `autocorrelation_gate(returns)` tells you which.

## The complete script

Everything above, assembled and runnable, is in
[`examples/custom_strategy/run.py`](../../examples/custom_strategy/run.py):

```bash
python examples/custom_strategy/run.py
```

Expected console output (deterministic — the data generator is seeded):

```
========================================================================
Quantester tutorial: momentum strategy from scratch
========================================================================
Backtest: total return -14.17%  sharpe -0.394  max DD -28.37%  calmar -0.176
Trades: 44 round-trips, 88 fills
Tearsheet written to examples/custom_strategy/output/momentum_tearsheet.png
Truncation test [PASS]: compared 720 rows after truncating 30 bars; 0 mismatch(es).
Fast-track parity: max |equity diff| = 4.37e-10
MCPT p-value (200 reps): 1.0000 (not significant)
```

**Read this output like a practitioner.** The strategy *loses* money — and
that is the correct answer, not a bug. The synthetic data is a geometric
Brownian motion: it has drift but no persistent trend, so there is no real
momentum to harvest, and transaction costs turn the coin-flip entries into a
steady bleed. The validation stack tells you exactly that: the truncation
test proves the loss is not a look-ahead artifact, the fast-track parity
check proves the vectorized twin is faithful, and the MCPT p-value of 1.0
says the strategy beats **none** of the permuted (pattern-free) markets — no
skill detected. The gates doing their job *is* the successful outcome of this
tutorial. When you evaluate your own strategy on real data, you want the same
machinery to light up green.

## Common mistakes (and how the engine catches them)

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Reading a raw DataFrame instead of `get_latest_bars` | Truncation test FAILS; results too good | Only read through the data handler. |
| Emitting the same signal every bar | Hundreds of fills, huge commissions | Keep `_position` on the instance; emit only on target changes. |
| Trading a `None` bar | `TypeError`/`KeyError` inside your strategy | Guard with `event.bars.get(symbol) is None`. |
| Dividing by the wrong price when sizing | `ref_price <= 0` guards make targets zero | Let a sizer do it — it receives `ref_price` from the portfolio. |
| Assuming `signal.strength` affects direction | Negative/zero strength silently flips/shrinks targets | Keep `strength ∈ (0, 1]`; it scales size only. |
| Skipping the vectorized twin then running MCPT | `NotImplementedError` | Implement Step 3c, or use event-loop validation only. |

## Where to go next

- Give your strategy a **secondary ML model** that scales position size by the
  probability the primary signal is right:
  [meta-labeling](../modules/strategy.md#meta-labeling).
- Add **margin monitoring** and spectral risk attribution:
  [portfolio reference](../modules/portfolio.md).
- Feed **dollar bars or imbalance bars** instead of time bars:
  [data reference](../modules/data.md#information-driven-bars).
- Run the **full Monte Carlo suite**:
  `python examples/monte_carlo/run.py` and the
  [Monte Carlo reference](../modules/montecarlo.md).
