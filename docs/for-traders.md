# Quantester for traders

This page is written for someone who trades (or researches strategies) and
does **not** want to learn software architecture first. Quantester is still a
Python library — you will write a little code — but the happy path is short.

## The 60-second mental model

| Trading idea | Quantester piece |
| --- | --- |
| Price history (OHLCV) | **Data** (`HistoricCSVDataHandler`, or just a DataFrame) |
| When to buy / sell / stay flat | **Strategy** (your rules) |
| How large a position | **Sizer** (e.g. 90% of equity) |
| Spreads, commissions, slippage | **Cost model** |
| The simulation clock | **Engine** (you usually do not touch this) |

Signals say *direction*. Sizers say *how much*. Costs make fills realistic.
You never invent a fill price yourself.

## Hello path (copy this)

```bash
python examples/hello_trader/run.py
```

Or in your own file:

```python
from quantester import MovingAverageCrossStrategy, run_backtest
from quantester.utils.synthetic import make_synthetic_ohlcv

prices = make_synthetic_ohlcv("AAA", seed=1)
result = run_backtest(
    prices,
    MovingAverageCrossStrategy,
    symbol="AAA",
    fast=10,
    slow=40,
    capital=100_000,
    equity_pct=0.9,
)
result.print_summary()
```

`run_backtest` wires the five internal modules for you and returns a
`BacktestResult` with `.equity`, `.trades`, `.sharpe`, and `.print_summary()`.

## Writing your own rules

Subclass `Strategy`, read prices only through `data_handler.get_latest_bars`,
and emit signals only when your target **changes**. Prefer
`self.emit_target(...)` — it prevents the common "trade every bar and die to
commissions" mistake:

```python
from quantester import Strategy

class MyMomentum(Strategy):
    def __init__(self, data_handler, symbol, lookback=20):
        self.data_handler = data_handler
        self.symbol = symbol
        self.lookback = lookback
        self.delay = 1          # signal at today's close → fill tomorrow's open
        self._position = 0.0

    def calculate_signals(self, event, events_queue):
        if event.bars.get(self.symbol) is None:
            return
        bars = self.data_handler.get_latest_bars(self.symbol, self.lookback + 1)
        if len(bars) < self.lookback + 1:
            return
        momentum = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0
        target = 1.0 if momentum > 0 else 0.0
        self.emit_target(events_queue, event.timestamp, self.symbol, target)
```

Full walkthrough: [Creating a Strategy](tutorials/creating-a-strategy.md).

## Reading the scoreboard

| Number | Plain meaning |
| --- | --- |
| **Total return** | What the account made or lost |
| **Sharpe** | Return per unit of volatility (higher better; near 0 ≈ no edge) |
| **Max drawdown** | Worst peak-to-trough loss |
| **Calmar** | Return divided by that drawdown |
| **Trades / fills** | How often you turned over |

A pretty equity curve is not enough. Before trusting a strategy, run at least:

1. **Truncation test** — chops the last bars; overlapping positions must match
   (look-ahead detector).
2. **Trials registry + DSR** — if you tried many parameter sets, deflate the
   "best" Sharpe by how many you tried.
3. **MCPT** — does the idea beat scrambled (edge-free) markets?

See [Validation Workflow](tutorials/validation-workflow.md) and
`examples/production_research/`.

## Common mistakes (and the error you will see)

| Mistake | What happens |
| --- | --- |
| `fast >= slow` on an MA cross | Clear `ValueError` with an example |
| `equity_pct=90` instead of `0.9` | `ValueError`: must be in `(0, 1]` |
| `spread_pct=5` meaning "5 bps" | `ValueError`: use `0.0005`, not `5` |
| Pass an already-built strategy into `run_backtest` | `TypeError` explaining class vs factory |
| Re-emit the same target every bar | Huge fill count / commissions (use `emit_target`) |
| Read a raw DataFrame inside the strategy | Truncation test **FAIL** |

## Where to go next

| Goal | Start here |
| --- | --- |
| Shortest script | `examples/hello_trader/run.py` |
| Sweep + DSR + tearsheet | `examples/ma_cross/run.py` |
| Build a strategy from scratch | `examples/custom_strategy/run.py` |
| Real Yahoo / crypto data | `examples/market_data/run.py` |
| Full research checklist | `examples/production_research/` |
| Word definitions (Sharpe, DSR, PBO, …) | [Glossary](glossary.md) |
