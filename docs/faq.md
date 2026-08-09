# FAQ & Troubleshooting

## Getting started

**`python` is not found.**
Use `python3` (or activate your virtualenv). The engine requires Python ≥ 3.12.

**`ModuleNotFoundError: quantester`.**
Install editable from the repo root: `pip install -e .[dev]`. Tests also work
without install because `pyproject.toml` sets `pythonpath = ["."]` for pytest.

## Writing strategies

**My strategy never trades.**
Check, in order: (1) your warmup guard — `get_latest_bars(symbol, n)` returns
fewer than `n` rows at the start of the series; (2) the availability guard —
`event.bars.get(symbol) is None` on missing bars; (3) `delay=1` strategies
only run at the close phase, and an order generated on the **last** bar is
dropped because no future bar exists to fill it.

**My strategy trades every bar and commissions eat everything.**
You are re-emitting the same target each bar. Use `self.emit_target(...)` (or
keep `self._position` and only `put` a `SignalEvent` when the target changes).
See `MovingAverageCrossStrategy` and `docs/for-traders.md`.

**`NotImplementedError: ... does not provide a vectorized twin`.**
You called a Monte Carlo fast-track function on a strategy without
`vectorized_signals`. Implement it (numerically identical to the event form)
or use event-loop validation only.

**Can a strategy trade multiple symbols?**
Yes — iterate `event.bars` and emit one `SignalEvent` per symbol. You can
also pass a *list* of strategies to `BacktestEngine`.

**How do I use `strength`?**
It scales the sizer's target (e.g. meta-labeling sets it to the secondary
model's probability). Keep it in `(0, 1]`; direction comes from
`signal_type`, not from the sign of `strength`.

## Data

**How are symbols with different calendars handled?**
Outer join: the master calendar is the union of all timestamps. A symbol
missing a bar is `None` in `event.bars` (untradeable at that timestamp) — the
timestamp is kept for the other symbols. Nothing is ever erased.

**How do I bring my own data?**
One CSV per symbol with header `datetime,open,high,low,close,volume`, or
pre-loaded DataFrames indexed by datetime, into `HistoricCSVDataHandler`. For
tick data, build bars first with `quantester/data/bars.py`.

## Results & validation

**The truncation test FAILED. What now?**
Your pipeline leaks future data. Look for: raw DataFrame reads that bypass
`get_latest_bars`, indicators computed on the full series before the run,
or normalization using full-sample statistics. `result.mismatches` lists the
first offending timestamps/symbols.

**Why did my stop order fill below the stop price?**
By design. A stop gapped through at the open fills at the open — the next
available price — never at the guaranteed stop. Perfect stops would understate
tail risk and silently unbound optimal-f sizing.

**Why is slippage not subtracted from cash a second time?**
`fill_price` is already all-in (reference ± spread/slippage/impact).
`slippage_cost` (φₜ) is recorded for cost analytics only; only `commission`
(cₜ) is charged separately. Double-deducting φₜ would misstate the ledger.

**DSR asks for N trials — can I just pass the number I remember?**
Don't. The registry (`TrialsRegistry`) exists precisely because humans
undercount trials. Log everything, including failed runs, and use
`dsr_from_registry`.

**MCPT is slow.**
It must retrain on every permuted path — that is the point. Make sure your
optimizer runs on the vectorized fast-track (`fast_backtest` +
`vectorized_signals`), never the event loop, and lower `n_reps` while
iterating (raise to ≥ 1,000 for conclusions).

**When is iid resampling invalid?**
When `autocorrelation_gate(returns)` reports serial correlation. Use
`empirical_resample(..., block_length=L)` or OU synthetic paths instead —
otherwise simulated paths are artificially smooth and downside risk is
underestimated.
