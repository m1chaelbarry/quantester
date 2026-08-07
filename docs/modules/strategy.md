# Strategy

Package: `quantester/strategy`

A strategy is a sandboxed, state-free-*looking* mathematical engine: it
consumes `MarketEvent`s and emits `SignalEvent`s. "Sandboxed" means it
interacts exclusively with standardized containers (bars in, signals out), so
the same code can run in research and in production.

For a guided build, start with the
[step-by-step tutorial](../tutorials/creating-a-strategy.md). This page is the
reference.

## `Strategy` (abstract interface)

`quantester/strategy/base.py`.

```python
class Strategy(ABC):
    delay: int = 1          # bars until execution (1 = T+1, 0 = Delay-0)
    fill_at: str = "open"   # reference price used for sizing

    def calculate_signals(self, event, events_queue): ...   # abstract
    def matches_phase(self, phase) -> bool: ...             # delay routing
    def vectorized_signals(self, data) -> dict: ...         # MC twin (optional)
```

| Member | Contract |
| --- | --- |
| `delay` | `1` (default): act on close-phase events, fill at next bar's open. `0`: act on open-phase events, fill at the current open under the intra-bar guard. |
| `matches_phase(phase)` | Engine-internal routing: delay-0 strategies only receive open-phase events; delay-≥1 only close-phase. Override only if you know why. |
| `calculate_signals(event, queue)` | Your logic. Read via `event.bars` + `data_handler.get_latest_bars`; write via `queue.put(SignalEvent(...))`. Emit **only on target changes**. |
| `vectorized_signals(data)` | Full-history target positions per symbol for the Monte Carlo fast-track. Must be numerically identical to the event form (share one pure function). Raises `NotImplementedError` by default. |

### The authoring contract (checklist)

1. Subclass `Strategy`; set `delay` deliberately.
2. Guard every bar access: `if event.bars.get(symbol) is None: return`.
3. Keep current-position state on the instance; emit only when the target
   changes.
4. Never read raw DataFrames — only the DataHandler interface.
5. If Monte Carlo validation matters, implement `vectorized_signals` and keep
   the parity test green.

## Bundled example strategies

`quantester/strategy/examples.py`.

### `BuyAndHoldStrategy(data_handler)`

Enters `LONG` every symbol once, at the first bar where the symbol is
tradeable. Vectorized twin: constant `+1` target. Useful as a benchmark and
in tests.

### `MovingAverageCrossStrategy(data_handler, symbol, fast=10, slow=30, direction="both", delay=1)`

SMA crossover on one symbol: long when the fast SMA crosses above the slow
SMA, short (or flat) on the downward cross.

- `direction`: `"both"` (long+short), `"long"` (long/flat), `"short"`
  (short/flat).
- Emits only on crosses; holds state between them.
- Shares the pure function `crossover_positions(close, fast, slow)` between
  its event form and vectorized twin, so fast-track parity holds by
  construction. **Copy this pattern** when you write your own strategy with a
  twin.

```python
from quantester.strategy.examples import MovingAverageCrossStrategy

strat = MovingAverageCrossStrategy(handler, "AAPL", fast=10, slow=40,
                                   direction="long")
```

## Meta-labeling

`quantester/strategy/meta_labeling.py` — the AFML ch. 3 scaffold.

The **primary model** decides the trade *side*; a **secondary binary
classifier** predicts the probability that the primary model is right, and
that probability scales the trade *size* (via `SignalEvent.strength`). This
filters false positives: precision rises at the cost of recall.

### `triple_barrier_labels(close, events, tp_pct, sl_pct, max_holding)`

Builds the training labels for the secondary model. `events` is a DataFrame
with columns `t0` (entry timestamp) and `side` (+1/−1). For each event the
outcome is decided by three barriers — take-profit at `entry·(1+side·tp_pct)`,
stop-loss at `entry·(1−side·sl_pct)`, and a vertical barrier `max_holding`
bars out. Label `y=1` when the primary signal was correct (TP touched first,
or positive realization at the vertical barrier), else `y=0`.

### `MetaLabelingStrategy(primary, data_handler, model=None, feature_fn=None, size_transform="linear")`

Wraps any primary `Strategy`. Intercepts its signals and re-emits them with
`strength` set from the secondary model's predicted probability:

- `model`: any sklearn-style estimator with `fit` / `predict_proba`. `None`
  passes primary signals through unchanged.
- `feature_fn(data_handler, signal) -> features`: builds the feature vector
  for one intercepted signal.
- `size_transform`: `"linear"` (strength = probability) or `"zscore"`
  (strength = `2·Φ(z) − 1` with `z = (p−0.5)/√(p(1−p))`, AFML ch. 10 — flagged
  as not notebook-verified).
- `fit_secondary(features, close, events, tp_pct, sl_pct, max_holding)`:
  builds triple-barrier labels and fits the secondary model.

```python
primary = MovingAverageCrossStrategy(handler, "AAPL", fast=10, slow=40)
meta = MetaLabelingStrategy(primary, handler, model=GradientBoostingClassifier(),
                            feature_fn=my_features)
# train on historical events first:
meta.fit_secondary(X_train, close, events, tp_pct=0.02, sl_pct=0.02,
                   max_holding=10)
```

> **Validation requirement:** meta-labeling trains a model, so run it through
> CPCV with label-overlap purging + embargo (see
> [validation](validation.md)) — plain k-fold leaks label overlap into the
> training set.
