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

## Tranche pullback ladder

`quantester/strategy/tranche_pullback.py`.

### `TranchePullbackStrategy(data_handler, symbol, regime_window=200, peak_window=20, atr_window=14, atr_spacing=1.5, tranche_fractions=(0.25, 0.35, 0.40), exit_window=5, stop_atr_mult=5.0, reanchor_every=1, cooldown_bars=0)`

Volatility-spaced dip-buying ladder for a single symbol (built for BTC):

- **Regime gate**: arms only while `close > SMA(regime_window)`.
- **Re-anchoring ladder**: while no tranche is filled, three resting `LIMIT`
  buys at `T_k = peak − k·atr_spacing·ATR` (peak = rolling `peak_window`
  close high, Wilder ATR) are canceled and re-anchored every `reanchor_every`
  bars (default 1 = every bar on daily data); losing the bull regime pulls
  the unfilled ladder. The **first fill freezes** the levels until the
  position is completely closed. On intraday ports set `reanchor_every` to
  bars-per-day and `cooldown_bars` to bars-per-day − 1 so the ladder still
  refreshes on a daily cadence while fills/stops resolve every bar.
- **Sizing**: tranche fractions ride on `SignalEvent.strength` against
  `limit_price=T_k`, so wire `PortfolioManager(sizer=PercentEquitySizer(1.0))`
  for the exact `q_k = equity·f_k/T_k` mapping.
- **Exits**: mean reversion at `close ≥ SMA(exit_window)` (next bar's open);
  hard stop at `peak − stop_atr_mult·ATR` executed per Kaufman's
  close-execution rule (notebook-verified): the intra-bar low triggers a
  market-on-close exit at that bar's close.
- `delay=1` (close-phase), no vectorized twin — the latched, path-dependent
  state machine has no closed form, so validate it with the block-bootstrap
  harness (see [montecarlo](montecarlo.md#stationary-block-bootstrap-ohlcv))
  rather than the fast-track.

```python
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.portfolio.risk import DailyDrawdownBreaker
from quantester.execution.costs import ConservativeFrictionCostModel

portfolio = PortfolioManager(handler, 25_000.0, sizer=PercentEquitySizer(1.0),
                             drawdown_breaker=DailyDrawdownBreaker(0.045))
engine = BacktestEngine(handler, TranchePullbackStrategy(handler, "BTC/USD"),
                        portfolio,
                        SimulatedExecutionHandler(ConservativeFrictionCostModel()))
```

Full evaluation on real CCXT data: `examples/tranche_pullback/run_ccxt.py`;
parameter study with PBO/DSR gating and bootstrap MC:
`examples/tranche_pullback/run_parameter_study.py` (daily) and
`examples/tranche_pullback/run_parameter_study_intraday.py --tf 4h|1h`
(calendar-scaled intraday ports).

## Donchian breakout

`quantester/strategy/donchian_breakout.py`.

### `DonchianBreakoutStrategy(data_handler, symbol, regime_window=200, entry_window=20, trail_window=10, exit_window=20, atr_window=14, adx_window=14, adx_threshold=25.0, stop_atr_mult=2.0, risk_fraction=0.02, long_only=False)`

SMA-gated Donchian breakout with an ADX intensity filter. Windows are
bar-counts (daily or hourly). The **survivable** configuration from the
Bitstamp study is **daily + `long_only=True`** with a book-level risk budget;
hourly both-sides BTC is friction-dominated (kept as a negative control).

- **Regime gate**: long while `close > SMA(regime_window)`; short while
  `close < SMA(regime_window)` (shorts disabled when `long_only=True`).
- **Entry boundaries**: prior-`entry_window` Donchian channel
  (`max(high_{t-1..t-N})` / `min(low_{t-1..t-N})`) — the signal bar is
  excluded from its own channel.
- **ADX filter**: new entries require `ADX(adx_window) > adx_threshold`.
- **Delay-1**: signals at close T fill at open T+1.
- **Sizing**: emits `SignalEvent.stop_distance = stop_atr_mult × ATR`; wire
  `FractionalRiskSizer(risk_fraction)`. For multi-coin books set
  `risk_fraction = book_budget / N` so concurrent breakouts cannot stack.
- **Exits**: `SMA(exit_window)` mean reversion; opposite Donchian trail;
  protective floor at entry ∓ `stop_atr_mult × ATR` (Kaufman MOC).
- No vectorized twin — validate with Protocol II MCPT / block-bootstrap.

```python
from quantester.strategy.donchian_breakout import DonchianBreakoutStrategy
from quantester.portfolio.portfolio import FractionalRiskSizer, PortfolioManager
from quantester.execution.costs import ConservativeFrictionCostModel

# Multi-coin: one strategy instance per symbol, shared handler + risk budget.
symbols = ["BTC/USD", "ETH/USD", "XRP/USD"]
risk_per_name = 0.02 / len(symbols)
strategies = [
    DonchianBreakoutStrategy(handler, s, long_only=True) for s in symbols
]
portfolio = PortfolioManager(handler, 100_000.0,
                             sizer=FractionalRiskSizer(risk_per_name))
engine = BacktestEngine(handler, strategies, portfolio,
                        SimulatedExecutionHandler(ConservativeFrictionCostModel()))
```

Examples live under [`examples/donchian_breakout/`](../../examples/donchian_breakout/):

| Script | Role |
| --- | --- |
| `run_multi_coin_viz.py` | Daily multi-coin dashboard (recommended) |
| `run_multi_coin.py` | Same book without charts |
| `run_mcpt.py` / `run_viz.py` | Hourly BTC MCPT + charts (negative result) |
| `run.py` / `run_ccxt.py` | Synthetic / hourly CCXT smoke tests |

## LETF dual EMA + Kakushadze Δ

`quantester/strategy/letf_dual_ema.py`.

### `LetfDualEmaDeltaStrategy(data_handler, symbol, fast=10, slow=30, delta=0.02, delay=1)`

Long-only trend follower for a **single x2 leveraged ETF**. Dual EMA golden
cross / death cross plus a daily Kakushadze Δ protective filter — buy-and-hold
is inefficient on LETFs because daily rebalancing creates volatility drag in
chop.

- **Entry**: `EMA(fast)` crosses above `EMA(slow)` (span EMA, `adjust=False`).
- **Trend exit**: `EMA(fast)` crosses below `EMA(slow)`.
- **Δ stop**: while long, if `close_t < (1 − Δ) · close_{t−1}` emit `EXIT`
  (delay=1). Default `Δ = 0.02` for x2. Re-entry requires a **fresh golden
  cross** (not merely `EMA_fast > EMA_slow`).
- **Sizing**: Kaufman — doubled price vol → half size. Wire
  `PercentEquitySizer(letf_equity_fraction(1.0, leverage=2.0))` (50% equity).
- Shares `dual_ema_delta_positions(close, …)` with its vectorized twin.

```python
from quantester.strategy.letf_dual_ema import LetfDualEmaDeltaStrategy
from quantester.portfolio.sizing import letf_equity_fraction
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager

portfolio = PortfolioManager(
    handler, 100_000.0,
    sizer=PercentEquitySizer(letf_equity_fraction(1.0, leverage=2.0)),
)
engine = BacktestEngine(
    handler, LetfDualEmaDeltaStrategy(handler, "ETFBW20LV"),
    portfolio, SimulatedExecutionHandler(CostModel()),
)
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
