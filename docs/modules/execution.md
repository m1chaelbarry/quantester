# Execution & Transaction Costs

Package: `quantester/execution`

The execution layer simulates the physical realities of trading: orders wait
for their eligible time, fills happen at real bar prices, and every fill pays
spread, slippage, impact, and commission. In production the same
`ExecutionHandler` interface would route to a live broker API — only the
implementation changes.

## `SimulatedExecutionHandler`

`quantester/execution/simulator.py`.

```python
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.execution.costs import CostModel

execution = SimulatedExecutionHandler(CostModel())   # or CostModel(...) tuned
```

### The pending-order ledger

The handler keeps a ledger of parked orders and enforces the temporal
firewall mechanically:

- `execute_order` fills the order immediately only if
  `earliest_fill_time <= current bar time`; otherwise the order is parked.
- On every **open-phase** `MarketEvent`, parked orders whose
  `earliest_fill_time` has arrived are retried.
- An order on a symbol with no bar that timestamp stays parked — no fill
  without a bar.

### Fill rules

| Order type | Fill price reference |
| --- | --- |
| `MARKET` | The current bar's **open**, plus the adverse cost adjustment. |
| `STOP` | The **next available price after the stop is touched**, including gap-through: a buy stop gapped at the open fills *at the open*, not at the stop. Untriggered stops stay pending. |

The fill price is then adjusted **adversely** (up for buys, down for sells,
floored at ~0): you always pay the adjustment. Fill details — all-in
`fill_price`, `commission`, `slippage_cost`, and the pre-cost
`reference_price` — are recorded on the `FillEvent` and in
`execution.fills`.

> **Why imperfect stops matter:** guaranteeing fills at the stop price
> understates tail risk and silently unbounds optimal-f position sizing. The
> gap-through rule is a safety feature, not a nuisance.

## `CostModel`

`quantester/execution/costs.py` — a dataclass of five knobs, all
**deterministic** functions of the order and the bar. Determinism is a hard
requirement: the event engine and the Monte Carlo fast-track share these
functions, and a parity test proves they agree. (Kyle's noise-trader flow is
deliberately not simulated — it is unobservable from historical bars and
would inject non-reproducible randomness.)

```python
@dataclass
class CostModel:
    fixed_commission: float = 1.0        # currency per order
    per_share_commission: float = 0.005  # currency per share
    spread_pct: float = 0.0005           # full bid-ask spread, fraction of price
    slippage_vol_coef: float = 0.1       # Kaufman coefficient on bar range %
    impact_coef: float = 0.1             # Kyle lambda scale (Amihud-style)
```

| Method | Computes |
| --- | --- |
| `commission(quantity)` | cₜ: `fixed + per_share × qty` — charged separately to cash. |
| `half_spread(price)` | `price × spread_pct / 2` — the cost of crossing the spread to take liquidity. |
| `kaufman_slippage(price, high, low)` | `price × coef × (high − low)/price` — slippage proportional to the bar's volatility. |
| `kyle_lambda(price, qty, volume, high, low)` | Market impact `dp = λ·dx` with `λ = impact_coef × volatility% / volume`: impact rises with volatility and falls with depth. |
| `adverse_adjustment(price, qty, bar)` | Sum of the three per-share adjustments; this is what gets embedded in `fill_price` (recorded as φₜ = `slippage_cost`). |

### Cost accounting in one line

`fill_price` = reference ± adverse adjustment (all-in) → cash pays
`qty × fill_price + commission`; `slippage_cost` is kept for analytics only.
See the
[architecture page](../architecture.md#ledger-accounting-rules).

### Adding a cost model

Add deterministic methods to `CostModel` (or extend `adverse_adjustment`) —
and nothing else. Because `montecarlo/fast_track.py` calls the same methods,
both engines stay in parity automatically. Any change to fill semantics must
keep `tests/test_montecarlo.py`'s parity test green.

## `ExecutionHandler` (interface)

`quantester/execution/base.py` — two methods: `on_market(market_event, queue)`
(drain the ledger on open phase) and `execute_order(order, queue)` (fill or
park). Implement this interface to connect a live broker; the rest of the
engine does not change.
