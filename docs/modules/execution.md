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
| `LIMIT` | Rests in the ledger until touched: a buy limit fills when the bar's low reaches the level, at `min(open, limit)` — a gap through the limit earns price improvement at the open, never a worse price (sells symmetric at `max(open, limit)`). |
| `MOC` | The bar's **close print**, on its `earliest_fill_time` bar only. Never parks: a missed close auction expires the order (live MOC semantics), so a MOC fill can never use a close print that was not yet known at decision time. |
| `CANCEL` | No fill — synchronously purges the symbol's resting limit/stop orders from the ledger. Parked **market** orders (committed exits/liquidations in flight) are never purged. |

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
| `commission(quantity, price=None)` | cₜ: `fixed + per_share × qty` — charged separately to cash. `price` (the fill reference) is required by notional-fee models. |
| `half_spread(price)` | `price × spread_pct / 2` — the cost of crossing the spread to take liquidity. |
| `kaufman_slippage(price, high, low)` | `price × coef × (high − low)/price` — slippage proportional to the bar's volatility. |
| `kyle_lambda(price, qty, volume, high, low)` | Market impact `dp = λ·dx` with `λ = impact_coef × volatility% / volume`: impact rises with volatility and falls with depth. |
| `adverse_adjustment(price, qty, bar)` | Sum of the three per-share adjustments; this is what gets embedded in `fill_price` (recorded as φₜ = `slippage_cost`). |

### `ConservativeFrictionCostModel(spread_pct=..., fee_rate=..., friction_multiplier=2.0)`

Stressed exchange friction from the tranche-ladder specification:
`C_trade = 2 × (S_bid-ask/2 + μ_fee)` — the FULL bid-ask spread as the
adverse adjustment plus the doubled maker/taker fee on notional, charged on
**every** fill (resting limit orders included; taker-grade friction on maker
fills is deliberately pessimistic). Deterministic, so fast-track parity
holds. Matches Carver's round-trip form `2 × (spread/2 + fee)` per leg
(notebook cross-reference: "it is better to be conservative and assume costs
are higher than you'd hope").

### `RetailCostModel` (OHLCV-only retail friction)

Configurable retail execution without Level-2 data:

```python
from quantester.execution import RetailCostModel, retail_cost_scenario

model = RetailCostModel(
    spread_bps=5.0,
    volatility_slippage_factor=0.1,
    impact_factor=0.1,
    impact_exponent=0.5,
    max_participation_rate=0.05,
)
# Named stress presets: BASE / CONSERVATIVE / STRESS
stress = retail_cost_scenario("STRESS")
```

Adverse adjustment = half-spread + vol-scaled range slippage + participation
impact (`impact_factor × vol_bps × participation ** exponent`). Tiny orders
against deep bars incur negligible impact; oversized orders are clipped by
`max_participation_rate` (default research policy: **partial fills**, with
residuals pending across subsequent bars). `liquidity_policy` on
`SimulatedExecutionHandler` may also be `"reject"` or `"none"` (legacy).

Execution diagnostics (`handler.diagnostics.summary()`) report median/P95/max
participation and impact, plus partial-fill and liquidity-rejection rates.

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
