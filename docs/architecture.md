# Architecture & Core Concepts

Quantester is an **event-driven** backtester. Instead of a script that loops
over rows and trades directly, the system is five decoupled components that
only talk to each other by posting events on a shared queue. This page
explains the moving parts; you do not need to memorize it to write a strategy,
but understanding the lifecycle will save you from subtle mistakes later.

## The five components

| Component | Responsibility | Base class |
| --- | --- | --- |
| Data Handler | Streams point-in-time OHLCV bars; enforces what is *visible* at each moment. | `quantester.data.base.DataHandler` |
| Strategy | Consumes market events, produces signals (`LONG` / `SHORT` / `EXIT`). | `quantester.strategy.base.Strategy` |
| Portfolio Manager | Turns signals into sized orders; keeps the cash/holdings ledger; risk overlays and margin. | `quantester.portfolio.base.Portfolio` |
| Execution Simulator | Fills eligible orders at realistic, cost-adjusted prices. | `quantester.execution.base.ExecutionHandler` |
| Analytics | Offline: performance stats, tearsheets, DSR, trials registry (not wired into the loop). | `quantester.analytics.*` |

The components never call each other directly. They communicate through four
event types on one synchronous queue:

```
MarketEvent  →  SignalEvent  →  OrderEvent  →  FillEvent
 (new bars)     (strategy)      (portfolio)    (execution)
```

## The event loop, bar by bar

`BacktestEngine.run_backtest()` (`quantester/engine.py`) is an outer loop over
the data stream with an inner loop that drains the event queue:

```
prime_data()
while data remains:
    timestamp, bars = data_handler.advance()

    # ---- OPEN phase ----
    data_handler.set_phase("open", timestamp)
    put MarketEvent(phase="open")
    drain queue:
        execution handler drains its pending-order ledger first
        delay=0 strategies calculate signals

    # ---- CLOSE phase ----
    data_handler.set_phase("close", timestamp)
    put MarketEvent(phase="close")
    drain queue:
        portfolio marks equity to market (and runs the margin monitor)
        delay>=1 strategies calculate signals
```

Inside the drain, each event type is routed:

| Event | Routed to |
| --- | --- |
| `MarketEvent` (open phase) | `ExecutionHandler.on_market`, then delay-0 strategies |
| `MarketEvent` (close phase) | `Portfolio.update_portfolio_valuation`, then delay-≥1 strategies |
| `SignalEvent` | `Portfolio.update_from_signal` → emits `OrderEvent` |
| `OrderEvent` | `ExecutionHandler.execute_order` → emits `FillEvent` (or parks it) |
| `FillEvent` | `Portfolio.update_from_fill` → updates the ledger |

Because events posted during a drain are themselves processed before the phase
ends, a full `Market → Signal → Order → Fill` cascade completes within one
phase when the order is immediately eligible.

## The State-Based Temporal Firewall

This is the engine's defining guarantee: **a backtest can never see the
future**, and it is enforced by mechanism rather than by convention.

### Two phases per bar

Every bar is processed twice — once at its `open`, once at its `close`. During
the open phase the DataHandler exposes:

- all bars **strictly before** the current bar, and
- the current bar's **open print only** (`get_current_open`).

During the close phase the full current bar becomes visible. A strategy can
therefore never peek at the current close while trading at the current open.

### `delay` and `earliest_fill_time`

Every strategy declares a `delay` — how many bars must pass before its orders
may execute:

| `delay` | Signal computed at | Fills at | Guard |
| --- | --- | --- | --- |
| `1` (default) | close of bar T | **open of bar T+1** | Classic T+1; the simplest safe choice. |
| `0` | open of bar T | **open of bar T** | Intra-bar guard: the strategy only sees data strictly before the fill timestamp (bars ≤ T−1 plus T's open). |

Mechanically, the portfolio stamps each `OrderEvent` with an
`earliest_fill_time` looked up on the DataHandler's master calendar
(`timestamp_at_offset(signal.timestamp, delay)`). The execution handler keeps a
**pending-order ledger**: an order whose earliest fill time has not arrived is
parked and re-evaluated on every subsequent open phase. An order can never be
filled before its stamped time, no matter what a strategy does.

### Availability masks (no silent history rewrites)

Multi-symbol data is aligned on the **outer join** of all symbols' timestamps.
If symbol `BBB` has no bar at a timestamp where `AAA` does, `BBB` is
**untradeable** at that timestamp (`event.bars["BBB"] is None`) — the timestamp
is kept, not deleted. Dropping incomplete bars would erase exactly the
high-stress, illiquid periods a backtest must survive, inducing selection
bias.

## Ledger accounting rules

The portfolio ledger follows two rules you must know when reading results:

1. **`fill_price` is all-in.** The execution handler embeds spread crossing,
   volatility slippage, and market impact into the fill price. Cash is charged
   `qty * fill_price + commission`.
2. **`slippage_cost` (φₜ) is informational.** It is recorded on the
   `FillEvent` for cost analytics but **never** deducted from cash again —
   that would double-count it. Commissions (cₜ) are the only separately
   charged cost.

Completed round-trips are booked into `portfolio.trades` with entry/exit
prices and realized PnL; open lots are tracked at volume-weighted average
price.

## Why event-driven?

- **Research-to-live parity.** The same interfaces that consume simulated
  fills today can consume a broker's fill messages tomorrow; the strategy code
  does not change.
- **Testability.** Each component is isolated behind a narrow interface with
  its own test suite (`tests/`).
- **Honesty.** The firewall, the pending-order ledger, and the availability
  masks make the most common backtest cheats (look-ahead, perfect fills,
  erased gaps) structurally impossible rather than merely discouraged.

## Extending the engine

- **New data feeds:** subclass `DataHandler` and honor the firewall contract
  exactly — phase-aware `get_latest_bars`, `bar_at` for execution, and
  `timestamp_at_offset` for order stamping. See [Data](modules/data.md).
- **New cost models:** add deterministic functions of `(order, bar)` to
  `execution/costs.py`. No randomness — the event engine and the Monte Carlo
  fast-track must share identical cost semantics, and determinism is
  reproducibility.
- **New sizers:** any callable `(signal, portfolio, ref_price) -> target_qty`,
  wired via `PortfolioManager(sizer=...)`. See
  [Portfolio, Sizing & Risk](modules/portfolio.md).
- **Any change to fill semantics** must keep `tests/test_montecarlo.py`'s
  fast-track parity test green — it is the guard against silent divergence
  between the event engine and the vectorized Monte Carlo path.
