# Events & Constants

Module: `quantester/events.py`

Every message that moves through the engine is one of four event types. All
events are dataclasses with two common fields: `type` (a string constant) and
`timestamp` (a `pd.Timestamp`).

## Constants

```python
# Event types
MARKET, SIGNAL, ORDER, FILL = "MARKET", "SIGNAL", "ORDER", "FILL"

# Signal directions
LONG, SHORT, EXIT = "LONG", "SHORT", "EXIT"

# Order directions
BUY, SELL = "BUY", "SELL"

# Order types
MARKET_ORDER = "MARKET"
STOP_ORDER = "STOP"
LIMIT_ORDER = "LIMIT"
MOC_ORDER = "MOC"        # market-on-close: this bar's close print only
CANCEL_ORDER = "CANCEL"  # purge resting limit/stop orders for the symbol

# Bar phases
OPEN, CLOSE = "open", "close"
```

## `MarketEvent`

New bar(s) are available. Posted by the engine twice per timestamp — once per
phase.

| Field | Type | Meaning |
| --- | --- | --- |
| `bars` | `dict[str, pd.Series \| None]` | Symbol → OHLCV row, or `None` when the symbol is untradeable at this timestamp (availability mask). |
| `phase` | `str` | `"open"` or `"close"`; drives which strategies run and what data is visible. |

## `SignalEvent`

Strategy output: a directional intention for one symbol.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `symbol` | `str` | — | Instrument to trade. |
| `signal_type` | `str` | — | `LONG`, `SHORT`, or `EXIT`. |
| `strength` | `float` | `1.0` | Conviction multiplier; scales the sizer's target (used by meta-labeling). |
| `delay` | `int` | `1` | Bars until execution. `1` → fill next bar's open; `0` → fill this bar's open under the intra-bar guard. |
| `fill_at` | `str` | `"open"` | Reference price the portfolio uses for sizing. `"close"` requests a market-on-close fill at the **current** bar's close (close-phase `delay >= 1` strategies only). |
| `limit_price` | `float \| None` | `None` | When set, the portfolio sizes the target **at this price** and rests a `LIMIT` order (tranche ladders priced off latched levels). |
| `cancel_orders` | `bool` | `False` | Emit a `CANCEL` order first (purge the symbol's resting book). Set on exits by strategies that rest orders, so unfilled levels cannot re-enter after an exit. |

## `OrderEvent`

A sized order produced by the portfolio manager.

| Field | Type | Meaning |
| --- | --- | --- |
| `symbol` | `str` | Instrument. |
| `order_type` | `str` | `MARKET`, `STOP`, `LIMIT`, `MOC`, or `CANCEL`. |
| `quantity` | `float` | Always positive; direction is carried separately. |
| `direction` | `str` | `BUY` or `SELL`. |
| `earliest_fill_time` | `pd.Timestamp` | Stamped from the signal's `delay`; the execution ledger will not fill before this time. For `MOC` it must equal the current bar — the fill is that bar's close auction. |
| `stop_price` | `float \| None` | Trigger price for stop orders. |
| `limit_price` | `float \| None` | Resting price for limit orders. |

## `FillEvent`

Execution confirmation; updates the ledger.

| Field | Type | Meaning |
| --- | --- | --- |
| `symbol` | `str` | Instrument. |
| `quantity` | `float` | Filled quantity (positive). |
| `direction` | `str` | `BUY` or `SELL`. |
| `fill_price` | `float` | **All-in** price: reference price ± spread/slippage/impact. |
| `commission` | `float` | cₜ — proportional + fixed costs, charged separately to cash. |
| `slippage_cost` | `float` | φₜ — implementation shortfall in currency. Recorded for analytics, **never** double-charged. |
| `reference_price` | `float` | Pre-cost bar price used for the fill. |
| `total_cost` | property | `commission + slippage_cost`. |

## Example: the life of one trade

```python
# 1. Engine posts at bar T close:
MarketEvent(timestamp=T, bars={"AAA": bar_T}, phase="close")

# 2. Strategy responds:
SignalEvent(timestamp=T, symbol="AAA", signal_type="LONG",
            strength=1.0, delay=1)

# 3. Portfolio sizes it and stamps the firewall time (T+1):
OrderEvent(timestamp=T, symbol="AAA", order_type="MARKET",
           quantity=42.0, direction="BUY", earliest_fill_time=T_plus_1)

# 4. Execution fills at T+1's open, adjusted for costs:
FillEvent(timestamp=T_plus_1, symbol="AAA", quantity=42.0, direction="BUY",
          fill_price=101.32, commission=1.21, slippage_cost=1.68,
          reference_price=101.28)
```
