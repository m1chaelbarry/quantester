# Quantester

Event-driven institutional backtester. This glossary holds terms already locked by engine invariants. Literature-remediation rulings land here only after their wayfinder tickets resolve.

## Language

**Temporal Firewall**:
The two-phase bar contract (`open`, then `close`) that a strategy may observe only data strictly before a fill timestamp.
_Avoid_: hardcoded T+1, look-ahead guard (when used as a synonym for the two-phase rule)

**Delay**:
The number of bars after a signal until the order is eligible to fill, carried as `earliest_fill_time` on the execution ledger.
_Avoid_: lag, latency (unless meaning physical venue delay)

**ETF-trick cash**:
The financing term `c_t` booked outside the spread process `K_t`, typically as a negative dividend at strategy level.
_Avoid_: embedding `c_t` in `K_t`

**Fill price**:
The execution price after spread, slippage, and impact. Cash is charged `qty * fill_price + commission`; `slippage_cost` (`phi_t`) is analytics-only and is never deducted a second time.
_Avoid_: charging slippage twice

**Resting stop**:
A `STOP_ORDER` on the execution ledger that can activate intra-bar and fill at the next available price after gap-through.
_Avoid_: calling a delay-1 `EXIT` signal a stop

**Literature conflict**:
A finding where specialist books disagree, so it is not a defect until a wayfinder ticket rules it.
_Avoid_: treating a conflict row as a bug

**Combined Forecast**:
The single signed forecast formed from weighted EWMAC and Carry Forecast, mapped to one net position in one SignalEvent.
_Avoid_: simultaneous execution, overlay books, dual position

**Carry Forecast**:
The funding-derived forecast in desired-position space (positive = long the perpetual). It has the opposite sign of a positive funding rate.
_Avoid_: adding high funding as a positive position forecast

**Inertia Buffer**:
A rebalance gate: no order unless `|q_target − q_current|` exceeds a fraction of `|q_target|`.
_Avoid_: dead zone, hysteresis (when used as a synonym for this gate)

**Funding Settlement**:
Signed perpetual-funding cash booked on the ledger as ETF-trick cash. Longs pay when the rate is positive.
_Avoid_: dividend (for funding), embedding funding in `K_t`

**Drawdown De-lever**:
Scale target volatility from a drawdown threshold down to zero risk at a drawdown cap, measured peak-to-trough on equity.
_Avoid_: DailyDrawdownBreaker (session-loss flatten; a different overlay)
