# Unadjusted Yahoo prices plus dividend cash

Type: task
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Goal

Default research path is an unadjusted price ledger with dividends booked as cash. Splits adjust quantity, not fill prices via Yahoo `auto_adjust`.

Ruling: [Adjusted total-return prices versus unadjusted plus corporate-action cash ledger?](13-adjusted-vs-ca-cash.md) (notebook D9). PIT universe is **out**: [Is point-in-time universe plus delist in the first-wave spec?](16-pit-first-wave.md).

## Files

- `quantester/data/yfinance_handler.py` — `auto_adjust: bool = True`
- `quantester/simple.py` — `load_yahoo` / `run_backtest` `auto_adjust=True`
- `quantester/events.py` — no CA event type today; add a cash-only dividend event **or** credit cash inside the data handler/portfolio on known dividend timestamps without a fifth lifecycle type (prefer a `CashEvent` / `CorporateActionEvent` that the engine routes to `PortfolioManager`, not a bypass of the queue)
- `quantester/engine.py` — dispatch
- `quantester/portfolio/portfolio.py` — cash credit; split quantity adjustment
- `tests/test_data_providers.py` — mocks `auto_adjust=True`
- `tests/test_free_data_sources.py` / `tests/test_simple_api.py` if they pin the flag

## Acceptance criteria

- Default `auto_adjust=False` on Yahoo loaders. `auto_adjust=True` remains a documented total-return ranking mode, not the default fill path.
- Dividend: cash increases by `shares * dividend_per_share` on the ex-date (long) / decreases when short; **do not** also leave the dividend inside close. Document as Peterson ch. 11 cash booking.
- Split: position quantity adjusts by the split ratio; OHLC stays raw. Fills after the split use post-split raw prices.
- ETF-trick \(c_t\) stays **external** to \(K_t\) — do not embed carry in the synthetic spread. Dividend cash is the cash ledger, not `K_t`.
- No PIT constituent file, no delist residual, no vendor schema invention.

## Tests

- Loader default forwards `auto_adjust=False`.
- Synthetic: 100 shares, $1 dividend → cash +100, close unchanged, equity +100 vs a no-dividend twin.
- Synthetic 2-for-1 split: quantity doubles, raw price halves, equity continuous aside from explicit CA cash.
- `auto_adjust=True` path still loads (ranking mode) and does **not** double-book dividends as cash.

## Out of scope

- Point-in-time membership / delist / halt.
- Borrow/short-locate fees beyond the existing documented simplification.
- Changing idle-cash yield `/365`.

## Answer

Shipped (commit c702f5c). `auto_adjust=False` is the default on `YFinanceDataHandler` and `load_yahoo` (True remains a documented total-return ranking mode and suppresses CA events, so no double-booking). New `CorporateActionEvent` (dividend / split) routes through the queue: `StreamingDataHandler(corporate_actions=...)` or `set_corporate_actions`; the engine drains CA events at the ex-date bar's open before fills/valuation; `PortfolioManager.update_from_corporate_action` books dividend cash (shorts pay) and split quantity (lot avg price divided by the ratio, P&L-continuous). PIT universe/delist stays out (ticket 16).
