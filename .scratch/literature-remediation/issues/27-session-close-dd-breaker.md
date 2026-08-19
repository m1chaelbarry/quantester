# Session-close drawdown breaker

Type: task
Status: open
Part of: [Literature remediation decision map](../map.md)

## Goal

Trip the daily loss breaker at **session close**, not UTC-midnight `.date()` rollover. Parameterize the session roll.

Ruling: notebook D11 (Harris ch. 22). First-wave leftover from [Which non-controversial critical defects are must-fix on the spec?](17-noncontroversial-must-fix.md).

## Files

- `quantester/portfolio/risk.py` — `DailyDrawdownBreaker.update` uses `pd.Timestamp(timestamp).date()`
- `quantester/portfolio/portfolio.py` — close-phase `drawdown_breaker.update`
- `tests/test_portfolio.py` — `test_breaker_threshold_and_rollover_reset`, `test_breaker_liquidates_cancels_and_blocks_entries`
- `quantester/strategy/tranche_pullback.py` — comments about 4.5% daily breaker

## Acceptance criteria

- Add `day_roll_time` (a `datetime.time`, default session close **16:00**) and `tz` (default `America/New_York`). Session id = the trading date of `timestamp` in that zone, with the roll at `day_roll_time`, not naive UTC midnight.
- For **daily** bars, the bar’s close **is** session close: existing daily tests should still trip on that close print.
- For **intraday** UTC timestamps of US equities, a 00:00 UTC date change must **not** reset the halt / baseline; the reset happens at the configured roll.
- Trip evaluation for the daily limit is the session-close valuation (close-phase at/after roll). Intraday MTM may still *observe* drawdown, but the product rule is Harris session-close: do not treat a 00:00 UTC print as a new “day.”
- Keep 4.5% default threshold and liquidate/cancel/block-entry behavior.

## Tests

- Daily bars: existing 4.5% trip + next-session reset still pass.
- Intraday: timestamps `2024-01-02 23:00 UTC` and `2024-01-03 01:00 UTC` are the **same** NY session; breaker must not reset at 00:00 UTC.
- After 16:00 America/New_York, the next bar belongs to the new session baseline.

## Out of scope

- Full exchange holiday calendar (note the fog: no first-class session calendar today; `day_roll_time` + tz is the first-wave substitute).
- Changing the 4.5% threshold.
