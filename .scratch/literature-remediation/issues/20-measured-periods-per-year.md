# Annualize with measured periods-per-year

Type: task
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Goal

Stop applying `TRADING_DAYS = 252` to every index. Metrics annualize with the series’ measured \(N_T\). Cash yield stays `/365`.

Ruling: [What is the canonical periods-per-year and cash day-count policy?](06-periods-per-year.md) (notebook D2; fact in [What does Carver prescribe for 256 business days versus measured frequency?](03-carver-256.md)).

## Files

- `quantester/analytics/performance.py` — `TRADING_DAYS`, `annualized_sharpe(..., periods=)`, `calmar_ratio`, `summarize`
- `quantester/analytics/dsr.py` — `periods_per_year: float = 252.0` defaults
- `quantester/montecarlo/fast_track.py` — hardcoded `np.sqrt(252)` in `FastResult.sharpe` (do not “fix” Sharpe representation here; ticket 23 owns the call)
- `quantester/visualization/static.py` — rolling Sharpe × \(\sqrt{252}\)
- `quantester/portfolio/portfolio.py` — `_accrue_cash_yield` `days / 365.0` **KEEP**
- Call sites that pass `periods=365` for crypto (examples) should use the helper instead of a magic constant where the index can speak

## Acceptance criteria

- Add `measured_periods_per_year(index) -> float`: \(N / ((t_N - t_0)/\text{1 year})\) on a DatetimeIndex with \(N\ge 2\) observations (Chan: measured frequency, not Carver 256, not median-\(\Delta t\) smear). Document the formula in the docstring.
- `summarize(equity)` and `annualized_sharpe(equity)` without an explicit `periods` use that helper when the index is datetime; keep `periods=` as an override.
- `TRADING_DAYS = 252` may remain a **fallback** for non-datetime indexes, not a silent daily assumption on hourly/crypto series.
- Cash yield day-count remains 365 simple. Do not reuse `measured_periods_per_year` for financing.
- Carver 256 is documented as a convenience, not wired as the default.

## Tests

- Daily `bdate_range` of one calendar year → \(N_T\) near 252, not 256.
- Hourly series spanning a known window → \(N_T\) near Chan’s clock, not 252.
- Cash-yield unit test still compounds `/365` if one exists; add one if not.
- Explicit `periods=256` still works as an override for Carver-style research.

## Out of scope

- Switching Sharpe from log to simple (ticket 19).
- Fast-track calling `annualized_sharpe` (ticket 23) — but if you touch `FastResult.sharpe`’s 252, leave a `# see ticket 23` comment rather than a private formula.

## Answer

Shipped (commit f603a80). `measured_periods_per_year(index)` = N / span-in-365.25-years (Chan measured frequency; Carver 256 documented as convenience, never a default). `annualized_sharpe` / `calmar_ratio` / `summarize` / `plot_rolling_metrics` default `periods_per_year=None` -> measured on datetime indexes, explicit override wins, `TRADING_DAYS = 252` is the non-datetime fallback. Example call sites dropped their magic 365 constants. Cash yield stays `/365` (untouched).
