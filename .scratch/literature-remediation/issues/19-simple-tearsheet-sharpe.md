# Switch tearsheet Sharpe to simple returns

Type: task
Status: open
Part of: [Literature remediation decision map](../map.md)

## Goal

Make `annualized_sharpe` (and everything that feeds DSR from a tearsheet) use simple returns \(r_t = E_t/E_{t-1}-1\), so Carver cost drag stays linear in Sharpe units.

Ruling: [Which Sharpe representation is canonical for tearsheet, MCPT, and DSR?](05-canonical-sharpe.md) (notebook D1).

## Files

- `quantester/analytics/performance.py` — `annualized_sharpe` / `log_returns` / `summarize`; module docstring
- `quantester/analytics/trials_registry.py` — `auto_register_from_equity` currently takes **log** moments
- `quantester/analytics/dsr.py` — docstring contract only (already accepts whatever SR is passed)
- `quantester/analytics/returns.py` — prefer `simple_returns_from_equity` as the Sharpe path
- `tests/test_analytics.py` — `test_sharpe_manual_annualization` currently rebuilds log Sharpe
- Docs that claim “Sharpe from log returns”: `docs/` tearsheet / performance pages if they echo the old docstring

Keep unchanged: `quantester/visualization/static.py` rolling Sharpe (already simple, `ddof=1`); MCPT log-price shuffle in `quantester/montecarlo/permutation.py` (documented Masters exception).

## Acceptance criteria

- `annualized_sharpe(equity)` = \((\bar r - R_f)/\sigma_r \times \sqrt{P}\) with \(r_t = E_t/E_{t-1}-1\), sample std matching today’s `rets.std()` convention unless tests force `ddof`.
- `log_returns` remains available; it is not the tearsheet default.
- `auto_register_from_equity` stores `sharpe` from `annualized_sharpe` and stores **simple** mean/std/skew/kurtosis for DSR moments (Pearson kurtosis still).
- Module docstring marks D1 as notebook-verified (Carver *Systematic Trading* ch. 12/15). Masters log = documented exception, not engine default.
- Do not implement [Fast-track Sharpe must call annualized_sharpe](23-fast-track-sharpe-parity.md) in this ticket — that is blocked on this landing plus measured \(N_T\).

## Tests

- Rewrite `test_sharpe_manual_annualization` onto simple `pct_change` equity.
- Registry: `auto_register_from_equity` Sharpe equals `annualized_sharpe` on the same curve; moments are simple, not log.
- A path where simple and log SR differ (large moves) proves the tearsheet moved.

## Out of scope

- Changing MCPT shuffle representation.
- Fast-track `FastResult.sharpe` (ticket 23).
- `periods` default / measured \(N_T\) (ticket 20).
