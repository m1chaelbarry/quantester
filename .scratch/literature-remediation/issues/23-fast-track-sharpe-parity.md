# Fast-track Sharpe must call annualized_sharpe

Type: task
Status: open
Part of: [Literature remediation decision map](../map.md)
Blocked by: 19, 20

## Goal

One Sharpe function on the same equity series. Fast-track must not keep a private simple-×\(\sqrt{252}\) formula once the tearsheet is simple + measured \(N_T\).

Ruling: [Dual-track: keep two engines, but must fast-track Sharpe match the tearsheet function?](14-dual-track-sharpe-parity.md) (notebook D7). **Must land after** [Switch tearsheet Sharpe to simple returns](19-simple-tearsheet-sharpe.md) and [Annualize with measured periods-per-year](20-measured-periods-per-year.md). Calling pre-D1 `annualized_sharpe` would switch fast-track **to log**.

## Files

- `quantester/montecarlo/fast_track.py` — `FastResult.sharpe`
- Tests that assert fast-track Sharpe values (search `FastResult` / `fast_backtest` in `tests/`)
- `SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md` § fast-track vs event Sharpe (doc only if you touch it)

## Acceptance criteria

- `FastResult.sharpe` = `annualized_sharpe(self.equity)` (no extra `periods` unless the helper inside `annualized_sharpe` already infers \(N_T\)).
- Delete the private `daily_returns.mean()/std() * sqrt(252)` path from the property. `daily_returns` may remain for diagnostics.
- Dual architecture stays: vectorized fast-track still does not implement stops/limits/MOC/delay-0; document that subset contract.
- Event-loop `summarize(equity)["sharpe"]` equals `fast_backtest(...).sharpe` on a delay-1 market-order path where the parity contract already matches equity.

## Tests

- Synthetic equity: `FastResult.sharpe == annualized_sharpe(equity)`.
- Existing fast-track vs event-loop parity test: Sharpe keys match when equity matches.

## Out of scope

- Implementing delay-0 / stops on the fast-track.
- MCPT log-price shuffle.
