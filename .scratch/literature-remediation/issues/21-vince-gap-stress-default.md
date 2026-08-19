# Default Vince gap_stress to 1.0

Type: task
Status: open
Part of: [Literature remediation decision map](../map.md)

## Goal

Default \(W\) = raw historical BiggestLoss. Keep `gap_stress` as an opt-in multiplier, default **1.0**.

Ruling: [Keep Vince gap_stress, use realized gap-through W, or drop the 1.5×?](09-vince-gap-stress.md) (notebook D3).

## Files

- `quantester/portfolio/sizing.py` — `optimal_f(..., gap_stress: float = 1.5)`
- `tests/test_portfolio.py` — `test_optimal_f_twr_maximizes` (already compares 1.5 vs 1.0 via explicit `worst_loss`)
- Docs / blueprint lines that say default `gap_stress=1.5`

## Acceptance criteria

- `optimal_f(trades)` with losing trades uses \(W = \min_i \mathrm{Trade}_i\) (gap_stress 1.0).
- `gap_stress=1.5` remains valid and still de-levers \(f^*\) relative to 1.0 (existing comparison).
- Docstring: notebook-verified Vince MoMM ch. 1 raw BiggestLoss; 1.5 was a silent de-lever, not Vince. Gap-through is already a stop-ledger invariant — do not invent a second realized-gap \(W\) series in this ticket.
- No change to stop fill semantics.

## Tests

- Default call equals `optimal_f(trades, gap_stress=1.0)` and equals `worst_loss=trades.min()`.
- `gap_stress=1.5` still produces a smaller \(f^*\) than 1.0 on a mixed win/loss book.

## Out of scope

- Wiring \(f^*\) into `PortfolioManager` (D5 KEEP adjacent).
- Replacing \(W\) with ledger gap-through PnL.
