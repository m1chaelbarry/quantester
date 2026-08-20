# Size live sizers on cash, not MTM equity

Type: task
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Goal

Stop sizing off mark-to-market \(E\) (procyclical). Live sizers use cash (optionally smoothed). Vol-target / Kelly / \(f^*\) stay **libraries**, not the default fill hook.

Ruling: notebook D10 (Carver *Systematic Trading* ch. 10) plus [Live allocation: Vince LSM, AFML covariance library, or both with an explicit live-sizer policy?](10-live-allocation-philosophy.md) (D5 KEEP adjacent). D10 did not pick “cash-only vs a vol-target formula”; this ticket picks **cash (optional EWMA) as the live base** and leaves vol-target in `sizing.py`.

## Files

- `quantester/portfolio/sizers.py` — `PercentEquitySizer`, `FractionalRiskSizer`, `HedgeRatioSizer` all read `portfolio.equity`
- `quantester/portfolio/portfolio.py` — default `PercentEquitySizer(0.5)`
- `quantester/simple.py` — `PercentEquitySizer(0.9)`
- `tests/test_portfolio.py` — percent-equity quantity vs equity
- Docs: `docs/tutorials/creating-a-strategy.md`

## Acceptance criteria

- Add `base: str = "cash"` on the three MTM sizers (`"cash"` | `"equity"`). Default **cash**. `"equity"` is the explicit procyclical opt-in.
- Optional `cash_ewma_span: int | None = None`: when set, size off EWMA of `portfolio.cash` (Carver-style smoothing); when None, use raw `portfolio.cash`.
- `FixedUnitSizer` unchanged.
- Do **not** wire Ledoit–Wolf, HRP, Vince LSM, or `optimal_f` into `PortfolioManager`.
- Keep class names for API stability; docstrings must say the default base is cash, not equity.
- Negative / zero cash → target 0 (no silent fall-back onto equity).

## Tests

- Same signal: rising MTM equity with flat cash does **not** increase target quantity on default base.
- `base="equity"` reproduces today’s `pct * equity / price` numbers.
- `cash_ewma_span` is slower than raw cash on a step change in cash (unit-level, no need for a full backtest).
- Hedge-ratio relationship \(q_X = -\beta q_Y\) still holds on the cash base.

## Out of scope

- Full Carver FDM / inertia / vol-target live stack (parked specialty).
- Renaming `PercentEquitySizer` (optional follow-on).

## Answer

Shipped (commit ada9465). `PercentEquitySizer` / `FractionalRiskSizer` / `HedgeRatioSizer` take `base="cash"` (default) | `"equity"` plus optional `cash_ewma_span` (one cash observation per signal timestamp; reused sizers restart on time travel). Non-positive cash targets 0, never a fall-back to equity. `FixedUnitSizer` unchanged; class names kept; Ledoit-Wolf/HRP/LSM/f* stay library-only (D5).
